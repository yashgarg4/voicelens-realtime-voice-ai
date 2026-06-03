import { useCallback, useRef, useState } from "react";
import type {
  ConnectionStatus,
  Exchange,
  FeedbackScores,
  ServerMessage,
} from "../types";

// Audio format constants (must match backend/config.py exactly) 
const SAMPLE_RATE_IN = 16_000; // mic -> Gemini
const SAMPLE_RATE_OUT = 24_000; // Gemini -> speakers
const CAPTURE_BUFFER_SIZE = 4096; // ScriptProcessor frame size (samples)

// Reconnection: up to 3 retries with exponential backoff (0.5s, 1s, 2s).
const MAX_RECONNECTS = 3;
const RECONNECT_BASE_MS = 500;
const OPEN_TIMEOUT_MS = 8000;

// Dev server proxies /ws to the FastAPI backend (see vite.config.ts).
function wsUrl(questionId?: number, sessionId?: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams();
  if (questionId != null) params.set("question_id", String(questionId));
  if (sessionId) params.set("session_id", sessionId);
  const qs = params.toString();
  return `${proto}://${location.host}/ws/session${qs ? `?${qs}` : ""}`;
}

// Clamp + scale Float32 [-1,1] samples to 16-bit signed PCM.
function floatTo16BitPCM(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// base64 (24 kHz PCM16) -> Float32 [-1,1] for Web Audio playback.
function base64ToFloat32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  // Platforms Vite targets are little-endian, matching Gemini's PCM16 LE.
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 0x8000;
  return float32;
}

export interface VoiceStreamState {
  status: ConnectionStatus;
  error: string | null;
  latestUser: string; // your most recent utterance (Gemini transcript)
  finetunedTranscript: string; // same answer re-transcribed by the QLoRA Whisper
  usingAdapter: boolean; // whether that transcript used the fine-tuned adapter
  latestCoach: string; // the coach's most recent / in-progress reply
  history: Exchange[]; // every completed round
  feedback: FeedbackScores | null; // latest parsed scores
  micLevel: number; // 0..1, for the mic-button visualiser
  start: (questionId?: number, sessionId?: string) => Promise<void>;
  stop: () => void;
}

export function useVoiceStream(): VoiceStreamState {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [latestUser, setLatestUser] = useState("");
  const [finetunedTranscript, setFinetunedTranscript] = useState("");
  const [usingAdapter, setUsingAdapter] = useState(false);
  const [latestCoach, setLatestCoach] = useState("");
  const [history, setHistory] = useState<Exchange[]>([]);
  const [feedback, setFeedback] = useState<FeedbackScores | null>(null);
  const [micLevel, setMicLevel] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  // Playback: a dedicated context + a scheduling cursor so chunks play back
  // gap-free, and a live list of sources so we can flush on interruption.
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextStartRef = useRef(0);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);

  // Turn / transcript bookkeeping.
  //  - the mic streams continuously (barge-in enabled), so the user can talk
  //    over the coach; Gemini's server-side VAD detects the interruption and
  //    emits an `interrupted` event, which flushes queued playback.
  //  - the buffers accumulate the *current* round's text; roundCompleteRef
  //    triggers a lazy reset when the next round's first text arrives, so the
  //    "latest" boxes keep showing the last reply until a new one begins.
  const userBufRef = useRef("");
  const coachBufRef = useRef("");
  const roundCompleteRef = useRef(false);
  const exchangeIdRef = useRef(0);

  // Reconnection bookkeeping.
  const shouldReconnectRef = useRef(false); // false once the user stops / fatal error
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const questionIdRef = useRef<number | undefined>(undefined);
  const sessionIdRef = useRef<string | undefined>(undefined);

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    if (openTimerRef.current) clearTimeout(openTimerRef.current);
    reconnectTimerRef.current = null;
    openTimerRef.current = null;
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    captureCtxRef.current?.close().catch(() => {});
    playbackCtxRef.current?.close().catch(() => {});
    processorRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    captureCtxRef.current = null;
    playbackCtxRef.current = null;
    nextStartRef.current = 0;
    activeSourcesRef.current = [];
    setMicLevel(0);
  }, []);

  const flushPlayback = useCallback(() => {
    // Barge-in: stop everything Gemini queued and reset the cursor.
    for (const src of activeSourcesRef.current) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    activeSourcesRef.current = [];
    if (playbackCtxRef.current) {
      nextStartRef.current = playbackCtxRef.current.currentTime;
    }
  }, []);

  const enqueuePlayback = useCallback((float32: Float32Array) => {
    const ctx = playbackCtxRef.current;
    if (!ctx || float32.length === 0) return;
    // AudioBuffer carries its own sampleRate (24 kHz); the source node
    // resamples to the context rate automatically on playback.
    const buffer = ctx.createBuffer(1, float32.length, SAMPLE_RATE_OUT);
    buffer.getChannelData(0).set(float32);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, nextStartRef.current);
    src.start(startAt);
    nextStartRef.current = startAt + buffer.duration;

    activeSourcesRef.current.push(src);
    src.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== src);
    };
  }, []);

  // Lazily clear the current round once the previous one is finished, so the
  // "You said" / "Coach said" boxes show the latest text until it's replaced.
  const startNewRoundIfNeeded = useCallback(() => {
    if (roundCompleteRef.current) {
      userBufRef.current = "";
      coachBufRef.current = "";
      setLatestUser("");
      setLatestCoach("");
      setFinetunedTranscript("");
      roundCompleteRef.current = false;
    }
  }, []);

  const handleMessage = useCallback(
    (msg: ServerMessage) => {
      switch (msg.type) {
        case "status":
          // Connected; the coach greets first (mic stays gated until then).
          setStatus("connecting");
          break;
        case "error":
          // Server-reported error (e.g. bad API key) is fatal — don't retry.
          shouldReconnectRef.current = false;
          setError(msg.message);
          setStatus("error");
          break;
        case "feedback":
          setFeedback(msg.scores);
          break;
        case "finetuned_transcript":
          setFinetunedTranscript(msg.text);
          setUsingAdapter(msg.using_adapter);
          break;
        case "audio":
          setStatus("speaking");
          enqueuePlayback(base64ToFloat32(msg.data));
          break;
        case "input_transcript":
          startNewRoundIfNeeded();
          userBufRef.current += msg.text;
          setLatestUser(userBufRef.current);
          break;
        case "output_transcript":
          startNewRoundIfNeeded();
          setStatus("speaking");
          coachBufRef.current += msg.text;
          setLatestCoach(coachBufRef.current);
          break;
        case "interrupted":
          // Barge-in: the user spoke over the coach. Drop queued audio.
          flushPlayback();
          setStatus("listening");
          break;
        case "turn_complete": {
          // Round finished: archive it and hand the floor back to the user.
          const user = userBufRef.current.trim();
          const coach = coachBufRef.current.trim();
          if (user || coach) {
            const entry: Exchange = {
              id: ++exchangeIdRef.current,
              user,
              coach,
              at: Date.now(),
            };
            setHistory((prev) => [...prev, entry]);
          }
          roundCompleteRef.current = true;
          setStatus("listening");
          break;
        }
        default:
          break;
      }
    },
    [enqueuePlayback, flushPlayback, startNewRoundIfNeeded]
  );

  // Open (or re-open) the WebSocket. The mic pipeline is set up once in start()
  // and survives reconnects — only the socket is recreated here.
  const connectSocket = useCallback(() => {
    const ws = new WebSocket(
      wsUrl(questionIdRef.current, sessionIdRef.current)
    );
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    if (openTimerRef.current) clearTimeout(openTimerRef.current);
    openTimerRef.current = setTimeout(() => {
      // Stuck in CONNECTING → force-close so onclose triggers a retry.
      if (ws.readyState !== WebSocket.OPEN) ws.close();
    }, OPEN_TIMEOUT_MS);

    ws.onopen = () => {
      if (openTimerRef.current) clearTimeout(openTimerRef.current);
      reconnectAttemptsRef.current = 0; // a clean connection resets the budget
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        handleMessage(JSON.parse(ev.data) as ServerMessage);
      }
    };
    ws.onerror = () => {
      /* let onclose drive reconnect/teardown */
    };
    ws.onclose = () => {
      if (openTimerRef.current) clearTimeout(openTimerRef.current);
      if (!shouldReconnectRef.current) {
        setStatus((s) => (s === "error" ? s : "closed"));
        return;
      }
      if (reconnectAttemptsRef.current >= MAX_RECONNECTS) {
        shouldReconnectRef.current = false;
        setError(
          `Connection lost — reconnection failed after ${MAX_RECONNECTS} attempts.`
        );
        setStatus("error");
        cleanup();
        return;
      }
      const attempt = reconnectAttemptsRef.current++;
      const delay = RECONNECT_BASE_MS * 2 ** attempt;
      setStatus("reconnecting");
      reconnectTimerRef.current = setTimeout(connectSocket, delay);
    };
  }, [handleMessage, cleanup]);

  const start = useCallback(
    async (questionId?: number, sessionId?: string) => {
      if (["connecting", "reconnecting", "listening", "speaking"].includes(status))
        return;
      setError(null);
      setStatus("connecting");
      setLatestUser("");
      setLatestCoach("");
      setFinetunedTranscript("");
      setHistory([]);
      setFeedback(null);
      userBufRef.current = "";
      coachBufRef.current = "";
      roundCompleteRef.current = false;
      questionIdRef.current = questionId;
      sessionIdRef.current = sessionId;
      shouldReconnectRef.current = true;
      reconnectAttemptsRef.current = 0;

      try {
        // 1) Mic capture at 16 kHz mono. Specifying sampleRate on the
        //    AudioContext makes the browser deliver 16 kHz frames directly, so
        //    no client-side downsampling is needed before sending to Gemini.
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: SAMPLE_RATE_IN,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
        streamRef.current = stream;

        const captureCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
        await captureCtx.resume();
        captureCtxRef.current = captureCtx;

        const playbackCtx = new AudioContext();
        await playbackCtx.resume();
        playbackCtxRef.current = playbackCtx;
        nextStartRef.current = playbackCtx.currentTime;

        // 2) Wire mic -> Int16 -> current WebSocket via ScriptProcessorNode.
        //    onaudioprocess reads wsRef.current so it follows reconnects.
        const source = captureCtx.createMediaStreamSource(stream);
        sourceRef.current = source;
        const processor = captureCtx.createScriptProcessor(
          CAPTURE_BUFFER_SIZE,
          1,
          1
        );
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);

          // Cheap RMS level for the mic-button visualiser.
          let sum = 0;
          for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
          setMicLevel(Math.min(1, Math.sqrt(sum / input.length) * 4));

          // Stream the mic continuously so the user can interrupt (barge-in).
          const sock = wsRef.current;
          if (sock && sock.readyState === WebSocket.OPEN) {
            sock.send(floatTo16BitPCM(input).buffer);
          }
        };

        source.connect(processor);
        // ScriptProcessor only fires while connected to the graph; route it to a
        // muted gain node so it ticks without echoing the mic to the speakers.
        const sink = captureCtx.createGain();
        sink.gain.value = 0;
        processor.connect(sink);
        sink.connect(captureCtx.destination);

        // 3) Open the socket (reconnects handled in connectSocket's onclose).
        connectSocket();
      } catch (err) {
        shouldReconnectRef.current = false;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
        cleanup();
      }
    },
    [status, connectSocket, cleanup]
  );

  const stop = useCallback(() => {
    shouldReconnectRef.current = false; // user-initiated: do not reconnect
    wsRef.current?.close();
    wsRef.current = null;
    cleanup();
    setStatus("idle");
  }, [cleanup]);

  return {
    status,
    error,
    latestUser,
    finetunedTranscript,
    usingAdapter,
    latestCoach,
    history,
    feedback,
    micLevel,
    start,
    stop,
  };
}
