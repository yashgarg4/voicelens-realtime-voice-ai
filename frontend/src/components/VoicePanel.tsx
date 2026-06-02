import type { ConnectionStatus, Exchange } from "../types";
import { ChatHistory } from "./ChatHistory";

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  idle: "Tap the mic to start the interview",
  connecting: "Connecting — the coach will ask the question…",
  reconnecting: "Connection dropped — reconnecting…",
  speaking: "Coach is speaking — talk any time to interrupt",
  listening: "Listening — your turn to answer",
  error: "Something went wrong",
  closed: "Session ended",
};

const STATUS_COLOR: Record<ConnectionStatus, string> = {
  idle: "text-slate-400",
  connecting: "text-amber-400",
  reconnecting: "text-amber-400",
  speaking: "text-violet-400",
  listening: "text-emerald-400",
  error: "text-red-400",
  closed: "text-slate-400",
};

interface Props {
  status: ConnectionStatus;
  error: string | null;
  latestUser: string;
  finetunedTranscript: string;
  usingAdapter: boolean;
  latestCoach: string;
  history: Exchange[];
  micLevel: number;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
}

export function VoicePanel({
  status,
  error,
  latestUser,
  finetunedTranscript,
  usingAdapter,
  latestCoach,
  history,
  micLevel,
  canStart,
  onStart,
  onStop,
}: Props) {
  const active =
    status === "listening" ||
    status === "connecting" ||
    status === "reconnecting" ||
    status === "speaking";
  const micLive = status === "listening" || status === "speaking";
  const ringColor = status === "speaking" ? "bg-violet-500" : "bg-emerald-500";
  const disabled = !active && !canStart;

  return (
    <div className="flex w-full flex-col items-center gap-6">
      <div className="relative flex h-40 w-40 items-center justify-center">
        {/* Audio-level visualiser: concentric rings that expand with mic RMS. */}
        {micLive &&
          [0, 1, 2].map((i) => (
            <span
              key={i}
              className={`absolute inset-0 rounded-full ${ringColor}`}
              style={{
                transform: `scale(${1 + micLevel * (0.45 + i * 0.45)})`,
                opacity: Math.max(0, (0.35 - i * 0.1) * (0.3 + micLevel)),
                transition: "transform 80ms ease-out, opacity 80ms ease-out",
              }}
            />
          ))}
        <button
          onClick={active ? onStop : onStart}
          disabled={disabled}
          className={[
            "relative z-10 flex h-28 w-28 items-center justify-center rounded-full",
            "text-3xl shadow-lg transition-colors",
            disabled
              ? "cursor-not-allowed bg-slate-800 text-slate-600"
              : status === "speaking"
                ? "bg-violet-500"
                : active
                  ? "bg-emerald-500 hover:bg-emerald-600"
                  : "bg-slate-700 hover:bg-slate-600",
          ].join(" ")}
          aria-label={active ? "Stop session" : "Start session"}
        >
          {active ? "■" : "🎙"}
        </button>
      </div>

      <p className={`text-sm font-medium ${STATUS_COLOR[status]}`}>
        {STATUS_LABEL[status]}
      </p>

      {error && (
        <p className="max-w-md rounded-lg bg-red-500/10 px-4 py-2 text-center text-sm text-red-300">
          {error}
        </p>
      )}

      <div className="grid w-full grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="min-h-24 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-sky-300">
            You said
          </p>
          <p className="text-sm leading-relaxed text-slate-200">
            {latestUser || <span className="text-slate-600">…</span>}
          </p>
          {finetunedTranscript && (
            <div className="mt-3 border-t border-slate-800 pt-2">
              <p className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-amber-300">
                Fine-tuned Whisper
                <span className="rounded bg-amber-500/15 px-1 py-0.5 text-[9px]">
                  {usingAdapter ? "QLoRA" : "base"}
                </span>
              </p>
              <p className="text-xs leading-relaxed text-slate-400">
                {finetunedTranscript}
              </p>
            </div>
          )}
        </div>
        <TranscriptCard
          label="Coach said"
          text={latestCoach}
          accent="text-emerald-300"
        />
      </div>

      <div className="w-full">
        <ChatHistory history={history} />
      </div>
    </div>
  );
}

function TranscriptCard({
  label,
  text,
  accent,
}: {
  label: string;
  text: string;
  accent: string;
}) {
  return (
    <div className="min-h-24 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <p className={`mb-2 text-xs font-semibold uppercase tracking-wide ${accent}`}>
        {label}
      </p>
      <p className="text-sm leading-relaxed text-slate-200">
        {text || <span className="text-slate-600">…</span>}
      </p>
    </div>
  );
}
