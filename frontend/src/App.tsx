import { useCallback } from "react";
import { useSession } from "./hooks/useSession";
import { useVoiceStream } from "./hooks/useVoiceStream";
import { QuestionDisplay } from "./components/QuestionDisplay";
import { VoicePanel } from "./components/VoicePanel";
import { FeedbackCard } from "./components/FeedbackCard";
import { ScoreHistory } from "./components/ScoreHistory";
import { WerBadge } from "./components/WerBadge";

export default function App() {
  const session = useSession();
  const voice = useVoiceStream();

  const active =
    voice.status === "listening" ||
    voice.status === "connecting" ||
    voice.status === "speaking";

  const handleStart = useCallback(async () => {
    try {
      const { questionId, sessionId } = await session.begin();
      await voice.start(questionId, sessionId);
    } catch (err) {
      console.error("Failed to start session", err);
    }
  }, [session, voice]);

  const handleStop = useCallback(async () => {
    voice.stop();
    await session.finish();
    // Scores were saved server-side on turn_complete; refresh the dashboard.
    await session.refreshHistory();
  }, [voice, session]);

  return (
    <div className="min-h-full bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-2xl flex-col px-6 py-10">
        <header className="mb-8 flex flex-col items-center gap-3 text-center">
          <h1 className="text-3xl font-bold tracking-tight">
            Voice<span className="text-emerald-400">Lens</span>
          </h1>
          <p className="text-sm text-slate-400">
            Real-time AI interview coach · Phase 4 — dashboard
          </p>
          <WerBadge wer={session.wer} />
        </header>

        {session.loadError && (
          <p className="mb-6 rounded-lg bg-red-500/10 px-4 py-2 text-center text-sm text-red-300">
            Couldn't load questions: {session.loadError} — is the backend running?
          </p>
        )}

        <main className="flex flex-1 flex-col gap-8">
          <QuestionDisplay
            questions={session.questions}
            selectedId={session.selectedId}
            onSelect={session.setSelectedId}
            active={active}
            currentQuestion={session.selectedQuestion}
          />

          <VoicePanel
            status={voice.status}
            error={voice.error}
            latestUser={voice.latestUser}
            finetunedTranscript={voice.finetunedTranscript}
            usingAdapter={voice.usingAdapter}
            latestCoach={voice.latestCoach}
            history={voice.history}
            micLevel={voice.micLevel}
            canStart={session.selectedId != null}
            onStart={handleStart}
            onStop={handleStop}
          />

          <FeedbackCard feedback={voice.feedback} />

          <ScoreHistory sessions={session.history} />
        </main>

        <footer className="mt-12 text-center text-xs text-slate-600">
          Powered by Gemini Live · Whisper QLoRA · 16 kHz in / 24 kHz out PCM
        </footer>
      </div>
    </div>
  );
}
