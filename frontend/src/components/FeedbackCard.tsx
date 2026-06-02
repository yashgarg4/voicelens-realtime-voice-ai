import type { FeedbackScores } from "../types";
import { RadarChart } from "./RadarChart";

function scoreColor(v: number): string {
  if (v >= 8) return "bg-emerald-500";
  if (v >= 5) return "bg-amber-500";
  return "bg-red-500";
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-slate-300">{label}</span>
        <span className="tabular-nums text-slate-400">{value}/10</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div
          className={`h-full rounded-full transition-all ${scoreColor(value)}`}
          style={{ width: `${value * 10}%` }}
        />
      </div>
    </div>
  );
}

export function FeedbackCard({ feedback }: { feedback: FeedbackScores | null }) {
  if (!feedback) return null;

  return (
    <div className="rounded-2xl border border-emerald-800/40 bg-emerald-950/20 p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-emerald-300">Feedback</h2>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold tabular-nums text-emerald-200">
            {feedback.overall}
          </span>
          <span className="text-xs text-slate-400">/10 overall</span>
        </div>
      </div>

      <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-center">
        <div className="shrink-0">
          <RadarChart
            content={feedback.content}
            depth={feedback.depth}
            structure={feedback.structure}
          />
        </div>
        <div className="flex w-full flex-col gap-3">
          <ScoreBar label="Content" value={feedback.content} />
          <ScoreBar label="Depth" value={feedback.depth} />
          <ScoreBar label="Structure" value={feedback.structure} />
        </div>
      </div>

      {feedback.summary && (
        <p className="mt-4 border-t border-slate-800 pt-4 text-sm leading-relaxed text-slate-300">
          {feedback.summary}
        </p>
      )}
    </div>
  );
}
