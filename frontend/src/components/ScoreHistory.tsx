import { useState } from "react";
import type { SessionRecord } from "../types";
import { RadarChart } from "./RadarChart";

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function overallColor(v: number | null): string {
  if (v == null) return "text-slate-500";
  if (v >= 8) return "text-emerald-300";
  if (v >= 5) return "text-amber-300";
  return "text-red-300";
}

export function ScoreHistory({ sessions }: { sessions: SessionRecord[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  // Only show sessions that produced a score.
  const scored = sessions.filter((s) => s.overall != null);

  if (scored.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-6 text-center text-sm text-slate-600">
        No scored sessions yet — finish an interview answer to build your history.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Score history
      </h2>
      <div className="overflow-hidden rounded-xl border border-slate-800">
        {scored.map((s) => {
          const open = expanded === s.id;
          return (
            <div key={s.id} className="border-b border-slate-800 last:border-b-0">
              <button
                onClick={() => setExpanded(open ? null : s.id)}
                className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-slate-800/40"
              >
                <span className="w-28 shrink-0 text-xs text-slate-500">
                  {fmtDate(s.started_at)}
                </span>
                <span className="hidden w-32 shrink-0 text-xs text-slate-400 sm:block">
                  {s.category}
                </span>
                <span className="flex-1 truncate text-sm text-slate-300">
                  {s.question}
                </span>
                <span
                  className={`shrink-0 text-sm font-bold tabular-nums ${overallColor(
                    s.overall
                  )}`}
                >
                  {s.overall}
                </span>
                <span className="shrink-0 text-slate-600">{open ? "▲" : "▼"}</span>
              </button>

              {open && (
                <div className="flex flex-col items-center gap-4 bg-slate-900/40 px-4 py-4 sm:flex-row">
                  <RadarChart
                    content={s.content ?? 0}
                    depth={s.depth ?? 0}
                    structure={s.structure ?? 0}
                    size={170}
                  />
                  <div className="flex-1">
                    <div className="mb-2 flex gap-4 text-xs text-slate-400">
                      <span>Content {s.content}</span>
                      <span>Depth {s.depth}</span>
                      <span>Structure {s.structure}</span>
                    </div>
                    {s.summary && (
                      <p className="text-sm leading-relaxed text-slate-300">
                        {s.summary}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
