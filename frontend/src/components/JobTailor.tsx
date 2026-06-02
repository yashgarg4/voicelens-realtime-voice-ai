import { useState } from "react";

interface Props {
  generating: boolean;
  error: string | null;
  onGenerate: (jobDescription: string, resume: string) => void;
}

export function JobTailor({ generating, error, onGenerate }: Props) {
  const [open, setOpen] = useState(false);
  const [jd, setJd] = useState("");
  const [resume, setResume] = useState("");

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-5 py-3 text-left"
      >
        <span className="text-sm font-medium text-slate-200">
          ✨ Tailor questions to a job description
        </span>
        <span className="text-slate-500">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-slate-800 px-5 py-4">
          <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Job description
          </label>
          <textarea
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            rows={4}
            placeholder="Paste the job description here…"
            className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
          />

          <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Résumé <span className="font-normal normal-case">(optional)</span>
          </label>
          <textarea
            value={resume}
            onChange={(e) => setResume(e.target.value)}
            rows={3}
            placeholder="Paste your résumé to tailor questions to your background…"
            className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
          />

          {error && (
            <p className="rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}

          <button
            onClick={() => onGenerate(jd, resume)}
            disabled={generating || jd.trim().length < 20}
            className={[
              "self-start rounded-lg px-4 py-2 text-sm font-medium transition-colors",
              generating || jd.trim().length < 20
                ? "cursor-not-allowed bg-slate-800 text-slate-600"
                : "bg-emerald-500 text-white hover:bg-emerald-600",
            ].join(" ")}
          >
            {generating ? "Generating…" : "Generate tailored questions"}
          </button>
        </div>
      )}
    </div>
  );
}
