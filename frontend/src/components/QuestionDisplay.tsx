import type { Question } from "../types";

const DIFFICULTY_COLOR: Record<string, string> = {
  easy: "bg-emerald-500/15 text-emerald-300",
  medium: "bg-amber-500/15 text-amber-300",
  hard: "bg-red-500/15 text-red-300",
};

function Badges({ q }: { q: Question }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="rounded-full bg-slate-700/60 px-2.5 py-0.5 text-xs font-medium text-slate-300">
        {q.category}
      </span>
      <span
        className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
          DIFFICULTY_COLOR[q.difficulty] ?? "bg-slate-700/60 text-slate-300"
        }`}
      >
        {q.difficulty}
      </span>
    </div>
  );
}

interface Props {
  questions: Question[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  active: boolean;
  currentQuestion: Question | null;
}

export function QuestionDisplay({
  questions,
  selectedId,
  onSelect,
  active,
  currentQuestion,
}: Props) {
  // During a live session: show the question being asked, prominently.
  if (active && currentQuestion) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6">
        <Badges q={currentQuestion} />
        <p className="mt-3 text-lg font-medium leading-relaxed text-slate-100">
          {currentQuestion.question}
        </p>
      </div>
    );
  }

  // Idle: pick a question, grouped by category.
  const categories = [...new Set(questions.map((q) => q.category))];
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <h2 className="mb-4 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Choose a question
      </h2>
      <div className="flex flex-col gap-5">
        {categories.map((cat) => (
          <div key={cat}>
            <p className="mb-2 text-xs font-semibold text-slate-400">{cat}</p>
            <div className="flex flex-col gap-1.5">
              {questions
                .filter((q) => q.category === cat)
                .map((q) => {
                  const isSel = q.id === selectedId;
                  return (
                    <button
                      key={q.id}
                      onClick={() => onSelect(q.id)}
                      className={[
                        "flex items-start gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors",
                        isSel
                          ? "bg-emerald-500/15 text-emerald-100 ring-1 ring-emerald-500/40"
                          : "text-slate-300 hover:bg-slate-800/60",
                      ].join(" ")}
                    >
                      <span
                        className={[
                          "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                          isSel ? "bg-emerald-400" : "bg-slate-600",
                        ].join(" ")}
                      />
                      <span>{q.question}</span>
                    </button>
                  );
                })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
