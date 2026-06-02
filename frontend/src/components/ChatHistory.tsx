import type { Exchange } from "../types";

function formatTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ChatHistory({ history }: { history: Exchange[] }) {
  if (history.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-6 text-center text-sm text-slate-600">
        Your conversation will appear here.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Chat history
      </h2>
      <div className="flex flex-col gap-4">
        {history.map((ex) => (
          <div key={ex.id} className="flex flex-col gap-2">
            {ex.user && (
              <div className="flex justify-end">
                <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-sky-600/80 px-4 py-2 text-sm text-white">
                  {ex.user}
                </div>
              </div>
            )}
            {ex.coach && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-2xl rounded-bl-sm bg-slate-800 px-4 py-2 text-sm text-slate-100">
                  {ex.coach}
                </div>
              </div>
            )}
            <span className="self-center text-[10px] text-slate-600">
              {formatTime(ex.at)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
