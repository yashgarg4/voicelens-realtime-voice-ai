import type { WerResult } from "../types";

export function WerBadge({ wer }: { wer: WerResult | null }) {
  if (!wer || !wer.available) return null;

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/60 px-3 py-1 text-xs"
      title={`Whisper-small fine-tuned with QLoRA on ${wer.dataset} (${wer.samples} held-out clips)`}
    >
      <span className="font-semibold text-slate-400">STT WER</span>
      <span className="tabular-nums text-slate-500 line-through">
        {wer.base_wer}%
      </span>
      <span className="text-slate-600">→</span>
      <span className="tabular-nums font-semibold text-emerald-300">
        {wer.finetuned_wer}%
      </span>
      {wer.relative != null && (
        <span className="tabular-nums text-emerald-500">
          ({wer.relative}%)
        </span>
      )}
    </div>
  );
}
