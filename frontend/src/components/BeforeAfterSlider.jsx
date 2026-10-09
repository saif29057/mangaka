import { useState } from "react";

/** Superpose l'original et le résultat ; le curseur découvre l'un ou l'autre. */
export default function BeforeAfterSlider({ beforeUrl, afterUrl, loading }) {
  const [position, setPosition] = useState(50);

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-2xl bg-ink/5">
        <img src={afterUrl || beforeUrl} alt="Version anime" className="block w-full" />
        <img
          src={beforeUrl}
          alt="Photo originale"
          className="absolute inset-0 h-full w-full"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        />
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow"
          style={{ left: `${position}%` }}
        />
        <span className="absolute left-3 top-3 rounded bg-ink/70 px-2 py-0.5 text-xs text-white">
          Avant
        </span>
        <span className="absolute right-3 top-3 rounded bg-sakura px-2 py-0.5 text-xs text-white">
          Après
        </span>
        {loading && (
          <div className="absolute inset-0 grid place-items-center bg-white/50 text-sm font-semibold">
            Génération en cours…
          </div>
        )}
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
        aria-label="Comparer avant et après"
        className="w-full accent-sakura"
      />
    </div>
  );
}
