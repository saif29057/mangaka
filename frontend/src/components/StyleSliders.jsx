/**
 * Curseurs d'interpolation (défi de l'axe 2) :
 *  - mix       : mélange des poids des générateurs Hayao -> Shinkai (0 à 1), calculé par l'API ;
 *  - intensity : dosage du style, de la photo (0) au style accentué (2).
 * Les valeurs sont envoyées à l'API quand on relâche le curseur, pas à chaque pixel.
 */
export default function StyleSliders({ canMix, mixEnabled, mix, intensity, onChange, onCommit }) {
  const commit = () => onCommit();

  return (
    <div className="space-y-5 rounded-xl bg-white p-4 shadow-sm">
      <div>
        <label className="flex items-center justify-between gap-3 text-sm font-semibold">
          <span>Interpolation Hayao ↔ Shinkai</span>
          <input
            type="checkbox"
            className="h-4 w-4 accent-sakura"
            checked={mixEnabled}
            disabled={!canMix}
            onChange={(event) => onChange({ mixEnabled: event.target.checked }, true)}
          />
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={mix}
          disabled={!canMix || !mixEnabled}
          aria-label="Mélange des poids Hayao vers Shinkai"
          onChange={(event) => onChange({ mix: Number(event.target.value) })}
          onPointerUp={commit}
          onKeyUp={commit}
          className="mt-2 w-full accent-sakura disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-ink/50">
          <span>Hayao</span>
          <span className="tabular-nums">
            {mixEnabled ? `${Math.round((1 - mix) * 100)} % / ${Math.round(mix * 100)} %` : "désactivé"}
          </span>
          <span>Shinkai</span>
        </div>
        {!canMix && (
          <p className="mt-1 text-xs text-ink/50">
            Disponible quand les deux générateurs ONNX sont chargés.
          </p>
        )}
      </div>

      <div>
        <p className="flex items-center justify-between text-sm font-semibold">
          <span>Intensité du style</span>
          <span className="tabular-nums text-ink/60">× {intensity.toFixed(1)}</span>
        </p>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={intensity}
          aria-label="Intensité du style"
          onChange={(event) => onChange({ intensity: Number(event.target.value) })}
          onPointerUp={commit}
          onKeyUp={commit}
          className="mt-2 w-full accent-sakura"
        />
        <div className="flex justify-between text-xs text-ink/50">
          <span>Photo</span>
          <span>Modèle</span>
          <span>Accentué</span>
        </div>
      </div>
    </div>
  );
}
