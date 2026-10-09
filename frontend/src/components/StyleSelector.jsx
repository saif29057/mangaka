export default function StyleSelector({ styles, value, onChange }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {styles.map((style) => {
        const active = style.name === value;
        return (
          <button
            key={style.name}
            type="button"
            onClick={() => onChange(style.name)}
            aria-pressed={active}
            className={`rounded-xl border-2 p-4 text-left transition ${
              active ? "border-sakura bg-sakura/10" : "border-ink/10 bg-white hover:border-sakura/60"
            }`}
          >
            <span className="flex items-center justify-between">
              <span className="text-base font-bold">{style.label}</span>
              <span className="rounded-full bg-ink/10 px-2 py-0.5 text-xs">
                {style.engine === "onnx" ? "ONNX" : "factice"}
              </span>
            </span>
            <span className="mt-1 block text-sm text-ink/60">{style.description}</span>
          </button>
        );
      })}
    </div>
  );
}
