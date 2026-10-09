export default function LatencyBadge({ result }) {
  if (!result) return null;
  const items = [
    ["Serveur", `${result.serverMs.toFixed(0)} ms`],
    ["Client", `${result.clientMs.toFixed(0)} ms`],
    ["Taille", `${result.width}×${result.height}`],
    ["Moteur", result.engine === "onnx" ? "ONNX Runtime" : "factice"],
  ];
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-xl bg-white p-3 shadow-sm">
          <dt className="text-xs uppercase tracking-wide text-ink/50">{label}</dt>
          <dd className="text-lg font-bold tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
