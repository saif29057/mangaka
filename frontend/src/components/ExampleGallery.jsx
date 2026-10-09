// Photos d'exemple servies depuis public/examples (CC0, sources dans le README).
const EXAMPLES = [
  ["exemple-1.jpg", "Lac et montagnes"],
  ["exemple-2.jpg", "Rue d'Amsterdam la nuit"],
  ["exemple-3.jpg", "Portrait de chat"],
  ["exemple-4.jpg", "Maison traditionnelle"],
];

export default function ExampleGallery({ onFile }) {
  const pick = async (name) => {
    const response = await fetch(`/examples/${name}`);
    onFile(await response.blob());
  };

  return (
    <div className="grid grid-cols-4 gap-3">
      {EXAMPLES.map(([name, label]) => (
        <button
          key={name}
          type="button"
          onClick={() => pick(name)}
          className="overflow-hidden rounded-xl ring-2 ring-transparent transition hover:ring-sakura"
        >
          <img
            src={`/examples/${name}`}
            alt={label}
            title={label}
            className="aspect-square w-full object-cover"
          />
        </button>
      ))}
    </div>
  );
}
