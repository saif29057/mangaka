// Images d'exemple servies depuis public/examples : remplacez-les par de vraies photos.
const EXAMPLES = ["exemple-1.jpg", "exemple-2.jpg", "exemple-3.jpg", "exemple-4.jpg"];

export default function ExampleGallery({ onFile }) {
  const pick = async (name) => {
    const response = await fetch(`/examples/${name}`);
    onFile(await response.blob());
  };

  return (
    <div className="grid grid-cols-4 gap-3">
      {EXAMPLES.map((name) => (
        <button
          key={name}
          type="button"
          onClick={() => pick(name)}
          className="overflow-hidden rounded-xl ring-2 ring-transparent transition hover:ring-sakura"
        >
          <img src={`/examples/${name}`} alt={name} className="aspect-square w-full object-cover" />
        </button>
      ))}
    </div>
  );
}
