import { useRef, useState } from "react";

export default function Dropzone({ onFile }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);

  const handleFiles = (files) => {
    const file = files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        handleFiles(event.dataTransfer.files);
      }}
      className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition ${
        over ? "border-sakura bg-sakura/10" : "border-ink/20 bg-white hover:border-sakura"
      }`}
    >
      <p className="text-lg font-semibold">Glisse une photo ici</p>
      <p className="mt-1 text-sm text-ink/60">ou clique pour choisir un fichier (JPG, PNG)</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </div>
  );
}
