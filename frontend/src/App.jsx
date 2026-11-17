import { useCallback, useEffect, useRef, useState } from "react";

import BeforeAfterSlider from "./components/BeforeAfterSlider.jsx";
import Dropzone from "./components/Dropzone.jsx";
import ExampleGallery from "./components/ExampleGallery.jsx";
import LatencyBadge from "./components/LatencyBadge.jsx";
import StyleSelector from "./components/StyleSelector.jsx";
import StyleSliders from "./components/StyleSliders.jsx";
import Webcam from "./components/Webcam.jsx";
import {
  API_DOWN_MESSAGE,
  getHealth,
  getMetadata,
  predict,
  predictVisualize,
} from "./services/api.js";

const FALLBACK_STYLES = [
  { name: "hayao", label: "Hayao", description: "Style Miyazaki", engine: "dummy" },
  { name: "shinkai", label: "Shinkai", description: "Style Makoto Shinkai", engine: "dummy" },
];
const HEALTH_POLL_MS = 5000;

/** Réglages de l'interface -> paramètres de l'API (mix seulement si le curseur est actif). */
const apiOptions = ({ style, mixEnabled, mix, intensity }) => ({
  style,
  mix: mixEnabled ? mix : null,
  intensity,
});
const TABS = [
  ["upload", "Fichier"],
  ["examples", "Exemples"],
  ["webcam", "Webcam"],
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [styles, setStyles] = useState(FALLBACK_STYLES);
  const [canMix, setCanMix] = useState(false);
  // Réglages envoyés à l'API : style, mélange des poids Hayao -> Shinkai, intensité.
  const [settings, setSettings] = useState({
    style: "hayao",
    mixEnabled: false,
    mix: 0.5,
    intensity: 1,
  });
  const [tab, setTab] = useState("upload");
  const [source, setSource] = useState(null); // { blob, url }
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Copie synchrone des réglages : la webcam en continu lit toujours la dernière valeur.
  const settingsRef = useRef(settings);
  const sourceRef = useRef(source);
  const requestRef = useRef(0);
  const { style } = settings;

  // Vérifie l'état de l'API au chargement puis toutes les 5 s : le bandeau passe au
  // rouge si le backend s'arrête, et les styles (ONNX / factice) sont rechargés au retour.
  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      getHealth()
        .then((body) => {
          if (cancelled) return;
          setHealth(body);
          // L'API est revenue : on efface l'ancien message « API ne répond pas ».
          setError((previous) => (previous === API_DOWN_MESSAGE ? "" : previous));
          return getMetadata().then((metadata) => {
            if (cancelled) return;
            setStyles(metadata.styles);
            setCanMix(Boolean(metadata.can_mix));
          });
        })
        .catch(() => !cancelled && setHealth({ status: "down" }));
    };
    refresh();
    const timer = setInterval(refresh, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const run = useCallback(async (blob) => {
    const requestId = ++requestRef.current;
    setLoading(true);
    setError("");
    try {
      const prediction = await predict(blob, apiOptions(settingsRef.current));
      if (requestId === requestRef.current) setResult(prediction);
    } catch (err) {
      if (requestId === requestRef.current) {
        setResult(null);
        setError(err.message);
      }
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  }, []);

  const handleFile = useCallback(
    (blob) => {
      const next = { blob, url: URL.createObjectURL(blob) };
      setSource((previous) => {
        if (previous) URL.revokeObjectURL(previous.url);
        return next;
      });
      sourceRef.current = next;
      return run(blob);
    },
    [run],
  );

  /** Met à jour les réglages ; relance l'inférence si `rerun` (clic ou curseur relâché). */
  const updateSettings = (patch, rerun = false) => {
    const next = { ...settingsRef.current, ...patch };
    settingsRef.current = next;
    setSettings(next);
    if (rerun && sourceRef.current) run(sourceRef.current.blob);
  };

  const rerun = () => {
    if (sourceRef.current) run(sourceRef.current.blob);
  };

  const handleStyle = (name) => updateSettings({ style: name, mixEnabled: false }, true);

  const downloadComparison = async () => {
    try {
      const url = await predictVisualize(source.blob, apiOptions(settings));
      const link = document.createElement("a");
      link.href = url;
      link.download = `anime-${style}.png`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    }
  };

  const online = health?.status === "ok";

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-black tracking-tight">
            Manga<span className="text-sakura">ka</span>
          </h1>
          <p className="mt-1 text-ink/60">Transforme une photo en image de style anime.</p>
        </div>
        <p className="flex items-center gap-2 text-sm">
          <span className={`h-2.5 w-2.5 rounded-full ${online ? "bg-green-500" : "bg-red-500"}`} />
          {health === null && "Connexion à l'API…"}
          {health && online && `API en ligne · modèle ${health.model_version}`}
          {health && online && (health.gpu_available ? " · GPU" : " · CPU")}
          {health && !online && "API indisponible"}
        </p>
      </header>

      {health && !online && (
        <p className="mb-6 rounded-xl bg-red-100 p-3 text-sm text-red-700" role="alert">
          {health.status === "down"
            ? API_DOWN_MESSAGE
            : "L'API est démarrée mais le modèle n'est pas encore chargé (erreur 503)."}
        </p>
      )}

      <main className="grid gap-8 lg:grid-cols-2">
        <section className="space-y-6">
          <div>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink/50">1. Image</h2>
            <div className="mb-3 flex gap-2">
              {TABS.map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setTab(key)}
                  className={`rounded-full px-4 py-1.5 text-sm font-semibold ${
                    tab === key ? "bg-ink text-white" : "bg-ink/10"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {tab === "upload" && <Dropzone onFile={handleFile} />}
            {tab === "examples" && <ExampleGallery onFile={handleFile} />}
            {tab === "webcam" && <Webcam onCapture={handleFile} />}
          </div>

          <div>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink/50">2. Style</h2>
            <StyleSelector
              styles={styles}
              value={settings.mixEnabled ? null : style}
              onChange={handleStyle}
            />
          </div>

          <div>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-ink/50">
              3. Réglages
            </h2>
            <StyleSliders
              canMix={canMix}
              mixEnabled={settings.mixEnabled}
              mix={settings.mix}
              intensity={settings.intensity}
              onChange={updateSettings}
              onCommit={rerun}
            />
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wide text-ink/50">4. Résultat</h2>
          {/* Si l'API est coupée, le bandeau du haut l'indique déjà : pas de doublon. */}
          {error && !(error === API_DOWN_MESSAGE && !online) && (
            <p className="rounded-xl bg-red-100 p-3 text-sm text-red-700">{error}</p>
          )}
          {source ? (
            <>
              <BeforeAfterSlider
                beforeUrl={source.url}
                afterUrl={result?.imageUrl}
                loading={loading}
              />
              <LatencyBadge result={result} />
              {result && (
                <div className="flex flex-wrap gap-2">
                  <a
                    href={result.imageUrl}
                    download={`anime-${result.style}.png`}
                    className="rounded-lg bg-sakura px-4 py-2 font-semibold text-white"
                  >
                    Télécharger le résultat
                  </a>
                  <button
                    type="button"
                    onClick={downloadComparison}
                    className="rounded-lg bg-ink/10 px-4 py-2"
                  >
                    Télécharger avant / après
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="grid aspect-video place-items-center rounded-2xl bg-white text-ink/40">
              Choisis une image pour commencer
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
