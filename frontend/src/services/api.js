// Appels vers l'API FastAPI. /api est redirigé vers le backend (Vite en dev, nginx en prod).
const BASE_URL = import.meta.env.VITE_API_URL || "/api";

async function parseError(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    return `Erreur ${response.status}`;
  }
}

async function getJson(path) {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export const getHealth = () => getJson("/health");
export const getMetadata = () => getJson("/model/metadata");

/** Envoie une image et un style ; retourne le résultat et la latence mesurée côté client. */
export async function predict(blob, style, signal) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");
  form.append("style", style);

  const start = performance.now();
  const response = await fetch(`${BASE_URL}/predict`, { method: "POST", body: form, signal });
  if (!response.ok) throw new Error(await parseError(response));
  const body = await response.json();
  const clientMs = performance.now() - start;

  return {
    style: body.style,
    engine: body.engine,
    width: body.width,
    height: body.height,
    serverMs: body.latency_ms,
    clientMs,
    imageUrl: `data:image/png;base64,${body.image_base64}`,
  };
}

/** Planche avant/après générée par le serveur (PNG) : utile pour le téléchargement. */
export async function predictVisualize(blob, style) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");
  form.append("style", style);
  const response = await fetch(`${BASE_URL}/predict/visualize`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await parseError(response));
  return URL.createObjectURL(await response.blob());
}
