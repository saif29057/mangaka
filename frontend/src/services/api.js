// Appels vers l'API FastAPI. /api est redirigé vers le backend (Vite en dev, nginx en prod).
const BASE_URL = import.meta.env.VITE_API_URL || "/api";

// Message affiché quand le backend ne répond pas (serveur arrêté, conteneur en redémarrage).
export const API_DOWN_MESSAGE =
  "L'API ne répond pas : vérifie que le backend est démarré (docker compose up ou uvicorn).";

async function parseError(response) {
  try {
    // FastAPI répond toujours en JSON : { "detail": ... }
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    // Réponse 5xx non JSON : c'est le proxy (Vite renvoie 500, nginx 502/504)
    // qui n'a pas pu joindre FastAPI.
    return response.status >= 500 ? API_DOWN_MESSAGE : `Erreur ${response.status}`;
  }
}

/** fetch qui transforme une erreur réseau (« Failed to fetch ») en message lisible. */
async function request(path, options) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(API_DOWN_MESSAGE);
  }
  if (!response.ok) throw new Error(await parseError(response));
  return response;
}

const getJson = async (path) => (await request(path)).json();

export const getHealth = () => getJson("/health");
export const getMetadata = () => getJson("/model/metadata");

/**
 * options : { style, mix, intensity }. mix (0 = Hayao, 1 = Shinkai) n'est envoyé que s'il
 * est défini ; l'API mélange alors les poids des deux générateurs.
 */
function imageForm(blob, { style, mix = null, intensity = 1 }) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");
  form.append("style", style);
  if (mix !== null && mix !== undefined) form.append("mix", String(mix));
  form.append("intensity", String(intensity));
  return form;
}

/** Envoie une image et les réglages ; retourne le résultat et la latence côté client. */
export async function predict(blob, options, signal) {
  const start = performance.now();
  const response = await request("/predict", {
    method: "POST",
    body: imageForm(blob, options),
    signal,
  });
  const body = await response.json();
  // Latence client = aller-retour réseau + inférence + décodage JSON.
  const clientMs = performance.now() - start;

  return {
    style: body.style,
    mix: body.mix,
    intensity: body.intensity,
    engine: body.engine,
    width: body.width,
    height: body.height,
    serverMs: body.latency_ms,
    clientMs,
    imageUrl: `data:image/png;base64,${body.image_base64}`,
  };
}

/** Planche avant/après générée par le serveur (PNG) : utile pour le téléchargement. */
export async function predictVisualize(blob, options) {
  const response = await request("/predict/visualize", {
    method: "POST",
    body: imageForm(blob, options),
  });
  return URL.createObjectURL(await response.blob());
}
