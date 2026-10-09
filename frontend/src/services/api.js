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

function imageForm(blob, style) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");
  form.append("style", style);
  return form;
}

/** Envoie une image et un style ; retourne le résultat et la latence mesurée côté client. */
export async function predict(blob, style, signal) {
  const start = performance.now();
  const response = await request("/predict", {
    method: "POST",
    body: imageForm(blob, style),
    signal,
  });
  const body = await response.json();
  // Latence client = aller-retour réseau + inférence + décodage JSON.
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
  const response = await request("/predict/visualize", {
    method: "POST",
    body: imageForm(blob, style),
  });
  return URL.createObjectURL(await response.blob());
}
