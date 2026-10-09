import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Capture webcam. `onCapture(blob)` doit retourner une promesse : en mode continu,
 * l'image suivante n'est envoyée qu'une fois la réponse précédente reçue.
 */
export default function Webcam({ onCapture }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const loopRef = useRef(false);
  const [active, setActive] = useState(false);
  const [continuous, setContinuous] = useState(false);
  const [error, setError] = useState("");

  const grabFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return Promise.resolve(null);
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }, []);

  const stop = useCallback(() => {
    loopRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setActive(false);
    setContinuous(false);
  }, []);

  const start = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setActive(true);
    } catch {
      setError("Impossible d'accéder à la webcam (permission refusée ou caméra absente).");
    }
  };

  const captureOnce = async () => {
    const blob = await grabFrame();
    if (blob) await onCapture(blob);
  };

  useEffect(() => {
    if (!continuous) {
      loopRef.current = false;
      return undefined;
    }
    loopRef.current = true;
    (async () => {
      while (loopRef.current) {
        const blob = await grabFrame();
        if (blob) await onCapture(blob);
        await new Promise((resolve) => setTimeout(resolve, 150));
      }
    })();
    return () => {
      loopRef.current = false;
    };
  }, [continuous, grabFrame, onCapture]);

  useEffect(() => stop, [stop]);

  return (
    <div className="space-y-3">
      <video
        ref={videoRef}
        muted
        playsInline
        className={`w-full rounded-2xl bg-ink/10 ${active ? "" : "hidden"}`}
      />
      {error && <p className="text-sm text-red-600">{error}</p>}
      <div className="flex flex-wrap gap-2">
        {!active ? (
          <button type="button" onClick={start} className="rounded-lg bg-ink px-4 py-2 text-white">
            Activer la webcam
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={captureOnce}
              disabled={continuous}
              className="rounded-lg bg-sakura px-4 py-2 font-semibold text-white disabled:opacity-40"
            >
              Capturer
            </button>
            <button
              type="button"
              onClick={() => setContinuous((value) => !value)}
              className="rounded-lg bg-sky px-4 py-2 font-semibold text-white"
            >
              {continuous ? "Arrêter le mode continu" : "Mode continu"}
            </button>
            <button type="button" onClick={stop} className="rounded-lg bg-ink/10 px-4 py-2">
              Couper
            </button>
          </>
        )}
      </div>
    </div>
  );
}
