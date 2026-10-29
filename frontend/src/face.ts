import * as faceapi from "face-api.js";

// Model weights are fetched at runtime from a public CDN mirror of the
// standard face-api.js model files — no API key, nothing bundled, nothing
// sent anywhere except your own backend. Requires internet on first load
// per browser session (models are cached by the browser after that).
const MODEL_URL = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights";

let modelsLoaded = false;

export async function ensureModelsLoaded() {
  if (modelsLoaded) return;
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsLoaded = true;
}

export async function startWebcam(video: HTMLVideoElement) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopWebcam(stream: MediaStream | null) {
  stream?.getTracks().forEach((t) => t.stop());
}

/** Detects a single face in the given video frame and returns its 128-d descriptor, or null if no face found. */
export async function captureDescriptor(video: HTMLVideoElement): Promise<number[] | null> {
  const result = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!result) return null;
  return Array.from(result.descriptor);
}
