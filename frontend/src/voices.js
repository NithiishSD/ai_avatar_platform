/*
 * Matching a voice to a face (owner's request). A face's gender is never guessed from the picture:
 * it is read from what was recorded when the face was made (a generated face's prompt says "man" or
 * "woman"). When nothing is recorded the user chooses, and that choice is remembered per face in this
 * browser (a convenience only; it may be unavailable, e.g. in a private window).
 *
 * Plain functions, no React: CreateVideo.jsx imports them. No backend route is called here; the face
 * and voice lists come from the caller.
 */

/** "female", "male" or null, from the face's provenance record. */
export function recordedGender(face) {
  // `?.` stops at the first missing link, so a face with no provenance gives "" instead of throwing.
  const prompt = String(face?.provenance?.extra?.prompt || "").toLowerCase();
  // \b is a word boundary, so /\bman\b/ does not match the "man" inside "woman" or "manager".
  if (/\bwoman\b/.test(prompt)) return "female";
  if (/\bman\b/.test(prompt)) return "male";
  return null; // nothing recorded: the caller lets the user choose
}

// One localStorage key per face, so each face remembers its own voice.
const KEY = (avatarId) => `voice-for-${avatarId}`;

// localStorage is the browser's small per-site key/value store. It can throw (private windows,
// blocked site data), so every access is wrapped in try/catch and a failure just means "nothing saved".
export function rememberedVoice(avatarId) {
  try { return window.localStorage.getItem(KEY(avatarId)); } catch { return null; }
}

export function rememberVoice(avatarId, voice) {
  try { window.localStorage.setItem(KEY(avatarId), voice); } catch { /* storage unavailable: just not remembered */ }
}

/** The voice to preselect for a face: what the user chose before, else the first voice of the recorded gender, else the default. */
export function voiceForFace(face, voices, fallback) {
  if (!face || !voices?.length) return fallback; // nothing to match yet (lists still loading)
  const remembered = rememberedVoice(face.avatarId);
  // A remembered voice is only reused if it is still offered and its model is on disk (`present`).
  if (remembered && voices.some((v) => v.id === remembered && v.present)) return remembered;
  const gender = recordedGender(face);
  // find() returns the first match or undefined; `?.id || fallback` covers "no voice of that gender".
  return voices.find((v) => v.present && v.gender === gender)?.id || fallback;
}
