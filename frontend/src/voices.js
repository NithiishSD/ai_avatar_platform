/*
 * Matching a voice to a face (owner's request). A face's gender is never guessed from the picture:
 * it is read from what was recorded when the face was made (a generated face's prompt says "man" or
 * "woman"). When nothing is recorded the user chooses, and that choice is remembered per face in this
 * browser (a convenience only; it may be unavailable, e.g. in a private window).
 */

/** "female", "male" or null, from the face's provenance record. */
export function recordedGender(face) {
  const prompt = String(face?.provenance?.extra?.prompt || "").toLowerCase();
  if (/\bwoman\b/.test(prompt)) return "female";
  if (/\bman\b/.test(prompt)) return "male";
  return null;
}

const KEY = (avatarId) => `voice-for-${avatarId}`;

export function rememberedVoice(avatarId) {
  try { return window.localStorage.getItem(KEY(avatarId)); } catch { return null; }
}

export function rememberVoice(avatarId, voice) {
  try { window.localStorage.setItem(KEY(avatarId), voice); } catch { /* storage unavailable: just not remembered */ }
}

/** The voice to preselect for a face: what the user chose before, else the first voice of the recorded gender, else the default. */
export function voiceForFace(face, voices, fallback) {
  if (!face || !voices?.length) return fallback;
  const remembered = rememberedVoice(face.avatarId);
  if (remembered && voices.some((v) => v.id === remembered && v.present)) return remembered;
  const gender = recordedGender(face);
  return voices.find((v) => v.present && v.gender === gender)?.id || fallback;
}
