/*
 * Helpers for the look controls (N-15): what "no change" means for a parameter, a readable label,
 * and the conversion of the panel's flat state into the AppearanceSpec the API takes.
 */

// The value that means "no change". Everything is 0 except gamma, where 1 is neutral.
export const neutralOf = (row) => (row.name === "appearance.gamma" ? 1 : 0);

// "appearance.skinSmoothing" -> "Skin smoothing"; "appearance.expression.browInnerUp" -> "Brow inner up".
export function labelOf(name) {
  const last = name.split(".").pop();
  const words = last.replace(/([A-Z])/g, " $1").toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/* Turn the flat map (parameter name -> value) into an AppearanceSpec, leaving out every neutral value. */
export function buildSpec(look, rows) {
  const spec = {};
  for (const row of rows) {
    const value = look[row.name];
    if (value === undefined || value === neutralOf(row) || value === false) continue; // neutral: leave it out
    const key = row.name.replace(/^appearance\./, "");
    if (key.startsWith("expression.")) {
      spec.expression = { ...(spec.expression || {}), [key.slice("expression.".length)]: value };
    } else {
      spec[key] = value;
    }
  }
  return Object.keys(spec).length ? spec : null; // null clears the look on the server
}
