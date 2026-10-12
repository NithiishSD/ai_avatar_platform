import { labelOf, neutralOf } from "./look"; // labels and neutral values shared with the panel

/*
 * Look and expression controls: lighting, colour grade, head pose and a standing expression
 * (N-15). Every slider is built from GET /api/v1/parameters, so its range, default and label
 * come from the API's own request models and cannot drift from them.
 *
 * The panel is controlled: it shows `look` (a flat map of parameter name -> value) and reports
 * changes through `onChange`. A parameter at its neutral value is left out of the spec that
 * `buildSpec` (look.js) makes, so an untouched panel sends nothing at all. `rows` are the
 * appearance rows of the catalogue, fetched once by the screen. `idPrefix` keeps element ids
 * unique when the controls are mounted on two screens at once ("live-look-...", "create-look-...").
 */

const GROUPS = [
  { id: "lighting", title: "Lighting and colour", pick: (r) => !r.name.startsWith("appearance.expression.") && !r.name.startsWith("appearance.head") },
  { id: "head", title: "Head position", pick: (r) => r.name.startsWith("appearance.head") },
  { id: "expression", title: "Standing expression", pick: (r) => r.name.startsWith("appearance.expression.") },
];

export default function LookControls({ rows, look, onChange, applied, idPrefix = "look" }) {
  if (!rows.length) return <div style={{ fontSize: "0.78rem", color: "#94a3b8" }}>Loading look controls…</div>;

  const set = (name, value) => onChange({ ...look, [name]: value });
  const changed = Object.entries(look).filter(([name, v]) => {
    const row = rows.find((r) => r.name === name);
    return row && v !== neutralOf(row) && v !== false && v !== undefined;
  }).length;

  return (
    <div id={`${idPrefix}-look-controls`} data-changed={changed} data-applied={applied}>
      {GROUPS.map((group) => (
        <details key={group.id} id={`${idPrefix}-look-${group.id}`} open={group.id === "lighting"} style={{ marginTop: "8px" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.85rem" }}>{group.title}</summary>
          {rows.filter(group.pick).map((row) => {
            const id = `${idPrefix}-look-${row.name.replace(/\./g, "-")}`;
            if (row.type === "boolean") {
              return (
                <label key={row.name} style={{ display: "block", fontSize: "0.78rem", marginTop: "6px" }}>
                  <input type="checkbox" id={id} checked={Boolean(look[row.name])} onChange={(e) => set(row.name, e.target.checked)} />{" "}
                  {labelOf(row.name)} <span style={{ color: "#94a3b8" }}>(evens out the light and colour cast)</span>
                </label>
              );
            }
            const value = look[row.name] ?? neutralOf(row);
            const step = (row.maximum - row.minimum) / 100;
            return (
              <label key={row.name} style={{ display: "block", fontSize: "0.78rem", marginTop: "6px", color: "#cbd5e1" }}>
                {labelOf(row.name)}: <strong>{Number(value).toFixed(2)}</strong>
                <input type="range" id={id} min={row.minimum} max={row.maximum} step={step} value={value}
                  onChange={(e) => set(row.name, Number(e.target.value))} style={{ width: "100%" }} />
              </label>
            );
          })}
        </details>
      ))}
      <button type="button" id={`${idPrefix}-look-reset`} className="secondary" style={{ marginTop: "10px" }} disabled={!changed} onClick={() => onChange({})}>
        Reset look
      </button>
    </div>
  );
}
