import { useState } from "react"; // useState: a value that survives re-renders and triggers one when set

/*
 * "Is this file ours?" - upload an audio or video file (and optionally its manifest) to
 * POST /api/v1/provenance/verify and show the evidence and the verdict.
 *
 * The server reports four kinds of evidence separately (audio watermark, video watermark, signed
 * manifest, audit record) because they fail in different ways. The panel shows the verdict with
 * its plain-language meaning, and keeps the "no evidence proves nothing" caveat visible: a file
 *
 * Styling here is inline (`style={...}` objects) rather than App.css classes; `card` and `small` are
 * shared style objects reused below.
 * with no marks is not shown to be real.
 */

const card = { background: "#0f172a", border: "1px solid #334155", borderRadius: "8px", padding: "12px 14px", marginBottom: "12px" };
const small = { fontSize: "0.78rem", color: "#94a3b8" };

// Colour of the verdict text, keyed by the verdict name the server returns.
// Green = ours and unchanged, amber = ours but something differs, red = a forged or broken manifest.
const TONE = {
  authentic_original: "#4ade80",
  ours_modified: "#fbbf24",
  manifest_for_another_file: "#fbbf24",
  tampered_manifest: "#f87171",
  foreign_manifest: "#f87171",
  no_evidence: "#94a3b8", // grey: no evidence either way
};

// One line of text for a watermark result, or null when the server did not report that watermark.
function markLine(label, mark) {
  if (!mark) return null; // null lines are dropped by .filter(Boolean) below
  if (mark.available === false) return `${label}: could not be checked (${mark.reason})`; // the check itself could not run, which is different from "not found"
  const bits = mark.tagBitsMatching ?? mark.bitsMatching; // `??` takes the right side only when the left is null or undefined
  const of = mark.tagBitsMatching !== undefined ? 128 : 16; // a mark that reports tagBitsMatching carries a 128-bit tag; the other kind carries 16 bits
  return `${label}: ${mark.detected ? "found" : "not found"} (${bits}/${of} bits match)`;
}

// `{ apiBase }` destructures the props object: App passes apiBase="http://localhost:8000".
export default function ProvenancePanel({ apiBase }) {
  const [file, setFile] = useState(null); // the File object the user picked
  const [manifestFile, setManifestFile] = useState(null); // optional signed manifest (.json)
  const [busy, setBusy] = useState(false); // disables the button while the request runs
  const [error, setError] = useState("");
  const [report, setReport] = useState(null); // the server's verdict and evidence

  // An async function can `await` a Promise, so the steps read top to bottom instead of nested .then calls.
  const verify = async (event) => {
    event.preventDefault(); // stop the browser from submitting the form and reloading the page
    if (!file) return; // the button is disabled without a file; this is a second guard
    setBusy(true);
    setError("");
    setReport(null); // clear the old result so it is never shown next to a new file
    // FormData builds a multipart/form-data body, the format HTML forms use to upload files.
    // The browser sets the Content-Type header (with its boundary) itself, so none is set here.
    const form = new FormData();
    form.append("file", file); // field names match the FastAPI route parameters
    if (manifestFile) form.append("manifest", manifestFile); // only sent when chosen
    try {
      const res = await fetch(`${apiBase}/api/v1/provenance/verify`, { method: "POST", body: form }); // fetch only rejects on network errors
      const body = await res.json(); // error replies are JSON too: FastAPI puts the reason in `detail`
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)); // res.ok is false for 4xx/5xx; throwing sends us to catch
      setReport(body);
    } catch (err) {
      setError(err.message); // shown in the red alert below
    } finally {
      setBusy(false); // finally runs after success and after failure alike
    }
  };

  return ( // the JSX the panel renders
    <div id="provenance-panel">
      <h2 style={{ marginTop: "1.5rem" }}>Verify a file</h2>
      <div style={card}>
        {/* onSubmit fires on the button click and on Enter; `required` below makes the browser check for a file first. */}
        <form onSubmit={verify}>
          <div style={{ ...small, marginBottom: "6px" }}>
            Is this audio or video from this platform? Add its manifest too if you have it.
          </div>
          <label style={small}>
            File
            <input id="verify-file" type="file" accept="video/*,audio/*,.mp4,.wav,.mp3,.m4a" required
              onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </label>
          <label style={{ ...small, display: "block", marginTop: "6px" }}>
            Manifest (optional)
            <input id="verify-manifest" type="file" accept="application/json,.json"
              onChange={(e) => setManifestFile(e.target.files?.[0] || null)} />
          </label>
          <button type="submit" id="verify-btn" style={{ marginTop: "10px" }} disabled={!file || busy}>
            {busy ? "Checking…" : "Verify"}
          </button>
        </form>
        {/* Render-if pattern: `cond && <X/>` shows X only when cond is truthy. */}
        {error && <div className="alert error" id="verify-error" style={{ marginTop: "8px" }}>{error}</div>}
        {report && (
          <div id="verify-result" style={{ marginTop: "10px" }}>
            {/* data-verdict lets tests read the raw verdict; TONE picks its colour, light grey if unknown. */}
            <div id="verify-verdict" data-verdict={report.verdict} style={{ fontWeight: 700, color: TONE[report.verdict] || "#e2e8f0" }}>
              {report.verdict}
            </div>
            <div style={{ ...small, color: "#e2e8f0", marginTop: "4px" }}>{report.meaning}</div>
            {/* One bullet per kind of evidence: the two watermarks, the manifest, the audit record.
               `...small` spreads the shared style object, then the following keys add to it. */}
            <ul style={{ ...small, margin: "8px 0 0 16px", padding: 0 }}>
              {[markLine("video watermark", report.watermarks.video), markLine("audio watermark", report.watermarks.audio),
                report.manifest ? `manifest: ${report.manifest.trustworthy ? "valid, ours, matches this file" : report.manifest.validSignature ? "valid signature, but not usable for this file" : "signature does NOT match"}` : "manifest: none supplied or found",
                report.record ? `audit record: issued ${report.record.ts}, job ${report.record.details.job ?? "?"}` : "audit record: none found",
              ].filter(Boolean).map((line) => <li key={line}>{line}</li>)}
              {/* After the evidence: the server's plain-language explanation. Each line's text doubles as its React key. */}
              {report.explanation.map((line) => <li key={line} style={{ color: "#cbd5e1" }}>{line}</li>)}
            </ul>
            {/* The caveat from the server, e.g. that a file with no marks is not proven real. Shown with every report. */}
            <div style={{ ...small, marginTop: "6px", fontStyle: "italic" }}>{report.limits}</div>
          </div>
        )}
      </div>
    </div>
  );
}
