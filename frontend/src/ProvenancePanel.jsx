import { useState } from "react";

/*
 * "Is this file ours?" - upload an audio or video file (and optionally its manifest) to
 * POST /api/v1/provenance/verify and show the evidence and the verdict.
 *
 * The server reports four kinds of evidence separately (audio watermark, video watermark, signed
 * manifest, audit record) because they fail in different ways. The panel shows the verdict with
 * its plain-language meaning, and keeps the "no evidence proves nothing" caveat visible: a file
 * with no marks is not shown to be real.
 */

const card = { background: "#0f172a", border: "1px solid #334155", borderRadius: "8px", padding: "12px 14px", marginBottom: "12px" };
const small = { fontSize: "0.78rem", color: "#94a3b8" };

const TONE = {
  authentic_original: "#4ade80",
  ours_modified: "#fbbf24",
  manifest_for_another_file: "#fbbf24",
  tampered_manifest: "#f87171",
  foreign_manifest: "#f87171",
  no_evidence: "#94a3b8",
};

function markLine(label, mark) {
  if (!mark) return null;
  if (mark.available === false) return `${label}: could not be checked (${mark.reason})`;
  const bits = mark.tagBitsMatching ?? mark.bitsMatching;
  const of = mark.tagBitsMatching !== undefined ? 128 : 16;
  return `${label}: ${mark.detected ? "found" : "not found"} (${bits}/${of} bits match)`;
}

export default function ProvenancePanel({ apiBase }) {
  const [file, setFile] = useState(null);
  const [manifestFile, setManifestFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);

  const verify = async (event) => {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    setReport(null);
    const form = new FormData();
    form.append("file", file);
    if (manifestFile) form.append("manifest", manifestFile);
    try {
      const res = await fetch(`${apiBase}/api/v1/provenance/verify`, { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
      setReport(body);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div id="provenance-panel">
      <h2 style={{ marginTop: "1.5rem" }}>Verify a file</h2>
      <div style={card}>
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
        {error && <div className="alert error" id="verify-error" style={{ marginTop: "8px" }}>{error}</div>}
        {report && (
          <div id="verify-result" style={{ marginTop: "10px" }}>
            <div id="verify-verdict" data-verdict={report.verdict} style={{ fontWeight: 700, color: TONE[report.verdict] || "#e2e8f0" }}>
              {report.verdict}
            </div>
            <div style={{ ...small, color: "#e2e8f0", marginTop: "4px" }}>{report.meaning}</div>
            <ul style={{ ...small, margin: "8px 0 0 16px", padding: 0 }}>
              {[markLine("video watermark", report.watermarks.video), markLine("audio watermark", report.watermarks.audio),
                report.manifest ? `manifest: ${report.manifest.trustworthy ? "valid, ours, matches this file" : report.manifest.validSignature ? "valid signature, but not usable for this file" : "signature does NOT match"}` : "manifest: none supplied or found",
                report.record ? `audit record: issued ${report.record.ts}, job ${report.record.details.job ?? "?"}` : "audit record: none found",
              ].filter(Boolean).map((line) => <li key={line}>{line}</li>)}
              {report.explanation.map((line) => <li key={line} style={{ color: "#cbd5e1" }}>{line}</li>)}
            </ul>
            <div style={{ ...small, marginTop: "6px", fontStyle: "italic" }}>{report.limits}</div>
          </div>
        )}
      </div>
    </div>
  );
}
