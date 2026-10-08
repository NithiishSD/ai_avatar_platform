import { useEffect, useState } from "react";
import "./App.css";
import CreateVideo from "./CreateVideo";
import LivePanel from "./LivePanel";
import ProvenancePanel from "./ProvenancePanel";

/*
 * The studio shell: a header with the backend's health, and three tabs.
 *
 *   Create video  - script + voice + face -> a finished, watermarked video (CreateVideo.jsx)
 *   Live avatar   - talk to the avatar in real time, by text or microphone (LivePanel.jsx)
 *   Verify a file - was this audio or video made here? (ProvenancePanel.jsx)
 *
 * The tab bodies stay mounted when hidden, so switching tabs does not lose what was typed or a
 * running live session.
 */

const API_BASE = "http://localhost:8000";

const TABS = [
  ["create", "Create video"],
  ["live", "Live avatar"],
  ["verify", "Verify a file"],
];

export default function App() {
  const [tab, setTab] = useState("create");
  const [health, setHealth] = useState({ text: "checking…", ok: true });

  useEffect(() => {
    let ignore = false;
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => !ignore && setHealth({ text: `${data.status} (${data.queueBackend})`, ok: data.status === "ok" }))
      .catch(() => !ignore && setHealth({ text: "backend offline", ok: false }));
    return () => { ignore = true; };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Avatar Creator Studio</h1>
          <p>Type a script, pick a face, get a talking video. Every output is watermarked and signed.</p>
        </div>
        <div className="status" id="backend-status"><span className={`status-dot${health.ok ? "" : " down"}`} />{health.text}</div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map(([key, label]) => (
          <button key={key} type="button" role="tab" id={`tab-${key}`} className="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </nav>

      {/* hidden, not unmounted: a live session or a half-written script survives a tab switch */}
      <section hidden={tab !== "create"}><CreateVideo apiBase={API_BASE} /></section>
      <section hidden={tab !== "live"} className="tab-body"><LivePanel apiBase={API_BASE} /></section>
      <section hidden={tab !== "verify"} className="tab-body"><ProvenancePanel apiBase={API_BASE} /></section>
    </div>
  );
}
