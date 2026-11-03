import { useEffect, useState } from "react"; // React hooks: useState holds a value that re-renders on change; useEffect runs side effects
import "./App.css"; // Vite bundles imported CSS into the page
import CreateVideo from "./CreateVideo";
import LivePanel from "./LivePanel";
import ProvenancePanel from "./ProvenancePanel";
import DemoShowcase from "./DemoShowcase";

/*
 * The studio shell: a header with the backend's health, and dynamic tabs.
 *
 * In normal mode, the application displays the three actual production tabs:
 *   Create video  - script + voice + face -> a finished, watermarked video (CreateVideo.jsx)
 *   Live avatar   - talk to the avatar in real time, by text or microphone (LivePanel.jsx)
 *   Verify a file - was this audio or video made here? (ProvenancePanel.jsx)
 *
 * When Demo Mode is toggled ON (in header, or via ?demo=true or run_demo.sh):
 *   Demo & Compare - instant evaluation suite (video engines, audio cloner, watermarks)
 */

// Where the FastAPI backend listens (docs/10-DEPLOYMENT.md runs it on port 8000).
// Every panel receives it as a prop instead of hard-coding it again.
const API_BASE = "http://localhost:8000";

function getInitialDemoMode() {
  if (typeof window === "undefined") return false;
  const params = new URLSearchParams(window.location.search);
  if (params.get("demo") === "true" || params.get("demo") === "1") return true;
  if (params.get("demo") === "false" || params.get("demo") === "0") return false;
  return localStorage.getItem("ai_avatar_demo_mode") === "true";
}

export default function App() {
  const [demoMode, setDemoMode] = useState(getInitialDemoMode);
  // Default to "create" (the actual core application) unless demo was explicitly requested
  const [tab, setTab] = useState(() => (getInitialDemoMode() ? "demo" : "create"));
  const [health, setHealth] = useState({ text: "checking…", ok: true }); // header text and whether the dot is green

  // useEffect runs after the first render. The empty dependency list [] below means "only once".
  // The function it returns is the cleanup, run when the component goes away.
  useEffect(() => {
    let ignore = false; // set by the cleanup so a reply that lands late is not applied
    fetch(`${API_BASE}/health`) // fetch returns a Promise; .then runs when the reply arrives
      .then((res) => res.json()) // the body is read as JSON, which is another Promise
      .then((data) => {
        if (ignore) return;
        setHealth({ text: `${data.status} (${data.queueBackend})`, ok: data.status === "ok" });
        // If server announces DEMO_MODE=true and user hasn't explicitly set localStorage:
        if (data.demoMode && localStorage.getItem("ai_avatar_demo_mode") === null) {
          setDemoMode(true);
        }
      })
      .catch(() => !ignore && setHealth({ text: "backend offline", ok: false })); // a network error means the server is not reachable
    return () => { ignore = true; }; // cleanup: in development StrictMode mounts twice, so the first reply is ignored
  }, []); // no dependencies: run once after the first render

  const toggleDemoMode = () => {
    setDemoMode((prev) => {
      const next = !prev;
      localStorage.setItem("ai_avatar_demo_mode", next ? "true" : "false");
      if (next) {
        setTab("demo");
      } else if (tab === "demo") {
        setTab("create");
      }
      return next;
    });
  };

  const tabs = [
    ...(demoMode ? [["demo", "Demo & Compare"]] : []),
    ["create", "Create video"],
    ["live", "Live avatar"],
    ["verify", "Verify a file"],
  ];

  return ( // JSX: HTML-like syntax that the build compiles to plain JavaScript function calls
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Avatar Creator Studio</h1>
          <p>Edge-Constrained Multimodal Platform: Speech Synthesis, Forced Alignment, Neural Lip-Sync &amp; Live Streaming.</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* Demo Mode Toggle Switch */}
          <button
            type="button"
            className={`demo-toggle-pill ${demoMode ? "active" : ""}`}
            onClick={toggleDemoMode}
            id="demo-mode-toggle"
            title="Toggle between actual application and evaluation demo mode"
          >
            <span className="toggle-dot" />
            <span>{demoMode ? "Demo Mode: ON" : "Demo Mode: OFF"}</span>
          </button>
          <div className="status" id="backend-status">
            <span className={`status-dot${health.ok ? "" : " down"}`} />
            {health.text}
          </div>
        </div>
      </header>

      {/* role and aria-selected tell screen readers this row behaves as tabs. */}
      <nav className="tabs" role="tablist">
        {/* map turns each [key, label] pair into a button. React needs a stable `key` on items in a list. */}
        {tabs.map(([key, label]) => (
          <button key={key} type="button" role="tab" id={`tab-${key}`} className="tab" aria-selected={tab === key} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </nav>

      {/* hidden, not unmounted: a live session or a half-written script survives a tab switch */}
      {demoMode && (
        <section hidden={tab !== "demo"} className="tab-body demo-tab-body">
          <DemoShowcase apiBase={API_BASE} />
        </section>
      )}
      <section hidden={tab !== "create"}><CreateVideo apiBase={API_BASE} /></section>
      <section hidden={tab !== "live"} className="tab-body"><LivePanel apiBase={API_BASE} /></section>
      <section hidden={tab !== "verify"} className="tab-body"><ProvenancePanel apiBase={API_BASE} /></section>
    </div>
  );
}
