import { useEffect, useState } from "react"; // React hooks: useState holds a value that re-renders on change; useEffect runs side effects
import "./App.css"; // Vite bundles imported CSS into the page
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
 *
 * Backend route called here: GET /health, once, to show whether the server is up and which job
 * queue it uses. Each tab body calls its own routes.
 */

// Where the FastAPI backend listens (docs/10-DEPLOYMENT.md runs it on port 8000).
// Every panel receives it as a prop instead of hard-coding it again.
const API_BASE = "http://localhost:8000";

// [key, label] pairs: the key decides which <section> shows, the label is the button text.
const TABS = [
  ["create", "Create video"],
  ["live", "Live avatar"],
  ["verify", "Verify a file"],
];

export default function App() {
  // useState returns [current value, setter]. Calling the setter re-renders this component.
  const [tab, setTab] = useState("create"); // which tab is showing
  const [health, setHealth] = useState({ text: "checking…", ok: true }); // header text and whether the dot is green

  // useEffect runs after the first render. The empty dependency list [] below means "only once".
  // The function it returns is the cleanup, run when the component goes away.
  useEffect(() => {
    let ignore = false; // set by the cleanup so a reply that lands late is not applied
    fetch(`${API_BASE}/health`) // fetch returns a Promise; .then runs when the reply arrives
      .then((res) => res.json()) // the body is read as JSON, which is another Promise
      .then((data) => !ignore && setHealth({ text: `${data.status} (${data.queueBackend})`, ok: data.status === "ok" })) // `a && b` only runs b when a is true
      .catch(() => !ignore && setHealth({ text: "backend offline", ok: false })); // a network error means the server is not reachable
    return () => { ignore = true; }; // cleanup: in development StrictMode mounts twice, so the first reply is ignored
  }, []); // no dependencies: run once after the first render

  return ( // JSX: HTML-like syntax that the build compiles to plain JavaScript function calls
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Avatar Creator Studio</h1>
          <p>Type a script, pick a face, get a talking video. Every output is watermarked and signed.</p>
        </div>
        <div className="status" id="backend-status"><span className={`status-dot${health.ok ? "" : " down"}`} />{health.text}</div>
      </header>

      {/* role and aria-selected tell screen readers this row behaves as tabs. */}
      <nav className="tabs" role="tablist">
        {/* map turns each [key, label] pair into a button. React needs a stable `key` on items in a list. */}
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
