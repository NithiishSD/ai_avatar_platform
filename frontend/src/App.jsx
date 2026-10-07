import { useEffect, useState, useCallback } from "react";
import "./App.css";
import AvatarPanel from "./AvatarPanel";

const API_BASE = "http://localhost:8000";

/* ------------------------------------------------------------------ */
/* Model routing hints shown in the UI                                 */
/* ------------------------------------------------------------------ */
const MODE_INFO = {
  fast:         { label: "⚡ Fast (Kokoro 82M)",         hint: "Sub-second English synthesis. Best for real-time use." },
  clone:        { label: "🎤 Voice Clone (XTTS-v2)",     hint: "Zero-shot cloning from a reference recording. Multilingual." },
  high_quality: { label: "🏆 High Quality (Higgs 3B)",   hint: "Ultra-high MOS, multilingual. Slower, GPU-intensive." },
  dialogue:     { label: "💬 Dialogue (Dia 1.6B)",        hint: "Multi-speaker with [S1] / [S2] tags. Great for conversations." },
  multilingual: { label: "🌍 Multilingual (MMS-TTS)",    hint: "Facebook MMS-TTS. Forces the MMS path for 1000+ languages." },
};

/* Phase 3 emotion presets. Descriptions mirror backend/emotion_engine.py. */
const EMOTION_INFO = {
  "":           { label: "— neutral —",      hint: "No prosody transform is applied." },
  joy:          { label: "😊 Joy",            hint: "Brighter, faster, rising pitch contour." },
  anger:        { label: "😠 Anger",          hint: "Louder, faster, harsher." },
  sorrow:       { label: "😢 Sorrow",         hint: "Lower, slower, falling contour." },
  authority:    { label: "🎓 Authority",      hint: "Deeper and level, but still projecting." },
  calm:         { label: "😌 Calm",           hint: "Soft and unhurried." },
  excitement:   { label: "🤩 Excitement",     hint: "Fast, loud, strongly rising." },
};

const ENGLISH_CODES = ["en", "en-us", "en-gb", "en-au", "en-ca"];

/*
 * Mirror of emotion_engine.to_render_emotion_vector for the render-job payload.
 * happy / neutral / eyeblinkRate are the Phase 0 required fields; the named
 * emotions ride along in the optional Phase 3 fields.
 */
const EMOTION_RENDER_HINTS = {
  joy:        { happy: 1.0,  blink: 1.6 },
  anger:      { happy: 0.0,  blink: 2.4 },
  sorrow:     { happy: 0.0,  blink: 0.6 },
  authority:  { happy: 0.1,  blink: 0.8 },
  calm:       { happy: 0.35, blink: 0.9 },
  excitement: { happy: 0.9,  blink: 2.0 },
};

function buildRenderEmotionVector(emotionReport) {
  const vector = emotionReport?.vector;
  if (!vector) return { happy: 0.8, neutral: 0.2, eyeblinkRate: 1.2 };

  let happy = 0;
  let blink = 0;
  let weightSum = 0;
  const extras = {};
  for (const [name, weight] of Object.entries(vector)) {
    if (name === "neutral" || !(weight > 0)) continue;
    const hint = EMOTION_RENDER_HINTS[name];
    if (!hint) continue;
    happy += hint.happy * weight;
    blink += hint.blink * weight;
    weightSum += weight;
    extras[name] = Number(weight.toFixed(4));
  }
  blink += 1.0 * Math.max(0, 1 - weightSum);   // neutral blink rate
  happy = Math.min(1, Math.max(0, happy));

  return {
    happy: Number(happy.toFixed(4)),
    neutral: Number((1 - happy).toFixed(4)),
    eyeblinkRate: Number(Math.min(10, Math.max(0, blink)).toFixed(4)),
    ...extras,
  };
}

function ModelBadge({ modelUsed }) {
  if (!modelUsed) return null;
  const colors = {
    "kokoro":     "#6ee7b7",
    "xtts-v2":    "#93c5fd",
    "higgs-tts-2":"#fbbf24",
    "dia-1.6b":   "#c4b5fd",
    "mms-tts":    "#f0abfc",
  };
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 10px",
      borderRadius: "999px",
      fontSize: "0.75rem",
      fontWeight: 700,
      background: colors[modelUsed] ?? "#4b5563",
      color: "#0f172a",
      marginTop: "0.4rem",
    }}>
      {modelUsed}
    </span>
  );
}

function App() {
  const [text, setText]           = useState("Hello! I am your AI avatar running from the connected backend.");
  const [mode, setMode]           = useState("fast");
  const [language, setLanguage]   = useState("en");
  const [quality, setQuality]     = useState("balanced");
  const [style, setStyle]         = useState("");
  const [speed, setSpeed]         = useState(1.0);
  const [pitch, setPitch]         = useState(1.0);
  const [returnAlignment, setReturnAlignment] = useState(true);
  const [selectedSample, setSelectedSample]   = useState("");

  /* Phase 3: emotion prosody, multilingual catalogue, quality auditing */
  const [emotion, setEmotion]                 = useState("");
  const [emotionIntensity, setEmotionIntensity] = useState(1.0);
  const [auditQuality, setAuditQuality]       = useState(false);
  const [languageQuery, setLanguageQuery]     = useState("");
  const [languageResults, setLanguageResults] = useState([]);
  const [languageTotal, setLanguageTotal]     = useState(0);
  const [languageInfo, setLanguageInfo]       = useState(null);
  const [qualityReport, setQualityReport]     = useState(null);
  const [emotionReport, setEmotionReport]     = useState(null);
  const [latencyMs, setLatencyMs]             = useState(null);

  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [taskId, setTaskId]       = useState("");
  const [taskStatus, setTaskStatus] = useState("idle");
  const [modelUsed, setModelUsed] = useState(null);
  const [error, setError]         = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [audioUrl, setAudioUrl]   = useState("");
  const [phonemeTimestamps, setPhonemeTimestamps] = useState([]);
  const [alignmentMethod, setAlignmentMethod] = useState("");
  const [activeViseme, setActiveViseme] = useState("viseme_sil");
  const [showTimeline, setShowTimeline] = useState(false);

  // Voice samples from inputs/ folder (for clone mode)
  const [voiceSamples, setVoiceSamples]       = useState([]);
  const [samplesLoading, setSamplesLoading]   = useState(false);
  const [samplesError, setSamplesError]       = useState("");
  const [inputsDir, setInputsDir]             = useState("");
  const [supportedFormats, setSupportedFormats] = useState([]);

  /* ---------- computed predicted model (client-side hint) ----------- */
  const predictedModel = (() => {
    if (mode === "dialogue" || style === "dialogue" || text.includes("[S1]") || text.includes("[S2]")) return "dia-1.6b";
    if (mode === "clone") return "xtts-v2";
    if (mode === "high_quality" || quality === "high") return "higgs-tts-2";
    if (mode === "multilingual") return languageInfo?.mmsSupported ? "mms-tts" : "unsupported";
    const lang = language.toLowerCase().replace("_", "-");
    if (ENGLISH_CODES.includes(lang)) return "kokoro";
    // Mirrors backend routing: MMS-TTS wins for the languages it covers.
    return languageInfo?.mmsSupported ? "mms-tts" : "higgs-tts-2";
  })();

  /* ---------- voice sample scanning --------------------------------- */
  /* Called from the events that need it - switching to clone mode and the
     refresh button - not from an effect. React's guidance for state that
     follows a user action is to update it in that action's handler. */
  const fetchSamples = useCallback(async () => {
    setSamplesLoading(true);
    setSamplesError("");
    try {
      const res = await fetch(`${API_BASE}/api/v1/audio/samples`);
      if (!res.ok) throw new Error(`Samples request failed: ${res.status}`);
      const data = await res.json();
      setVoiceSamples(data.samples ?? []);
      setInputsDir(data.inputs_dir ?? "");
      setSupportedFormats(data.supported_formats ?? []);
      // Keep the user's choice, otherwise default to the first sample. The
      // functional update reads the current value without making this
      // callback depend on it (it used to be rebuilt on every selection).
      if (data.samples?.length > 0) {
        const first = data.samples[0].path;
        setSelectedSample((current) => current || first);
      }
    } catch (err) {
      setSamplesError(err.message || "Could not load voice samples");
    } finally {
      setSamplesLoading(false);
    }
  }, []);

  /* ---------- requests that follow state ----------------------------- */
  /* Each effect below owns an `ignore` flag that its cleanup sets. If the
     input changes again before a response arrives, the stale response is
     dropped instead of overwriting the newer one: without it, a slow reply
     for the previous language could land last and show the wrong one. */

  useEffect(() => {
    let ignore = false;
    fetch(`${API_BASE}/health`)
      .then((res) => {
        if (!res.ok) throw new Error("Backend unavailable");
        return res.json();
      })
      .then((data) => {
        if (!ignore) setBackendStatus(`${data.status} (${data.queueBackend})`);
      })
      .catch((err) => {
        if (ignore) return;
        setBackendStatus("Offline");
        setError(err.message || "Unable to reach backend");
      });
    return () => { ignore = true; };
  }, []);

  /* Debounced catalogue search: 1077 languages, one request per pause. The
     first run, with an empty query, loads the initial list. */
  useEffect(() => {
    let ignore = false;
    const timer = window.setTimeout(() => {
      fetch(`${API_BASE}/api/v1/audio/languages?q=${encodeURIComponent(languageQuery)}&limit=40`)
        .then((res) => {
          if (!res.ok) throw new Error(`Language search failed: ${res.status}`);
          return res.json();
        })
        .then((data) => {
          if (ignore) return;
          setLanguageResults(data.languages ?? []);
          setLanguageTotal(data.total ?? 0);
        })
        .catch((err) => {
          if (!ignore) setError(err.message || "Could not search languages");
        });
    }, 250);
    return () => {
      ignore = true;
      window.clearTimeout(timer);
    };
  }, [languageQuery]);

  /* Resolve whichever code is selected, so the UI can say which backend
     will actually speak it before the user hits Generate. */
  useEffect(() => {
    if (!language) return undefined;
    let ignore = false;
    fetch(`${API_BASE}/api/v1/audio/languages/${encodeURIComponent(language)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!ignore && data) setLanguageInfo(data);
      })
      .catch(() => {
        if (!ignore) setLanguageInfo(null);
      });
    return () => { ignore = true; };
  }, [language]);

  /* ---------- shared reader for a synthesis response ---------------- */
  const applySynthesisPayload = useCallback((payload) => {
    if (payload.modelUsed) setModelUsed(payload.modelUsed);
    if (payload.phonemeTimestamps) setPhonemeTimestamps(payload.phonemeTimestamps);
    if (payload.alignmentMethod) setAlignmentMethod(payload.alignmentMethod);
    if (payload.qualityReport) setQualityReport(payload.qualityReport);
    if (payload.emotion) setEmotionReport(payload.emotion);
    if (payload.latencyMs != null) setLatencyMs(payload.latencyMs);
  }, []);

  /* ---------- audio playback time synchronization ------------------- */
  const handleAudioTimeUpdate = (e) => {
    const currentMs = e.target.currentTime * 1000;
    if (!phonemeTimestamps || phonemeTimestamps.length === 0) return;
    const current = phonemeTimestamps.find(
      (t) => currentMs >= t.startMs && currentMs < t.endMs
    );
    if (current) {
      setActiveViseme(current.viseme);
    } else {
      setActiveViseme("viseme_sil");
    }
  };

  const handleAudioEnded = () => {
    setActiveViseme("viseme_sil");
  };

  /* ---------- task status polling ----------------------------------- */
  useEffect(() => {
    if (!taskId) return;
    const terminal = ["SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"];
    if (terminal.includes(taskStatus)) return;

    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/audio/synthesize/${taskId}`);
        if (!res.ok) throw new Error("Task status check failed");
        const payload = await res.json();
        setTaskStatus(payload.status);
        applySynthesisPayload(payload);
        if (payload.status === "SUCCESS") {
          setAudioUrl(`${API_BASE}/outputs/speech.wav?t=${Date.now()}`);
        }
      } catch (err) {
        setError(err.message || "Could not poll task status");
      }
    }, 1500);

    return () => window.clearTimeout(timer);
    // applySynthesisPayload is a stable useCallback; listing it is free and
    // keeps the effect honest if that ever changes.
  }, [taskId, taskStatus, applySynthesisPayload]);

  /* ---------- synthesis submit -------------------------------------- */
  const handleSynthesize = async (event) => {
    event.preventDefault();
    setError("");
    setAudioUrl("");
    setModelUsed(null);
    setPhonemeTimestamps([]);
    setQualityReport(null);
    setEmotionReport(null);
    setLatencyMs(null);
    setIsSubmitting(true);

    try {
      const body = {
        text,
        mode,
        language,
        quality,
        speed: parseFloat(speed),
        pitch: parseFloat(pitch),
        returnAlignment,
        auditQuality,
      };
      if (style) body.style = style;
      if (emotion) {
        body.emotion = emotion;
        body.emotionIntensity = parseFloat(emotionIntensity);
      }
      // For clone mode, pass the selected input file path
      if (mode === "clone") {
        if (!selectedSample) {
          throw new Error(
            "No voice sample selected. Place a WAV/MP3/FLAC file in inputs/ and refresh the sample list."
          );
        }
        body.speakerWav = selectedSample;
      }

      const res = await fetch(`${API_BASE}/api/v1/audio/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || `Request failed ${res.status}`);
      if (!payload.taskId) throw new Error("Missing taskId in response");

      setTaskId(payload.taskId);
      const newStatus = payload.status || "QUEUED";
      setTaskStatus(newStatus);
      applySynthesisPayload(payload);
      if (newStatus === "SUCCESS") {
        setAudioUrl(`${API_BASE}/outputs/speech.wav?t=${Date.now()}`);
      }
    } catch (err) {
      setError(err.message || "Could not start synthesis");
    } finally {
      setIsSubmitting(false);
    }
  };

  /* ---------- render ------------------------------------------------ */
  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>AI Avatar Creator Studio</h1>
          <p>Phase 3 — Multilingual Synthesis, Emotion Prosody & Quality Auditing</p>
        </div>
        <div className="status">
          <span className="status-dot" />
          {backendStatus}
        </div>
      </header>

      <main className="workspace">
        {/* -------- LEFT: Controls -------- */}
        <section className="canvas-panel">
          <div className="panel-header">
            <h2>Voice, Language & Emotion Controls</h2>
            <span>Phase 3 Neural Stack</span>
          </div>

          <form className="studio-form" onSubmit={handleSynthesize}>
            <label>
              Text
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                placeholder="Enter text • Use [S1] / [S2] tags for dialogue mode"
              />
            </label>

            {/* Mode row */}
            <label>
              Synthesis Mode
              <select
                id="mode-select"
                value={mode}
                onChange={(e) => {
                  setMode(e.target.value);
                  if (e.target.value === "clone") fetchSamples();
                }}
              >
                {Object.entries(MODE_INFO).map(([val, { label }]) => (
                  <option key={val} value={val}>{label}</option>
                ))}
              </select>
              <span style={{ fontSize: "0.75rem", color: "#9ca3af", marginTop: "4px", display: "block" }}>
                {MODE_INFO[mode]?.hint}
              </span>
            </label>

            {/* Clone mode — voice sample picker from inputs/ */}
            {mode === "clone" && (
              <div style={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "10px",
                padding: "14px",
              }}>
                <div style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "10px",
                }}>
                  <span style={{ fontWeight: 600, fontSize: "0.85rem" }}>🎤 Voice Reference Sample</span>
                  <button
                    type="button"
                    id="refresh-samples-btn"
                    onClick={fetchSamples}
                    disabled={samplesLoading}
                    style={{
                      background: "transparent",
                      border: "1px solid #475569",
                      color: "#94a3b8",
                      borderRadius: "6px",
                      padding: "3px 10px",
                      fontSize: "0.75rem",
                      cursor: "pointer",
                    }}
                  >
                    {samplesLoading ? "Scanning…" : "↻ Refresh"}
                  </button>
                </div>

                {samplesError && (
                  <div style={{ color: "#f87171", fontSize: "0.8rem", marginBottom: "8px" }}>
                    {samplesError}
                  </div>
                )}

                {voiceSamples.length === 0 && !samplesLoading ? (
                  <div style={{ fontSize: "0.8rem", color: "#64748b", lineHeight: 1.6 }}>
                    <strong style={{ color: "#f59e0b" }}>No audio files found in inputs/</strong><br />
                    Place a recording in:<br />
                    <code style={{ fontSize: "0.75rem", color: "#94a3b8" }}>{inputsDir || "…/inputs/"}</code><br />
                    <span style={{ marginTop: "6px", display: "block" }}>
                      Supported: {supportedFormats.join(", ") || "WAV, MP3, FLAC, OGG, M4A…"}
                    </span>
                    <span style={{ marginTop: "4px", display: "block", color: "#475569" }}>
                      Ideal: 30–60 s of clean speech. Auto-converted to WAV 24 kHz.
                    </span>
                  </div>
                ) : (
                  <>
                    <select
                      id="sample-select"
                      value={selectedSample}
                      onChange={(e) => setSelectedSample(e.target.value)}
                      style={{ marginBottom: "8px" }}
                    >
                      {voiceSamples.map((s) => (
                        <option key={s.path} value={s.path}>
                          {s.filename} ({s.duration_label}, {s.format.toUpperCase()})
                        </option>
                      ))}
                    </select>

                    {/* Selected sample details */}
                    {(() => {
                      const selected = voiceSamples.find((s) => s.path === selectedSample);
                      if (!selected) return null;
                      return (
                        <div style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr 1fr",
                          gap: "6px",
                          fontSize: "0.72rem",
                          color: "#94a3b8",
                          marginTop: "6px",
                        }}>
                          <div>📏 {selected.duration_label}</div>
                          <div>🎵 {selected.sample_rate > 0 ? `${(selected.sample_rate / 1000).toFixed(1)} kHz` : "—"}</div>
                          <div>📁 {(selected.size_bytes / 1024).toFixed(0)} KB</div>
                          <div style={{ gridColumn: "1/-1", marginTop: "2px" }}>
                            {selected.ready_for_cloning
                              ? <span style={{ color: "#4ade80" }}>✅ Ready (WAV 24 kHz mono)</span>
                              : <span style={{ color: "#fbbf24" }}>⚡ Will auto-convert on synthesis</span>}
                          </div>
                        </div>
                      );
                    })()}
                  </>
                )}
              </div>
            )}

            {/* Language: searchable picker over the full MMS-TTS catalogue */}
            <div style={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "10px",
              padding: "14px",
            }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "8px",
              }}>
                <span style={{ fontWeight: 600, fontSize: "0.85rem" }}>🌍 Language</span>
                <span style={{ fontSize: "0.72rem", color: "#64748b" }}>
                  {languageTotal > 0 ? `${languageTotal} MMS-TTS languages` : "loading…"}
                </span>
              </div>

              <input
                id="language-search"
                type="text"
                value={languageQuery}
                onChange={(e) => setLanguageQuery(e.target.value)}
                placeholder="Search by name or ISO-639-3 code — e.g. Tamil, swh, Yoruba"
                style={{ marginBottom: "8px" }}
              />

              <select
                id="language-select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                {/* Kokoro's English path is not in the MMS catalogue, so it is
                    offered explicitly alongside the search results. */}
                <option value="en">English (en) → Kokoro</option>
                {languageResults
                  .filter((entry) => entry.iso3 !== "eng")
                  .map((entry) => (
                    <option key={entry.iso3} value={entry.iso3}>
                      {entry.name} ({entry.iso3})
                    </option>
                  ))}
              </select>

              {languageInfo && (
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "4px",
                  fontSize: "0.72rem",
                  color: "#94a3b8",
                  marginTop: "8px",
                }}>
                  <div>🔤 {languageInfo.name} ({languageInfo.iso3 || "—"})</div>
                  <div>
                    {languageInfo.mmsSupported
                      ? <span style={{ color: "#4ade80" }}>✅ MMS-TTS available</span>
                      : <span style={{ color: "#fbbf24" }}>⚡ No MMS checkpoint → Higgs</span>}
                  </div>
                  {languageInfo.mmsModel && (
                    <div style={{ gridColumn: "1/-1", color: "#475569" }}>
                      <code style={{ fontSize: "0.7rem" }}>{languageInfo.mmsModel}</code>
                    </div>
                  )}
                  {languageInfo.xttsSupported && (
                    <div style={{ gridColumn: "1/-1", color: "#60a5fa" }}>
                      🎤 XTTS-v2 can also clone into this language
                    </div>
                  )}
                </div>
              )}
            </div>

            <label>
              Quality
              <select id="quality-select" value={quality} onChange={(e) => setQuality(e.target.value)}>
                <option value="fast">fast</option>
                <option value="balanced">balanced</option>
                <option value="high">high → Higgs</option>
              </select>
            </label>

            {/* Phase 3: emotion prosody */}
            <div style={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "10px",
              padding: "14px",
            }}>
              <span style={{ fontWeight: 600, fontSize: "0.85rem", display: "block", marginBottom: "8px" }}>
                🎭 Emotion Prosody
              </span>
              <select
                id="emotion-select"
                value={emotion}
                onChange={(e) => setEmotion(e.target.value)}
              >
                {Object.entries(EMOTION_INFO).map(([value, { label }]) => (
                  <option key={value || "neutral"} value={value}>{label}</option>
                ))}
              </select>
              <span style={{ fontSize: "0.72rem", color: "#9ca3af", marginTop: "4px", display: "block" }}>
                {EMOTION_INFO[emotion]?.hint}
              </span>

              {emotion && (
                <label style={{ marginTop: "10px", display: "block" }}>
                  Intensity: <span style={{ color: "#38bdf8", fontWeight: "bold" }}>
                    {Math.round(emotionIntensity * 100)}%
                  </span>
                  <input
                    id="emotion-intensity"
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={emotionIntensity}
                    onChange={(e) => setEmotionIntensity(e.target.value)}
                    style={{ width: "100%", accentColor: "#a855f7" }}
                  />
                  <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                    The remainder of the vector stays neutral.
                  </span>
                </label>
              )}
            </div>

            {/* Prosody controls: Speed & Pitch */}
            <div className="grid-two" style={{ marginTop: "4px" }}>
              <label>
                Speed / Rhythm: <span style={{ color: "#38bdf8", fontWeight: "bold" }}>{speed}x</span>
                <input
                  type="range"
                  min="0.6"
                  max="1.6"
                  step="0.05"
                  value={speed}
                  onChange={(e) => setSpeed(e.target.value)}
                  style={{ width: "100%", accentColor: "#38bdf8" }}
                />
              </label>

              <label>
                Pitch Shift: <span style={{ color: "#38bdf8", fontWeight: "bold" }}>{pitch}x</span>
                <input
                  type="range"
                  min="0.7"
                  max="1.4"
                  step="0.05"
                  value={pitch}
                  onChange={(e) => setPitch(e.target.value)}
                  style={{ width: "100%", accentColor: "#38bdf8" }}
                />
              </label>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px", margin: "6px 0" }}>
              <input
                type="checkbox"
                id="alignment-checkbox"
                checked={returnAlignment}
                onChange={(e) => setReturnAlignment(e.target.checked)}
                style={{ width: "auto", cursor: "pointer" }}
              />
              <label htmlFor="alignment-checkbox" style={{ fontSize: "0.82rem", color: "#e2e8f0", cursor: "pointer", margin: 0 }}>
                ⚡ Extract Millisecond Phoneme & Viseme Timestamps
              </label>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px", margin: "6px 0" }}>
              <input
                type="checkbox"
                id="audit-checkbox"
                checked={auditQuality}
                onChange={(e) => setAuditQuality(e.target.checked)}
                style={{ width: "auto", cursor: "pointer" }}
              />
              <label htmlFor="audit-checkbox" style={{ fontSize: "0.82rem", color: "#e2e8f0", cursor: "pointer", margin: 0 }}>
                📊 Audit Speech Quality (SQUIM MOS / PESQ) — adds ~1s
              </label>
            </div>

            <label>
              Style <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>(optional)</span>
              <select id="style-select" value={style} onChange={(e) => setStyle(e.target.value)}>
                <option value="">— none —</option>
                <option value="dialogue">dialogue (multi-speaker)</option>
                <option value="expressive">expressive</option>
                <option value="narration">narration</option>
              </select>
            </label>

            {/* Predicted model hint */}
            <div style={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              padding: "10px 14px",
              fontSize: "0.8rem",
              color: "#94a3b8",
              marginBottom: "4px",
            }}>
              🤖 Router will select: <ModelBadge modelUsed={predictedModel} />
            </div>

            <div className="controls">
              <button type="submit" id="generate-btn" disabled={isSubmitting}>
                {isSubmitting ? "Synthesizing..." : "Generate Speech"}
              </button>
            </div>
          </form>

          {error ? <div className="alert error">{error}</div> : null}
        </section>

        {/* -------- RIGHT: Status & Viseme Visualizer panel -------- */}
        <section className="info-panel">
          <h2>Synthesis & Lip-Sync Status</h2>

          <div className="info-card">
            <label>Task ID</label>
            <strong>{taskId || "Not started"}</strong>
          </div>

          <div className="info-card">
            <label>Task Status</label>
            <strong>{taskStatus}</strong>
          </div>

          <div className="info-card">
            <label>Model Used</label>
            {modelUsed
              ? <ModelBadge modelUsed={modelUsed} />
              : <strong style={{ color: "#4b5563" }}>—</strong>
            }
          </div>

          {/* Viseme / Facial Shape Indicator */}
          <div className="info-card" style={{
            background: "linear-gradient(135deg, #1e293b, #0f172a)",
            border: "1px solid #3b82f6",
          }}>
            <label style={{ color: "#60a5fa" }}>👄 Live Viseme Sync</label>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginTop: "6px",
            }}>
              <span style={{
                fontSize: "1.1rem",
                fontWeight: "bold",
                color: activeViseme === "viseme_sil" ? "#64748b" : "#38bdf8",
              }}>
                {activeViseme}
              </span>
              <span style={{
                padding: "2px 8px",
                borderRadius: "4px",
                fontSize: "0.72rem",
                background: activeViseme === "viseme_sil" ? "#334155" : "#1d4ed8",
                color: "#f8fafc",
              }}>
                {activeViseme === "viseme_sil" ? "Rest / Silence" : "Active Speaking"}
              </span>
            </div>
          </div>

          {/* Phoneme timestamps timeline */}
          {phonemeTimestamps && phonemeTimestamps.length > 0 && (
            <div className="info-card">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <label>Aligned Phonemes ({phonemeTimestamps.length})</label>
                <button
                  type="button"
                  onClick={() => setShowTimeline(!showTimeline)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "#38bdf8",
                    fontSize: "0.75rem",
                    cursor: "pointer",
                    textDecoration: "underline",
                  }}
                >
                  {showTimeline ? "Hide Details" : "Show Details"}
                </button>
              </div>

              {showTimeline && (
                <div style={{
                  maxHeight: "130px",
                  overflowY: "auto",
                  marginTop: "8px",
                  fontSize: "0.72rem",
                  background: "#0f172a",
                  padding: "6px",
                  borderRadius: "6px",
                  border: "1px solid #334155",
                }}>
                  {phonemeTimestamps.map((t, idx) => (
                    <div
                      key={idx}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "2px 4px",
                        borderBottom: "1px solid #1e293b",
                        color: activeViseme === t.viseme ? "#38bdf8" : "#94a3b8",
                      }}
                    >
                      <span><strong>{t.phoneme}</strong> → {t.viseme}</span>
                      <span>{t.startMs}ms - {t.endMs}ms</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Phase 3: speech quality audit */}
          {qualityReport && (
            <div className="info-card" style={{
              background: "linear-gradient(135deg, #1e293b, #0f172a)",
              border: `1px solid ${qualityReport.passesMosTarget ? "#22c55e" : "#f59e0b"}`,
            }}>
              <label style={{ color: qualityReport.passesMosTarget ? "#4ade80" : "#fbbf24" }}>
                📊 Speech Quality
              </label>
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "6px",
                fontSize: "0.78rem",
                color: "#cbd5e1",
                marginTop: "6px",
              }}>
                <div>
                  MOS <strong style={{ color: qualityReport.passesMosTarget ? "#4ade80" : "#fbbf24" }}>
                    {qualityReport.mos ?? "—"}
                  </strong>
                  <span style={{ color: "#64748b" }}> / target &gt; {qualityReport.mosTarget}</span>
                </div>
                <div>PESQ <strong>{qualityReport.pesq ?? "—"}</strong></div>
                <div>STOI <strong>{qualityReport.stoi ?? "—"}</strong></div>
                <div>SI-SDR <strong>{qualityReport.siSdr ?? "—"}</strong></div>
              </div>
              <div style={{ fontSize: "0.7rem", color: "#64748b", marginTop: "6px" }}>
                method: {qualityReport.method}
              </div>
              {(qualityReport.warnings ?? []).map((warning, index) => (
                <div key={index} style={{ fontSize: "0.7rem", color: "#fbbf24", marginTop: "3px" }}>
                  ⚠️ {warning}
                </div>
              ))}
            </div>
          )}

          {/* Phase 3: what the emotion transform actually did */}
          {emotionReport?.applied && (
            <div className="info-card">
              <label>🎭 Emotion Applied</label>
              <div style={{ fontSize: "0.78rem", color: "#cbd5e1", marginTop: "4px" }}>
                <strong style={{ color: "#a855f7" }}>{emotionReport.dominant}</strong>
                {" "}at {Math.round((emotionReport.intensity ?? 0) * 100)}%
              </div>
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 1fr",
                gap: "4px",
                fontSize: "0.7rem",
                color: "#94a3b8",
                marginTop: "6px",
              }}>
                <div>pitch {emotionReport.prosody?.pitch_semitones > 0 ? "+" : ""}
                  {emotionReport.prosody?.pitch_semitones}st</div>
                <div>rate {emotionReport.prosody?.rate}x</div>
                <div>energy {emotionReport.prosody?.energy}x</div>
              </div>
            </div>
          )}

          {latencyMs != null && (
            <div className="info-card">
              <label>Synthesis Latency</label>
              <strong>{Math.round(latencyMs)} ms</strong>
            </div>
          )}

          <div className="info-card">
            <label>Output Audio</label>
            <strong>
              {taskStatus === "SUCCESS"
                ? "Audio & Lip-Sync Ready"
                : taskStatus === "FAILED"
                ? "Generation failed"
                : taskStatus === "QUEUED" || taskStatus === "PROCESSING"
                ? "Generating speech & alignment…"
                : "Waiting for generation"}
            </strong>
            {audioUrl ? (
              <div style={{ marginTop: "0.75rem" }}>
                <audio
                  controls
                  src={audioUrl}
                  style={{ width: "100%" }}
                  onTimeUpdate={handleAudioTimeUpdate}
                  onEnded={handleAudioEnded}
                  onPause={handleAudioEnded}
                >
                  Your browser does not support audio playback.
                </audio>
                <div style={{ marginTop: "0.35rem", fontSize: "0.8rem", color: "#9ca3af" }}>
                  outputs/speech.wav
                </div>
              </div>
            ) : null}
          </div>

          <AvatarPanel
            apiBase={API_BASE}
            phonemeTimestamps={phonemeTimestamps}
            emotionVector={buildRenderEmotionVector(emotionReport)}
            audioReady={taskStatus === "SUCCESS"}
            alignmentMethod={alignmentMethod}
          />
        </section>
      </main>
    </div>
  );
}

export default App;

