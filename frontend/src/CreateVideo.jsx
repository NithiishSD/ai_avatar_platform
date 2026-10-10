import { useCallback, useEffect, useRef, useState } from "react"; // React hooks used below; each is explained where it first appears
import FacePicker from "./FacePicker"; // step 1, the face grid and "Manage faces"
import { recordedGender, rememberVoice, voiceForFace } from "./voices"; // matching a speaker voice to the face

/*
 * "Create video": the studio's main screen, one flow from words to a finished, watermarked video.
 *
 *   1. Face           (FacePicker)
 *   2. Script & voice  standard voice, a cloned voice, or the user's own recording; language; emotion
 *   3. Video           size, lip-sync quality, optional background colour
 *   -> "Create video"  runs both steps the old screen made the user run by hand:
 *        speech:  POST /api/v1/audio/synthesize (always with phoneme timing, which the mouth needs), polled
 *        video:   POST /api/v1/avatar/render-job with that speech, polled
 *      or, for the user's own recording, POST /api/v1/avatar/voice-to-avatar (it does both server-side).
 *
 * Settings that were only for development (engine routing, quality tiers that cannot run here, the
 * speech-quality audit, raw task ids) are gone from the screen; the API still offers all of them.
 *
 * Other routes read here: GET /health (which clone engines have weights), GET /api/v1/audio/voices,
 * GET /api/v1/audio/languages, GET /api/v1/audio/samples, and POST .../render-job/{id}/lipsync-score.
 */

// [value sent to the server, label on the chip]. "" means no emotion field is sent at all.
const EMOTIONS = [
  ["", "Neutral"], ["joy", "Joy"], ["calm", "Calm"], ["excitement", "Excited"],
  ["authority", "Confident"], ["sorrow", "Sad"], ["anger", "Angry"],
];

/* Face weights for the render job from the speech's emotion report (mirror of emotion_engine.to_render_emotion_vector). */
const EMOTION_FACE = {
  joy: { happy: 1.0, blink: 1.6 }, anger: { happy: 0.0, blink: 2.4 }, sorrow: { happy: 0.0, blink: 0.6 },
  authority: { happy: 0.1, blink: 0.8 }, calm: { happy: 0.35, blink: 0.9 }, excitement: { happy: 0.9, blink: 2.0 },
};

// Turn the speech step's emotion report into the render job's emotionVector: a weighted mix of
// the per-emotion face hints above, so a mostly joyful line smiles and blinks a little more.
function faceEmotion(report) {
  const vector = report?.vector;
  if (!vector) return { happy: 0.0, neutral: 1.0, eyeblinkRate: 1.0 }; // no report: a neutral face
  let happy = 0;
  let blink = 0;
  let total = 0;
  const named = {};
  for (const [name, weight] of Object.entries(vector)) { // Object.entries gives [key, value] pairs
    const hint = EMOTION_FACE[name];
    if (!hint || !(weight > 0)) continue; // skip unknown emotions and zero, negative or non-number weights
    happy += hint.happy * weight;
    blink += hint.blink * weight;
    total += weight; // how much of the vector the named emotions cover
    named[name] = Number(weight.toFixed(4)); // also pass each emotion through, rounded to 4 places
  }
  blink += Math.max(0, 1 - total); // the rest of the vector blinks at the neutral rate
  happy = Math.min(1, Math.max(0, happy)); // clamp to 0..1
  return { happy: Number(happy.toFixed(4)), neutral: Number((1 - happy).toFixed(4)), eyeblinkRate: Number(Math.min(10, blink).toFixed(4)), ...named }; // blink capped at 10; `...named` spreads the per-emotion weights in
}

// The server's error reason as text (FastAPI's `detail` is a string or an object), else a fixed message.
function detailOf(payload, fallback) {
  const detail = payload?.detail;
  if (!detail) return fallback;
  return typeof detail === "string" ? detail : detail.message || JSON.stringify(detail);
}

// A Promise that resolves after `ms` milliseconds, so a loop can `await sleep(1000)` between polls.
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The component. Every useState below is one piece of screen state; changing any of them re-renders.
export default function CreateVideo({ apiBase }) {
  const [avatarId, setAvatarId] = useState(""); // the chosen face; FacePicker sets it
  const [engines, setEngines] = useState({ blendshape: true }); // installed lip-sync engines, reported by FacePicker
  const [faces, setFaces] = useState([]); // all faces, so the chosen one can be looked up for voice matching
  // Speaker voices (GET /api/v1/audio/voices) and the speech model to use.
  const [voices, setVoices] = useState([]);
  // The user's explicit speaker choice per face; otherwise the speaker is derived from the face below.
  const [chosenSpeaker, setChosenSpeaker] = useState({});
  const [speechModel, setSpeechModel] = useState("auto"); // auto | kokoro | mms | bark

  // Script & voice
  const [text, setText] = useState("Hello! I am your AI avatar. Type anything here and I will say it.");
  const [voice, setVoice] = useState("standard"); // standard | clone | own
  const [language, setLanguage] = useState("en"); // ISO code sent to the server; "en" is English
  const [languageQuery, setLanguageQuery] = useState(""); // what is typed in the language search box
  const [languages, setLanguages] = useState([]); // search results from the server
  const [emotion, setEmotion] = useState(""); // one of EMOTIONS; "" = neutral
  const [speed, setSpeed] = useState(1.0); // 1.0 = unchanged; only sent when changed
  const [pitch, setPitch] = useState(1.0); // same
  const [samples, setSamples] = useState([]); // consented recordings that can be cloned
  const [sample, setSample] = useState(""); // the chosen recording's server path
  const [cloneEngines, setCloneEngines] = useState([]); // clone engines whose weights are present
  const [cloneEngine, setCloneEngine] = useState(""); // the chosen one
  const [own, setOwn] = useState({ file: null, consentBasis: "", transcript: "" }); // the "Use my recording" form

  // Video
  const [quality, setQuality] = useState("PREVIEW"); // renderQuality in the render job
  const [engine, setEngine] = useState("blendshape"); // lip-sync engine; wav2lip only when installed
  const [motion, setMotion] = useState(1.0); // head-and-shoulder movement: 0 still .. 1.5 lively
  const [background, setBackground] = useState(false); // replace the background?
  const [backgroundColor, setBackgroundColor] = useState("#0b3d91"); // and with which colour

  // Run
  const [phase, setPhase] = useState("idle"); // idle | speech | video | done | failed
  const [speech, setSpeech] = useState(null); // what the voice step produced (audio url, model, transcript...)
  const [job, setJob] = useState(null); // the render job as last polled (status, progress, result)
  const [error, setError] = useState(""); // shown in the red alert at the bottom
  const [score, setScore] = useState(null); // lip-sync score, measured on request
  // Set synchronously on the first click. The button's disabled state only takes effect after React
  // re-renders, so a fast double click used to start two videos (owner's report); this ref cannot lag.
  // useRef gives a box ({ current }) that keeps its value across renders. Writing to it does
  // not re-render, and a read always sees the latest write, which is what a lock needs.
  const runningRef = useRef(false);

  /* Clone engines whose weights are on disk, from /health; never offer one that would be refused. */
  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/health`).then((r) => r.json()).then((health) => {
      if (ignore) return;
      const present = Object.fromEntries((health.modelWeights?.models ?? []).map((m) => [m.key, m.present])); // { modelKey: true/false } from the health report
      const usable = (health.capabilities?.cloneEngines ?? []).filter((key) => present[key]); // offered by the server AND on disk
      setCloneEngines(usable);
      setCloneEngine((current) => current || usable[0] || ""); // keep a choice already made
    }).catch(() => {}); // offline: no clone engines are offered
    return () => { ignore = true; };
  }, [apiBase]); // only re-run if the backend address changes

  useEffect(() => {
    let ignore = false;
    // Speaker voices for "Standard voice". Same pattern: fetch once, drop a reply that lands after unmount.
    fetch(`${apiBase}/api/v1/audio/voices`).then((r) => r.json()).then((data) => !ignore && setVoices(data.voices ?? [])).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  /* The speaker for the chosen face: picked here in this session, else remembered, else the face's recorded gender, else the default. */
  const face = faces.find((f) => f.avatarId === avatarId); // undefined until faces load
  const speaker = chosenSpeaker[avatarId] || voiceForFace(face, voices, "af_heart"); // computed on every render, not stored: it can never go stale

  // Remember the choice in state for this session and in localStorage for next time.
  const chooseSpeaker = (id) => {
    setChosenSpeaker((current) => ({ ...current, [avatarId]: id })); // copy the map with one key changed
    if (avatarId) rememberVoice(avatarId, id);
  };

  /* Language search (1,077 languages): one request per pause in typing; a stale reply is dropped. */
  useEffect(() => {
    let ignore = false;
    // Debounce: wait 250 ms after the last keystroke. Each keystroke re-runs this effect, and the
    // cleanup cancels the previous timer, so only the final pause sends a request.
    const timer = setTimeout(() => {
      fetch(`${apiBase}/api/v1/audio/languages?q=${encodeURIComponent(languageQuery)}&limit=40`) // encodeURIComponent makes the text safe inside a URL
        .then((r) => r.json()).then((data) => !ignore && setLanguages(data.languages ?? [])).catch(() => {});
    }, 250);
    return () => { ignore = true; clearTimeout(timer); }; // cleanup: drop a stale reply and cancel a pending request
  }, [apiBase, languageQuery]); // re-run whenever the search text changes

  /* Voice references for cloning, loaded the first time "Clone a voice" is chosen. */
  const chooseVoice = async (choice) => {
    setVoice(choice); // switch the tab at once; the samples load behind it
    if (choice === "clone" && samples.length === 0) { // load only once
      try {
        const data = await (await fetch(`${apiBase}/api/v1/audio/samples`)).json(); // inner await: the reply; outer await: its JSON body
        setSamples(data.samples ?? []);
        setSample((current) => current || data.samples?.[0]?.path || ""); // preselect the first recording
      } catch {
        setError("Could not load the voice recordings.");
      }
    }
  };

  /* Poll a URL until its status is one of `done`, reporting each state through `onState`. */
  // The polling loop: ask for the status, report it, stop on a final state, else wait 1 s and ask
  // again. useCallback with [] keeps one copy of this function for the component's life.
  const pollUntil = useCallback(async (url, done, onState) => {
    for (;;) { // loop forever; the `return` and `throw` inside are the only exits
      const res = await fetch(url);
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Status check failed"));
      onState?.(body); // e.g. setJob, so the progress bar moves while we wait
      if (done.includes(body.status)) return body; // a final state: hand the last status to the caller
      await sleep(1000); // one request per second
    }
  }, []);

  // Step two: send the finished speech to the renderer as an AvatarRenderJob (backend/contracts.py,
  // the only audio-to-vision interface) and poll until the video is done or failed.
  const renderVideo = async (speechResult) => {
    setPhase("video");
    const payload = {
      jobId: `JOB-${Date.now()}`, avatarId, audioUrl: speechResult.audioUrl, sampleRate: 24000, // Date.now() (milliseconds) gives a new id per run
      durationSeconds: Math.max(speechResult.durationSeconds, 0.5), phonemeTimestamps: speechResult.phonemeTimestamps, // at least 0.5 s; the contract requires a positive duration. The phoneme times drive the mouth
      emotionVector: faceEmotion(speechResult.emotion), renderQuality: quality, targetFps: 25, motionIntensity: motion, // motionIntensity: 0 still .. 1.5 lively (contract allows up to 2)
      ...(background ? { background: { color: backgroundColor } } : {}), // conditional spread: adds the key only when wanted
    };
    const res = await fetch(`${apiBase}/api/v1/avatar/render-job?engine=${encodeURIComponent(engine)}`, { // the engine is a query parameter, not part of the contract
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), // a JSON body needs this header
    });
    const accepted = await res.json(); // the accepted job, before it has run
    if (!res.ok) throw new Error(detailOf(accepted, "The video was refused")); // e.g. consent or validation failed
    setJob(accepted); // show the job at once, before the first poll
    return pollUntil(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(payload.jobId)}`, ["COMPLETED", "FAILED"], setJob); // resolves with the final job
  };

  // Step one: text to speech. The server writes the audio to outputs/<filename>, which it serves
  // at /outputs, so the URL is known before the task finishes.
  const makeSpeech = async () => {
    setPhase("speech");
    const filename = `studio-${Date.now()}.wav`;
    const body = {
      text, language, returnAlignment: true, outputFilename: filename, // returnAlignment asks for phoneme timing
      // "auto" lets the server pick by language (Kokoro for English, MMS otherwise); the others force an engine.
      mode: voice === "clone" ? "clone" : { auto: "fast", kokoro: "fast", mms: "multilingual", bark: "dialogue" }[speechModel],
      ...(voice === "standard" && speechModel !== "mms" && speechModel !== "bark" ? { voice: speaker } : {}), // MMS and Bark have no speaker choice here
      ...(speed !== 1 ? { speed } : {}), ...(pitch !== 1 ? { pitch } : {}), // `{ speed }` is shorthand for { speed: speed }
      ...(emotion ? { emotion } : {}),
      ...(voice === "clone" ? { speakerWav: sample, ...(cloneEngine ? { cloneEngine } : {}) } : {}), // the recording to imitate, and the engine if one is known
    };
    const res = await fetch(`${apiBase}/api/v1/audio/synthesize`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let state = await res.json(); // `let`: replaced below by the polled final state
    if (!res.ok) throw new Error(detailOf(state, "The voice step was refused"));
    if (!["SUCCESS", "FAILED"].includes(state.status)) { // not final yet: poll until it is
      state = await pollUntil(`${apiBase}/api/v1/audio/synthesize/${state.taskId}`, ["SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"]);
    }
    if (state.status !== "SUCCESS") throw new Error(state.error || `The voice step ended ${state.status}`); // FAILED, CANCELLED or UNKNOWN
    if (!state.phonemeTimestamps?.length) throw new Error("The speech came back without timing, so the mouth cannot follow it."); // no silent fallback: without timing, stop and say so
    const result = { ...state, audioUrl: `${apiBase}/outputs/${filename}` }; // the file the server wrote, as a full URL
    setSpeech(result); // shows the audio player and model name
    return result;
  };

  /* The user's own recording: one request does transcription, alignment and the render job. */
  const fromRecording = async () => {
    setPhase("speech");
    // FormData: a multipart body that carries the audio file plus text fields. The browser sets
    // the Content-Type (with its boundary) itself, so no header is given.
    const form = new FormData();
    form.append("file", own.file); // the recording itself
    form.append("avatarId", avatarId);
    form.append("consentBasis", own.consentBasis); // required: whose voice it is
    form.append("engine", engine);
    form.append("renderQuality", quality);
    form.append("motionIntensity", String(motion)); // form fields are strings
    if (own.transcript.trim()) form.append("transcript", own.transcript.trim()); // empty: the server recognises the words
    const res = await fetch(`${apiBase}/api/v1/avatar/voice-to-avatar`, { method: "POST", body: form });
    const accepted = await res.json();
    if (!res.ok) throw new Error(detailOf(accepted, "The recording was refused"));
    setSpeech({ ...accepted.speech, audioUrl: `${apiBase}${accepted.speech.audioUrl}`, own: true }); // `own` switches the info line to show the words
    setJob(accepted);
    setPhase("video"); // the server already started the render
    return pollUntil(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(accepted.jobId)}`, ["COMPLETED", "FAILED"], setJob);
  };

  // The "Create video" click. The ref check-and-set happens before any await, so a second click
  // in the same instant sees `true` and returns.
  const create = async () => {
    if (runningRef.current) return; // already running: ignore this click
    runningRef.current = true; // take the lock
    setError("");
    setSpeech(null); // clear the previous run's results
    setJob(null);
    setScore(null);
    try {
      const finished = voice === "own" ? await fromRecording() : await renderVideo(await makeSpeech()); // own recording: one request; else speech, then video
      setPhase(finished.status === "COMPLETED" ? "done" : "failed");
      if (finished.status !== "COMPLETED") setError(finished.error || "The video could not be made."); // the job ended FAILED
    } catch (err) {
      setPhase("failed");
      setError(err.message); // any thrown step lands here
    } finally {
      runningRef.current = false; // finally: release the lock on success or failure
    }
  };

  // Ask the server to score how well the lips match the sound (offset in frames, confidence).
  const measure = async () => {
    const res = await fetch(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(job.jobId)}/lipsync-score`, { method: "POST" });
    const body = await res.json();
    if (res.ok) setScore(body.score);
    else setError(detailOf(body, "Lip-sync scoring failed"));
  };

  const working = phase === "speech" || phase === "video"; // derived values: computed each render, not stored
  const ready = Boolean(avatarId) && (voice === "own" ? own.file && own.consentBasis : text.trim()) && (voice !== "clone" || sample); // enough input to start: a face, words or a consented recording, and a sample when cloning
  const result = job?.result; // size, fps, watermark, manifest of the finished video

  return (
    // A plain container, not a <form>: FacePicker has forms of its own, and forms cannot nest.
    <div className="create">
      <div>
        {/* Step 1. The parent owns avatarId and hands FacePicker the setter ("lifting state up"). */}
        <FacePicker apiBase={apiBase} avatarId={avatarId} setAvatarId={setAvatarId} onEngines={setEngines} onFaces={setFaces} />

        <div className="card">
          <h2><span className="step">2</span> Script &amp; voice</h2>
          {/* Three voice sources. aria-pressed shows which is active. */}
          <div className="segmented" role="group" aria-label="voice">
            {[["standard", "Standard voice"], ["clone", "Clone a voice"], ["own", "Use my recording"]].map(([value, label]) => (
              <button key={value} type="button" id={`voice-${value}`} aria-pressed={voice === value} onClick={() => chooseVoice(value)}>{label}</button>
            ))}
          </div>

          {/* Conditional rendering: each block below shows only for the matching voice source. */}
          {voice !== "own" && (
            <label className="field"><span>What should the avatar say?</span>
              {/* Controlled textarea: React state is the single source of the text. */}
              <textarea id="script" value={text} onChange={(e) => setText(e.target.value)} />
            </label>
          )}

          {voice === "standard" && (
            <div className="row">
              <label className="field"><span>Voice model</span>
                {/* Which speech engine; see the `mode` mapping in makeSpeech. */}
                <select id="speech-model-select" value={speechModel} onChange={(e) => setSpeechModel(e.target.value)}>
                  <option value="auto">Auto (best for the language)</option>
                  <option value="kokoro">Kokoro (English, most natural)</option>
                  <option value="mms">MMS-TTS (1,100+ languages)</option>
                  <option value="bark">Bark (two speakers: [S1] / [S2])</option>
                </select>
              </label>
              {speechModel !== "mms" && speechModel !== "bark" && (
                <label className="field"><span>Speaker{face && recordedGender(face) ? ` (this face: ${recordedGender(face)})` : ""}</span>
                  {/* Voices whose model is not installed are listed but disabled. */}
                  <select id="speaker-select" value={speaker} onChange={(e) => chooseSpeaker(e.target.value)}>
                    {voices.map((v) => <option key={v.id} value={v.id} disabled={!v.present}>{v.name} ({v.gender}){v.present ? "" : " - not installed"}</option>)}
                  </select>
                  {face && !recordedGender(face) && !chosenSpeaker[avatarId] && (
                    <span className="hint" id="speaker-hint">This face has no recorded gender: pick the speaker that fits (remembered for this face).</span>
                  )}
                </label>
              )}
            </div>
          )}

          {voice === "clone" && (
            <div className="row">
              <label className="field"><span>Voice to clone (a consented recording)</span>
                {/* Only consented recordings are listed; the value is the server path sent as speakerWav. */}
                <select id="sample-select" value={sample} onChange={(e) => setSample(e.target.value)}>
                  {samples.map((s) => <option key={s.path} value={s.path}>{s.filename}</option>)}
                </select>
              </label>
              <label className="field"><span>Cloning engine</span>
                {/* Only engines whose weights are on disk (from /health). */}
                <select id="clone-engine-select" value={cloneEngine} onChange={(e) => setCloneEngine(e.target.value)}>
                  {cloneEngines.map((key) => <option key={key} value={key}>{key === "xtts-v2" ? "XTTS-v2 (most similar)" : "OpenVoice V2 (faster)"}</option>)}
                </select>
              </label>
            </div>
          )}

          {voice === "own" && (
            <>
              <label className="field"><span>Your recording</span>
                {/* An uncontrolled file input: the browser keeps the file, onChange copies it into state. */}
                <input id="own-audio-file" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" onChange={(e) => setOwn({ ...own, file: e.target.files?.[0] || null })} />
              </label>
              <label className="field"><span>Whose voice is it?</span>
                {/* No consent, no use: the Create button stays disabled until this is chosen. */}
                <select id="own-audio-consent" value={own.consentBasis} onChange={(e) => setOwn({ ...own, consentBasis: e.target.value })}>
                  <option value="">choose…</option>
                  <option value="speaker-recorded">my own voice</option>
                  <option value="written-consent">someone who consented in writing</option>
                  <option value="open-licence">an openly licensed recording</option>
                </select>
              </label>
              <label className="field"><span>Transcript (optional: left empty, the words are recognised automatically)</span>
                <textarea id="own-audio-transcript" style={{ minHeight: 56 }} value={own.transcript} onChange={(e) => setOwn({ ...own, transcript: e.target.value })} />
              </label>
            </>
          )}

          {voice !== "own" && (
            <>
              <div className="row">
                <label className="field"><span>Language</span>
                  {/* Typing here re-runs the debounced search effect above. */}
                  <input id="language-search" type="search" placeholder="Search languages…" value={languageQuery} onChange={(e) => setLanguageQuery(e.target.value)} />
                </label>
                <label className="field"><span>&nbsp;</span>
                  <select id="language-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                    {/* English is always first and uses "en", which the server routes to the fast Kokoro voice. */}
                    <option value="en">English</option>
                    {/* Keep the chosen language listed even when the current search no longer returns it. */}
                    {language !== "en" && !languages.some((l) => l.iso3 === language) && <option value={language}>{language}</option>}
                    {languages.filter((l) => !l.isEnglish).map((l) => <option key={l.iso3} value={l.iso3}>{l.name} ({l.iso3})</option>)}
                  </select>
                </label>
              </div>
              <span className="hint">Emotion</span>
              {/* The key falls back to "neutral" because the neutral value is "" and a key should not be empty. */}
              <div className="chips" id="emotion-chips">
                {EMOTIONS.map(([value, label]) => (
                  <button key={value || "neutral"} type="button" data-emotion={value || "neutral"} aria-pressed={emotion === value} onClick={() => setEmotion(value)}>{label}</button>
                ))}
              </div>
              {/* <details> is a native collapsible section; no state is needed to open or close it. */}
              <details className="more">
                <summary>More voice options</summary>
                <div className="row">
                  <label className="field"><span>Speed {speed.toFixed(1)}×</span>
                    {/* Range inputs give strings; Number() converts before storing. */}
                    <input type="range" min="0.5" max="2" step="0.1" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
                  </label>
                  <label className="field"><span>Pitch {pitch.toFixed(1)}×</span>
                    <input type="range" min="0.5" max="2" step="0.1" value={pitch} onChange={(e) => setPitch(Number(e.target.value))} />
                  </label>
                </div>
              </details>
            </>
          )}
        </div>

        <div className="card">
          <h2><span className="step">3</span> Video</h2>
          <div className="row">
            <label className="field"><span>Video quality</span>
              {/* renderQuality in the render job. */}
              <select id="quality-select" value={quality} onChange={(e) => setQuality(e.target.value)}>
                <option value="PREVIEW">Preview (512 px, fastest)</option>
                <option value="1080P_HQ">1080p</option>
              </select>
            </label>
            <label className="field"><span>Lip-sync model</span>
              {/* Wav2Lip and SadTalker are offered only when FacePicker reported them installed. */}
              <select id="engine-select" value={engine} onChange={(e) => setEngine(e.target.value)}>
                <option value="blendshape">Standard (fast)</option>
                <option value="wav2lip" disabled={!engines.wav2lip}>High quality (Wav2Lip){engines.wav2lip ? "" : " - not installed"}</option>
                {/* SadTalker moves the whole head, jaw and expression; head movement there is only on or off. */}
                <option value="sadtalker" disabled={!engines.sadtalker}>Whole head moves (SadTalker){engines.sadtalker ? "" : " - not installed"}</option>
              </select>
            </label>
          </div>
          <label className="field"><span>Head movement</span>
            {/* Option values become strings in the DOM, so onChange converts back with Number(). */}
            <select id="motion-select" value={motion} onChange={(e) => setMotion(Number(e.target.value))}>
              <option value={0}>Still (lips and blinks only)</option>
              <option value={0.6}>Subtle</option>
              <option value={1}>Natural</option>
              <option value={1.5}>Lively</option>
            </select>
          </label>
          <label className="check">
            <input id="background-toggle" type="checkbox" checked={background} onChange={(e) => setBackground(e.target.checked)} /> New background colour
            {background && <input id="background-color" type="color" aria-label="background colour" value={backgroundColor} onChange={(e) => setBackgroundColor(e.target.value)} />}
          </label>
        </div>
      </div>

      <div>
        <div className="card result">
          {/* disabled is the visible guard; runningRef in create() is the one that cannot lag. */}
          <button type="button" id="create-btn" className="primary big" disabled={!ready || working} onClick={create}>
            {phase === "speech" ? "Making the voice…" : phase === "video" ? "Animating the face…" : "Create video"}
          </button>
          {!avatarId && <p className="hint">Choose a face first.</p>}

          {/* `!= null` is true for any number, including 0. progress is 0..1, the bar width a percentage. */}
          {working && job?.progress != null && <div className="progress"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>}

          {/* What the voice step produced: the words (own recording) or the model and length, plus a player. */}
          {speech && (
            <div className="meta" id="speech-info">
              {speech.own
                ? <>Words {speech.transcriptSource === "asr" ? "recognised" : "given"} ({speech.language}): “{speech.transcript}”</>
                : <>Voice: {speech.modelUsed} · {speech.durationSeconds?.toFixed?.(1)} s</>}
              {speech.alignmentMethod === "acoustic-fallback" && <div className="warn">Timing was estimated, not measured: lip sync will be loose.</div>}
              {speech.audioUrl && <audio id="speech-audio" controls src={speech.audioUrl} />}
            </div>
          )}

          {/* The render job status as last polled. */}
          {job && (
            <div className="meta">Video: <strong id="render-status" className={job.status === "FAILED" ? "warn" : ""}>{job.status}</strong>{job.engine ? ` · ${job.engine}` : ""}</div>
          )}

          {/* The finished video and its provenance. */}
          {job?.status === "COMPLETED" && job.videoUrl && (
            <>
              {/* crossOrigin asks for the video with CORS. The `?t=` query string differs per job, so the browser
                 does not reuse a cached file from an earlier job. */}
              <video id="avatar-video" controls crossOrigin="anonymous" src={`${apiBase}${job.videoUrl}?t=${job.jobId}`} />
              {result && (
                <div className="meta">
                  {result.width}×{result.height} · {result.fps} fps · made in {result.renderSeconds}s
                  {result.background ? ` · background: ${result.background}` : ""}
                </div>
              )}
              {/* Provenance line: whether the invisible watermark was applied and read back, the signed manifest, a download link. */}
              {result?.watermark && (
                <div id="provenance-line" className={`meta ${result.watermark.applied ? "ok" : "warn"}`}>
                  {result.watermark.applied ? `🔒 invisible watermark verified (${result.watermark.tagBitsMatching}/128 bits)` : `⚠ not watermarked: ${result.watermark.reason}`}
                  {result.manifest?.url && <> · <a id="manifest-link" href={`${apiBase}${result.manifest.url}`} target="_blank" rel="noreferrer">signed manifest</a></>}
                  {" · "}<a id="download-link" href={`${apiBase}${job.videoUrl}`} download>download</a>
                </div>
              )}
              {result?.warnings?.map((w) => <div key={w} className="meta warn">⚠ {w}</div>)}
              {/* Scoring runs only when asked. */}
              <button type="button" className="link" id="lipsync-score-btn" onClick={measure}>Measure lip sync</button>
              {score && (
                <div id="lipsync-score" className="meta">
                  offset {score.offsetFrames} frames · confidence {score.lseC} · {score.secondsWithinOneFrame?.percent ?? "–"}% of seconds in sync
                </div>
              )}
            </>
          )}
          {/* Any error from any step ends up here. */}
          {error && <div className="alert error" id="create-error">{error}</div>}
        </div>
      </div>
    </div>
  );
}
