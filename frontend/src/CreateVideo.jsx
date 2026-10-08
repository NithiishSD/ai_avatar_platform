import { useCallback, useEffect, useRef, useState } from "react";
import FacePicker from "./FacePicker";
import { recordedGender, rememberVoice, voiceForFace } from "./voices";

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
 */

const EMOTIONS = [
  ["", "Neutral"], ["joy", "Joy"], ["calm", "Calm"], ["excitement", "Excited"],
  ["authority", "Confident"], ["sorrow", "Sad"], ["anger", "Angry"],
];

/* Face weights for the render job from the speech's emotion report (mirror of emotion_engine.to_render_emotion_vector). */
const EMOTION_FACE = {
  joy: { happy: 1.0, blink: 1.6 }, anger: { happy: 0.0, blink: 2.4 }, sorrow: { happy: 0.0, blink: 0.6 },
  authority: { happy: 0.1, blink: 0.8 }, calm: { happy: 0.35, blink: 0.9 }, excitement: { happy: 0.9, blink: 2.0 },
};

function faceEmotion(report) {
  const vector = report?.vector;
  if (!vector) return { happy: 0.0, neutral: 1.0, eyeblinkRate: 1.0 };
  let happy = 0;
  let blink = 0;
  let total = 0;
  const named = {};
  for (const [name, weight] of Object.entries(vector)) {
    const hint = EMOTION_FACE[name];
    if (!hint || !(weight > 0)) continue;
    happy += hint.happy * weight;
    blink += hint.blink * weight;
    total += weight;
    named[name] = Number(weight.toFixed(4));
  }
  blink += Math.max(0, 1 - total); // the rest of the vector blinks at the neutral rate
  happy = Math.min(1, Math.max(0, happy));
  return { happy: Number(happy.toFixed(4)), neutral: Number((1 - happy).toFixed(4)), eyeblinkRate: Number(Math.min(10, blink).toFixed(4)), ...named };
}

function detailOf(payload, fallback) {
  const detail = payload?.detail;
  if (!detail) return fallback;
  return typeof detail === "string" ? detail : detail.message || JSON.stringify(detail);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export default function CreateVideo({ apiBase }) {
  const [avatarId, setAvatarId] = useState("");
  const [engines, setEngines] = useState({ blendshape: true });
  const [faces, setFaces] = useState([]);
  // Speaker voices (GET /api/v1/audio/voices) and the speech model to use.
  const [voices, setVoices] = useState([]);
  // The user's explicit speaker choice per face; otherwise the speaker is derived from the face below.
  const [chosenSpeaker, setChosenSpeaker] = useState({});
  const [speechModel, setSpeechModel] = useState("auto"); // auto | kokoro | mms | bark

  // Script & voice
  const [text, setText] = useState("Hello! I am your AI avatar. Type anything here and I will say it.");
  const [voice, setVoice] = useState("standard"); // standard | clone | own
  const [language, setLanguage] = useState("en");
  const [languageQuery, setLanguageQuery] = useState("");
  const [languages, setLanguages] = useState([]);
  const [emotion, setEmotion] = useState("");
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [samples, setSamples] = useState([]);
  const [sample, setSample] = useState("");
  const [cloneEngines, setCloneEngines] = useState([]);
  const [cloneEngine, setCloneEngine] = useState("");
  const [own, setOwn] = useState({ file: null, consentBasis: "", transcript: "" });

  // Video
  const [quality, setQuality] = useState("PREVIEW");
  const [engine, setEngine] = useState("blendshape");
  const [background, setBackground] = useState(false);
  const [backgroundColor, setBackgroundColor] = useState("#0b3d91");

  // Run
  const [phase, setPhase] = useState("idle"); // idle | speech | video | done | failed
  const [speech, setSpeech] = useState(null); // what the voice step produced (audio url, model, transcript...)
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [score, setScore] = useState(null);
  // Set synchronously on the first click. The button's disabled state only takes effect after React
  // re-renders, so a fast double click used to start two videos (owner's report); this ref cannot lag.
  const runningRef = useRef(false);

  /* Clone engines whose weights are on disk, from /health; never offer one that would be refused. */
  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/health`).then((r) => r.json()).then((health) => {
      if (ignore) return;
      const present = Object.fromEntries((health.modelWeights?.models ?? []).map((m) => [m.key, m.present]));
      const usable = (health.capabilities?.cloneEngines ?? []).filter((key) => present[key]);
      setCloneEngines(usable);
      setCloneEngine((current) => current || usable[0] || "");
    }).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/api/v1/audio/voices`).then((r) => r.json()).then((data) => !ignore && setVoices(data.voices ?? [])).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  /* The speaker for the chosen face: picked here in this session, else remembered, else the face's recorded gender, else the default. */
  const face = faces.find((f) => f.avatarId === avatarId);
  const speaker = chosenSpeaker[avatarId] || voiceForFace(face, voices, "af_heart");

  const chooseSpeaker = (id) => {
    setChosenSpeaker((current) => ({ ...current, [avatarId]: id }));
    if (avatarId) rememberVoice(avatarId, id);
  };

  /* Language search (1,077 languages): one request per pause in typing; a stale reply is dropped. */
  useEffect(() => {
    let ignore = false;
    const timer = setTimeout(() => {
      fetch(`${apiBase}/api/v1/audio/languages?q=${encodeURIComponent(languageQuery)}&limit=40`)
        .then((r) => r.json()).then((data) => !ignore && setLanguages(data.languages ?? [])).catch(() => {});
    }, 250);
    return () => { ignore = true; clearTimeout(timer); };
  }, [apiBase, languageQuery]);

  /* Voice references for cloning, loaded the first time "Clone a voice" is chosen. */
  const chooseVoice = async (choice) => {
    setVoice(choice);
    if (choice === "clone" && samples.length === 0) {
      try {
        const data = await (await fetch(`${apiBase}/api/v1/audio/samples`)).json();
        setSamples(data.samples ?? []);
        setSample((current) => current || data.samples?.[0]?.path || "");
      } catch {
        setError("Could not load the voice recordings.");
      }
    }
  };

  /* Poll a URL until its status is one of `done`, reporting each state through `onState`. */
  const pollUntil = useCallback(async (url, done, onState) => {
    for (;;) {
      const res = await fetch(url);
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Status check failed"));
      onState?.(body);
      if (done.includes(body.status)) return body;
      await sleep(1000);
    }
  }, []);

  const renderVideo = async (speechResult) => {
    setPhase("video");
    const payload = {
      jobId: `JOB-${Date.now()}`, avatarId, audioUrl: speechResult.audioUrl, sampleRate: 24000,
      durationSeconds: Math.max(speechResult.durationSeconds, 0.5), phonemeTimestamps: speechResult.phonemeTimestamps,
      emotionVector: faceEmotion(speechResult.emotion), renderQuality: quality, targetFps: 25,
      ...(background ? { background: { color: backgroundColor } } : {}),
    };
    const res = await fetch(`${apiBase}/api/v1/avatar/render-job?engine=${encodeURIComponent(engine)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const accepted = await res.json();
    if (!res.ok) throw new Error(detailOf(accepted, "The video was refused"));
    setJob(accepted);
    return pollUntil(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(payload.jobId)}`, ["COMPLETED", "FAILED"], setJob);
  };

  const makeSpeech = async () => {
    setPhase("speech");
    const filename = `studio-${Date.now()}.wav`;
    const body = {
      text, language, returnAlignment: true, outputFilename: filename,
      // "auto" lets the server pick by language (Kokoro for English, MMS otherwise); the others force an engine.
      mode: voice === "clone" ? "clone" : { auto: "fast", kokoro: "fast", mms: "multilingual", bark: "dialogue" }[speechModel],
      ...(voice === "standard" && speechModel !== "mms" && speechModel !== "bark" ? { voice: speaker } : {}),
      ...(speed !== 1 ? { speed } : {}), ...(pitch !== 1 ? { pitch } : {}),
      ...(emotion ? { emotion } : {}),
      ...(voice === "clone" ? { speakerWav: sample, ...(cloneEngine ? { cloneEngine } : {}) } : {}),
    };
    const res = await fetch(`${apiBase}/api/v1/audio/synthesize`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let state = await res.json();
    if (!res.ok) throw new Error(detailOf(state, "The voice step was refused"));
    if (!["SUCCESS", "FAILED"].includes(state.status)) {
      state = await pollUntil(`${apiBase}/api/v1/audio/synthesize/${state.taskId}`, ["SUCCESS", "FAILED", "CANCELLED", "UNKNOWN"]);
    }
    if (state.status !== "SUCCESS") throw new Error(state.error || `The voice step ended ${state.status}`);
    if (!state.phonemeTimestamps?.length) throw new Error("The speech came back without timing, so the mouth cannot follow it.");
    const result = { ...state, audioUrl: `${apiBase}/outputs/${filename}` };
    setSpeech(result);
    return result;
  };

  /* The user's own recording: one request does transcription, alignment and the render job. */
  const fromRecording = async () => {
    setPhase("speech");
    const form = new FormData();
    form.append("file", own.file);
    form.append("avatarId", avatarId);
    form.append("consentBasis", own.consentBasis);
    form.append("engine", engine);
    form.append("renderQuality", quality);
    if (own.transcript.trim()) form.append("transcript", own.transcript.trim());
    const res = await fetch(`${apiBase}/api/v1/avatar/voice-to-avatar`, { method: "POST", body: form });
    const accepted = await res.json();
    if (!res.ok) throw new Error(detailOf(accepted, "The recording was refused"));
    setSpeech({ ...accepted.speech, audioUrl: `${apiBase}${accepted.speech.audioUrl}`, own: true });
    setJob(accepted);
    setPhase("video");
    return pollUntil(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(accepted.jobId)}`, ["COMPLETED", "FAILED"], setJob);
  };

  const create = async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setError("");
    setSpeech(null);
    setJob(null);
    setScore(null);
    try {
      const finished = voice === "own" ? await fromRecording() : await renderVideo(await makeSpeech());
      setPhase(finished.status === "COMPLETED" ? "done" : "failed");
      if (finished.status !== "COMPLETED") setError(finished.error || "The video could not be made.");
    } catch (err) {
      setPhase("failed");
      setError(err.message);
    } finally {
      runningRef.current = false;
    }
  };

  const measure = async () => {
    const res = await fetch(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(job.jobId)}/lipsync-score`, { method: "POST" });
    const body = await res.json();
    if (res.ok) setScore(body.score);
    else setError(detailOf(body, "Lip-sync scoring failed"));
  };

  const working = phase === "speech" || phase === "video";
  const ready = Boolean(avatarId) && (voice === "own" ? own.file && own.consentBasis : text.trim()) && (voice !== "clone" || sample);
  const result = job?.result;

  return (
    // A plain container, not a <form>: FacePicker has forms of its own, and forms cannot nest.
    <div className="create">
      <div>
        <FacePicker apiBase={apiBase} avatarId={avatarId} setAvatarId={setAvatarId} onEngines={setEngines} onFaces={setFaces} />

        <div className="card">
          <h2><span className="step">2</span> Script &amp; voice</h2>
          <div className="segmented" role="group" aria-label="voice">
            {[["standard", "Standard voice"], ["clone", "Clone a voice"], ["own", "Use my recording"]].map(([value, label]) => (
              <button key={value} type="button" id={`voice-${value}`} aria-pressed={voice === value} onClick={() => chooseVoice(value)}>{label}</button>
            ))}
          </div>

          {voice !== "own" && (
            <label className="field"><span>What should the avatar say?</span>
              <textarea id="script" value={text} onChange={(e) => setText(e.target.value)} />
            </label>
          )}

          {voice === "standard" && (
            <div className="row">
              <label className="field"><span>Voice model</span>
                <select id="speech-model-select" value={speechModel} onChange={(e) => setSpeechModel(e.target.value)}>
                  <option value="auto">Auto (best for the language)</option>
                  <option value="kokoro">Kokoro (English, most natural)</option>
                  <option value="mms">MMS-TTS (1,100+ languages)</option>
                  <option value="bark">Bark (two speakers: [S1] / [S2])</option>
                </select>
              </label>
              {speechModel !== "mms" && speechModel !== "bark" && (
                <label className="field"><span>Speaker{face && recordedGender(face) ? ` (this face: ${recordedGender(face)})` : ""}</span>
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
                <select id="sample-select" value={sample} onChange={(e) => setSample(e.target.value)}>
                  {samples.map((s) => <option key={s.path} value={s.path}>{s.filename}</option>)}
                </select>
              </label>
              <label className="field"><span>Cloning engine</span>
                <select id="clone-engine-select" value={cloneEngine} onChange={(e) => setCloneEngine(e.target.value)}>
                  {cloneEngines.map((key) => <option key={key} value={key}>{key === "xtts-v2" ? "XTTS-v2 (most similar)" : "OpenVoice V2 (faster)"}</option>)}
                </select>
              </label>
            </div>
          )}

          {voice === "own" && (
            <>
              <label className="field"><span>Your recording</span>
                <input id="own-audio-file" type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac" onChange={(e) => setOwn({ ...own, file: e.target.files?.[0] || null })} />
              </label>
              <label className="field"><span>Whose voice is it?</span>
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
                  <input id="language-search" type="search" placeholder="Search languages…" value={languageQuery} onChange={(e) => setLanguageQuery(e.target.value)} />
                </label>
                <label className="field"><span>&nbsp;</span>
                  <select id="language-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                    {/* English is always first and uses "en", which the server routes to the fast Kokoro voice. */}
                    <option value="en">English</option>
                    {language !== "en" && !languages.some((l) => l.iso3 === language) && <option value={language}>{language}</option>}
                    {languages.filter((l) => !l.isEnglish).map((l) => <option key={l.iso3} value={l.iso3}>{l.name} ({l.iso3})</option>)}
                  </select>
                </label>
              </div>
              <span className="hint">Emotion</span>
              <div className="chips" id="emotion-chips">
                {EMOTIONS.map(([value, label]) => (
                  <button key={value || "neutral"} type="button" data-emotion={value || "neutral"} aria-pressed={emotion === value} onClick={() => setEmotion(value)}>{label}</button>
                ))}
              </div>
              <details className="more">
                <summary>More voice options</summary>
                <div className="row">
                  <label className="field"><span>Speed {speed.toFixed(1)}×</span>
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
              <select id="quality-select" value={quality} onChange={(e) => setQuality(e.target.value)}>
                <option value="PREVIEW">Preview (512 px, fastest)</option>
                <option value="1080P_HQ">1080p</option>
              </select>
            </label>
            <label className="field"><span>Lip-sync model</span>
              <select id="engine-select" value={engine} onChange={(e) => setEngine(e.target.value)}>
                <option value="blendshape">Standard (fast)</option>
                <option value="wav2lip" disabled={!engines.wav2lip}>High quality (Wav2Lip){engines.wav2lip ? "" : " - not installed"}</option>
              </select>
            </label>
          </div>
          <label className="check">
            <input id="background-toggle" type="checkbox" checked={background} onChange={(e) => setBackground(e.target.checked)} /> New background colour
            {background && <input id="background-color" type="color" aria-label="background colour" value={backgroundColor} onChange={(e) => setBackgroundColor(e.target.value)} />}
          </label>
        </div>
      </div>

      <div>
        <div className="card result">
          <button type="button" id="create-btn" className="primary big" disabled={!ready || working} onClick={create}>
            {phase === "speech" ? "Making the voice…" : phase === "video" ? "Animating the face…" : "Create video"}
          </button>
          {!avatarId && <p className="hint">Choose a face first.</p>}

          {working && job?.progress != null && <div className="progress"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>}

          {speech && (
            <div className="meta" id="speech-info">
              {speech.own
                ? <>Words {speech.transcriptSource === "asr" ? "recognised" : "given"} ({speech.language}): “{speech.transcript}”</>
                : <>Voice: {speech.modelUsed} · {speech.durationSeconds?.toFixed?.(1)} s</>}
              {speech.alignmentMethod === "acoustic-fallback" && <div className="warn">Timing was estimated, not measured: lip sync will be loose.</div>}
              {speech.audioUrl && <audio id="speech-audio" controls src={speech.audioUrl} />}
            </div>
          )}

          {job && (
            <div className="meta">Video: <strong id="render-status" className={job.status === "FAILED" ? "warn" : ""}>{job.status}</strong>{job.engine ? ` · ${job.engine}` : ""}</div>
          )}

          {job?.status === "COMPLETED" && job.videoUrl && (
            <>
              <video id="avatar-video" controls crossOrigin="anonymous" src={`${apiBase}${job.videoUrl}?t=${job.jobId}`} />
              {result && (
                <div className="meta">
                  {result.width}×{result.height} · {result.fps} fps · made in {result.renderSeconds}s
                  {result.background ? ` · background: ${result.background}` : ""}
                </div>
              )}
              {result?.watermark && (
                <div id="provenance-line" className={`meta ${result.watermark.applied ? "ok" : "warn"}`}>
                  {result.watermark.applied ? `🔒 invisible watermark verified (${result.watermark.tagBitsMatching}/128 bits)` : `⚠ not watermarked: ${result.watermark.reason}`}
                  {result.manifest?.url && <> · <a id="manifest-link" href={`${apiBase}${result.manifest.url}`} target="_blank" rel="noreferrer">signed manifest</a></>}
                  {" · "}<a id="download-link" href={`${apiBase}${job.videoUrl}`} download>download</a>
                </div>
              )}
              {result?.warnings?.map((w) => <div key={w} className="meta warn">⚠ {w}</div>)}
              <button type="button" className="link" id="lipsync-score-btn" onClick={measure}>Measure lip sync</button>
              {score && (
                <div id="lipsync-score" className="meta">
                  offset {score.offsetFrames} frames · confidence {score.lseC} · {score.secondsWithinOneFrame?.percent ?? "–"}% of seconds in sync
                </div>
              )}
            </>
          )}
          {error && <div className="alert error" id="create-error">{error}</div>}
        </div>
      </div>
    </div>
  );
}
