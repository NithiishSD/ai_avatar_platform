import { useCallback, useEffect, useRef, useState } from "react";

/*
 * Avatar panel: pick (or register) a face, see its landmarks, render the
 * synthesized speech onto it, and measure the lip sync.
 *
 * Everything here talks to the vision routes in backend/app.py. The panel
 * shows what the backend reports -- which engine ran, which warnings it
 * raised, the method behind the sync score -- rather than deciding any of
 * that itself.
 */

const FINISHED = ["COMPLETED", "FAILED"];

const card = {
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: "8px",
  padding: "12px 14px",
  marginBottom: "12px",
};
const small = { fontSize: "0.78rem", color: "#94a3b8" };
const field = {
  width: "100%",
  padding: "6px 8px",
  marginTop: "4px",
  background: "#1e293b",
  color: "#e2e8f0",
  border: "1px solid #334155",
  borderRadius: "6px",
};

function detailOf(payload, fallback) {
  const detail = payload?.detail;
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  return detail.message || JSON.stringify(detail);
}

/* Photo with its landmarks and bounding box drawn on a canvas (task G1-05). */
function LandmarkCanvas({ imageUrl, analysis, showMesh }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return undefined;
    let cancelled = false;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      if (cancelled) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);
      if (!analysis || !showMesh) return;
      const { width, height } = canvas;
      const box = analysis.boundingBox;
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = Math.max(1, width / 256);
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      const radius = Math.max(0.8, width / 420);
      (analysis.landmarks || []).forEach((point, index) => {
        // Points 468-477 are the iris ring; draw them in a second colour.
        ctx.fillStyle = index >= 468 ? "#f472b6" : "#6ee7b7";
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, radius, 0, 2 * Math.PI);
        ctx.fill();
      });
    };
    image.src = imageUrl;
    return () => {
      cancelled = true;
    };
  }, [imageUrl, analysis, showMesh]);

  return (
    <canvas
      ref={canvasRef}
      id="avatar-landmark-canvas"
      style={{ width: "100%", borderRadius: "8px", display: "block", background: "#020617" }}
    />
  );
}

export default function AvatarPanel({ apiBase, phonemeTimestamps, emotionVector, audioReady, alignmentMethod }) {
  const [avatars, setAvatars] = useState([]);
  const [consentBases, setConsentBases] = useState([]);
  const [engines, setEngines] = useState({ blendshape: true });
  const [avatarId, setAvatarId] = useState("");
  // A face analysis is stored with the avatarId it describes and only read
  // back while that avatar is still selected. Switching avatars therefore
  // shows nothing - not the previous face's landmarks - until the new result
  // lands, with no reset-to-null inside an effect.
  const [faceResult, setFaceResult] = useState({ avatarId: "", analysis: null, quality: null });
  const currentResult = faceResult.avatarId === avatarId ? faceResult : null;
  const analysis = currentResult?.analysis ?? null;
  const quality = currentResult?.quality ?? null;
  const [showMesh, setShowMesh] = useState(true);
  const [engine, setEngine] = useState("blendshape");
  const [renderQuality, setRenderQuality] = useState("PREVIEW");
  // Background replacement: off, or one flat colour (the API also accepts an
  // image under outputs/ - used by scripts, not offered here).
  const [replaceBackground, setReplaceBackground] = useState(false);
  const [backgroundColor, setBackgroundColor] = useState("#0b3d91");
  const [job, setJob] = useState(null);
  const [score, setScore] = useState(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [upload, setUpload] = useState({ avatarId: "", subject: "", consentBasis: "", licence: "" });
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  /* Synthetic face generation (Stable Diffusion, a background task). */
  const [showGenerate, setShowGenerate] = useState(false);
  const [genOptions, setGenOptions] = useState(null);
  const [gen, setGen] = useState({ avatarId: "", age: "adult", presentation: "person", hair: "short-dark", glasses: false });
  const [genTask, setGenTask] = useState(null);

  /* Fetching and applying are separate so the mount effect can drop a
     response that arrives after the panel unmounts, while the registration
     handler simply awaits both. */
  const fetchAvatarList = useCallback(async () => {
    const res = await fetch(`${apiBase}/api/v1/avatar/faces`);
    const payload = await res.json();
    if (!res.ok) throw new Error(detailOf(payload, "Could not list avatars"));
    return payload;
  }, [apiBase]);

  const applyAvatarList = useCallback((payload) => {
    setAvatars(payload.avatars);
    setConsentBases(payload.consentBases);
    setEngines(payload.renderEngines);
    setAvatarId((current) => {
      if (current && payload.avatars.some((a) => a.avatarId === current)) return current;
      return payload.avatars.find((a) => a.usable)?.avatarId || "";
    });
  }, []);

  useEffect(() => {
    let ignore = false;
    fetchAvatarList()
      .then((payload) => {
        if (!ignore) applyAvatarList(payload);
      })
      .catch((err) => {
        if (!ignore) setError(err.message);
      });
    return () => { ignore = true; };
  }, [fetchAvatarList, applyAvatarList]);

  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/api/v1/avatar/generate/options`)
      .then((res) => (res.ok ? res.json() : null))
      .then((options) => {
        if (!ignore && options) setGenOptions(options);
      })
      .catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  /* Poll the generation until it finishes; on success refresh the list and
     select the new face. Generation takes seconds on a GPU, minutes on a CPU. */
  useEffect(() => {
    if (!genTask || ["COMPLETED", "FAILED"].includes(genTask.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/api/v1/avatar/generate/${genTask.taskId}`);
        const body = await res.json();
        if (!res.ok) return;
        setGenTask(body);
        if (body.status === "COMPLETED") {
          applyAvatarList(await fetchAvatarList());
          setAvatarId(body.result.avatarId);
        }
      } catch {
        /* transient: keep polling */
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [apiBase, genTask, applyAvatarList, fetchAvatarList]);

  const handleGenerate = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const res = await fetch(`${apiBase}/api/v1/avatar/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(gen),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Generation was rejected"));
      setGenTask(body);
    } catch (err) {
      setError(err.message);
    }
  };

  /* Landmarks, pose and the quality verdict for the selected face. */
  useEffect(() => {
    if (!avatarId) return undefined;
    let cancelled = false;
    const form = new FormData();
    form.append("avatarId", avatarId);
    form.append("includeLandmarks", "true");
    fetch(`${apiBase}/api/v1/avatar/face/analyze`, { method: "POST", body: form })
      .then(async (res) => {
        const payload = await res.json();
        if (!res.ok) throw new Error(detailOf(payload, "Face analysis failed"));
        if (!cancelled) {
          setFaceResult({ avatarId, analysis: payload.analysis, quality: payload.quality });
        }
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [apiBase, avatarId]);

  /* Poll the render job until it finishes. */
  useEffect(() => {
    if (!job || FINISHED.includes(job.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(job.jobId)}`);
        const payload = await res.json();
        if (res.ok) setJob(payload);
      } catch {
        /* transient: keep polling */
      }
    }, 1000); // well under the API's default 120 requests/minute limit
    return () => clearInterval(timer);
  }, [apiBase, job]);

  const selected = avatars.find((a) => a.avatarId === avatarId);
  const canRender = Boolean(selected?.usable && audioReady && phonemeTimestamps.length > 0);
  const rendering = job && !FINISHED.includes(job.status);

  const handleRender = async () => {
    setError("");
    setScore(null);
    const duration = phonemeTimestamps[phonemeTimestamps.length - 1].endMs / 1000.0;
    const payload = {
      jobId: `JOB-${Date.now()}`,
      avatarId,
      audioUrl: `${apiBase}/outputs/speech.wav`,
      sampleRate: 24000,
      durationSeconds: Math.max(duration, 0.5),
      phonemeTimestamps,
      emotionVector,
      renderQuality,
      targetFps: 25,
      ...(replaceBackground ? { background: { color: backgroundColor } } : {}),
    };
    try {
      const res = await fetch(
        `${apiBase}/api/v1/avatar/render-job?engine=${encodeURIComponent(engine)}`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
      );
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Render job was rejected"));
      setJob(body);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleScore = async () => {
    setScoring(true);
    setError("");
    try {
      const res = await fetch(`${apiBase}/api/v1/avatar/render-job/${encodeURIComponent(job.jobId)}/lipsync-score`, { method: "POST" });
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Lip-sync scoring failed"));
      setScore(body.score);
    } catch (err) {
      setError(err.message);
    } finally {
      setScoring(false);
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!uploadFile) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", uploadFile);
    Object.entries(upload).forEach(([key, value]) => form.append(key, value));
    try {
      const res = await fetch(`${apiBase}/api/v1/avatar/faces`, { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "Registration failed"));
      setShowUpload(false);
      setUploadFile(null);
      applyAvatarList(await fetchAvatarList());
      setAvatarId(body.avatar.avatarId);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const result = job?.result;

  return (
    <div id="avatar-panel">
      <h2 style={{ marginTop: "1.5rem" }}>Avatar & Video</h2>

      <div style={card}>
        <label style={small}>
          Avatar face
          <select id="avatar-select" style={field} value={avatarId} onChange={(e) => setAvatarId(e.target.value)}>
            {avatars.length === 0 && <option value="">no avatars registered</option>}
            {avatars.map((a) => (
              <option key={a.avatarId} value={a.avatarId} disabled={!a.usable}>
                {a.avatarId} ({a.provenance?.source || "no record"}){a.usable ? "" : " - blocked"}
              </option>
            ))}
          </select>
        </label>
        {avatars.length === 0 && (
          <div style={{ ...small, marginTop: "6px" }}>
            Create one with <code>scripts/make_avatar.py --synthetic --avatar-id demo</code>, or register a photo below.
          </div>
        )}
        {selected && (
          <div style={{ marginTop: "10px" }}>
            <LandmarkCanvas
              imageUrl={selected.usable ? `${apiBase}${selected.imageUrl}` : ""}
              analysis={analysis}
              showMesh={showMesh}
            />
            <div style={{ ...small, marginTop: "6px" }}>{selected.usabilityReason}</div>
            {analysis && (
              <div style={{ ...small, marginTop: "4px" }}>
                {analysis.landmarkCount} landmarks · yaw {analysis.headPose.yaw}° · pitch {analysis.headPose.pitch}° ·
                roll {analysis.headPose.roll}° · {Object.keys(analysis.blendshapes).length} blendshapes
              </div>
            )}
            {quality?.warnings?.map((w) => (
              <div key={w.code} style={{ ...small, color: "#fbbf24", marginTop: "4px" }}>⚠ {w.message}</div>
            ))}
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              <input type="checkbox" checked={showMesh} onChange={(e) => setShowMesh(e.target.checked)} /> show
              landmarks
            </label>
          </div>
        )}
        <button type="button" className="secondary" style={{ marginTop: "10px" }} onClick={() => setShowUpload((v) => !v)}>
          {showUpload ? "Cancel" : "Register a photo"}
        </button>
        <button type="button" className="secondary" id="generate-face-toggle"
          style={{ marginTop: "10px", marginLeft: "8px" }} onClick={() => setShowGenerate((v) => !v)}>
          {showGenerate ? "Cancel" : "Generate a synthetic face"}
        </button>
        {showGenerate && (
          <form onSubmit={handleGenerate} style={{ marginTop: "10px" }}>
            <div style={{ ...small, marginBottom: "6px" }}>
              Creates a face that depicts nobody (Stable Diffusion 1.5). The choices are fixed so a real
              person cannot be described by name.
              {genOptions && !genOptions.available && (
                <div style={{ color: "#fbbf24" }}>⚠ {genOptions.detail}</div>
              )}
            </div>
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              Avatar id
              <input id="generate-avatar-id" style={field} required pattern="[A-Za-z0-9][A-Za-z0-9_\-]{0,63}"
                value={gen.avatarId} onChange={(e) => setGen({ ...gen, avatarId: e.target.value })} />
            </label>
            {["age", "presentation", "hair"].map((key) => (
              <label key={key} style={{ ...small, display: "block", marginTop: "6px" }}>
                {key}
                <select style={field} value={gen[key]} onChange={(e) => setGen({ ...gen, [key]: e.target.value })}>
                  {(genOptions?.[key] ?? [gen[key]]).map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
            ))}
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              <input type="checkbox" checked={gen.glasses}
                onChange={(e) => setGen({ ...gen, glasses: e.target.checked })} /> glasses
            </label>
            <button type="submit" id="generate-face-btn" style={{ marginTop: "10px" }}
              disabled={genOptions?.available === false || ["QUEUED", "PROCESSING"].includes(genTask?.status)}>
              {["QUEUED", "PROCESSING"].includes(genTask?.status) ? "Generating…" : "Generate face"}
            </button>
            {genTask && (
              <div id="generate-status" style={{ ...small, marginTop: "6px" }}>
                {genTask.status}
                {genTask.status === "COMPLETED" && ` · seed ${genTask.result.seed}`}
                {genTask.status === "FAILED" && <div className="alert error">{genTask.error}</div>}
              </div>
            )}
          </form>
        )}
        {showUpload && (
          <form onSubmit={handleUpload} style={{ marginTop: "10px" }}>
            <div style={{ ...small, marginBottom: "6px" }}>
              Only register a face whose owner has agreed to it being animated. The photo must show one person
              facing the camera.
            </div>
            <input id="avatar-file" type="file" accept="image/png,image/jpeg,image/webp" required
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              Avatar id
              <input style={field} required pattern="[A-Za-z0-9][A-Za-z0-9_\-]{0,63}" value={upload.avatarId}
                onChange={(e) => setUpload({ ...upload, avatarId: e.target.value })} />
            </label>
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              Who is in the photo
              <input style={field} required value={upload.subject}
                onChange={(e) => setUpload({ ...upload, subject: e.target.value })} />
            </label>
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              Consent basis
              <select style={field} required value={upload.consentBasis}
                onChange={(e) => setUpload({ ...upload, consentBasis: e.target.value })}>
                <option value="">— choose —</option>
                {consentBases.map((basis) => (
                  <option key={basis} value={basis}>{basis}</option>
                ))}
              </select>
            </label>
            <label style={{ ...small, display: "block", marginTop: "6px" }}>
              Licence / note (optional)
              <input style={field} value={upload.licence}
                onChange={(e) => setUpload({ ...upload, licence: e.target.value })} />
            </label>
            <button type="submit" style={{ marginTop: "10px" }} disabled={uploading}>
              {uploading ? "Checking photo…" : "Register avatar"}
            </button>
          </form>
        )}
      </div>

      <div style={card}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          <label style={small}>
            Lip-sync engine
            <select id="engine-select" style={field} value={engine} onChange={(e) => setEngine(e.target.value)}>
              <option value="blendshape">blendshape (CPU)</option>
              <option value="wav2lip" disabled={!engines.wav2lip}>
                wav2lip (GPU){engines.wav2lip ? "" : " - weights not fetched"}
              </option>
            </select>
          </label>
          <label style={small}>
            Quality
            <select style={field} value={renderQuality} onChange={(e) => setRenderQuality(e.target.value)}>
              <option value="PREVIEW">Preview (≤ 512 px)</option>
              <option value="1080P_HQ">1080p HQ</option>
            </select>
          </label>
        </div>
        <div style={{ marginTop: "10px", display: "flex", alignItems: "center", gap: "10px" }}>
          <label style={small}>
            <input id="background-toggle" type="checkbox" checked={replaceBackground}
              onChange={(e) => setReplaceBackground(e.target.checked)} /> replace background
          </label>
          {replaceBackground && (
            <input id="background-color" type="color" aria-label="background colour" value={backgroundColor}
              onChange={(e) => setBackgroundColor(e.target.value)} />
          )}
        </div>
        <button type="button" id="render-video-btn" style={{ marginTop: "10px", width: "100%" }}
          disabled={!canRender || rendering} onClick={handleRender}>
          {rendering ? "Rendering…" : "Render Avatar Video"}
        </button>
        {alignmentMethod === "acoustic-fallback" && (
          <div style={{ ...small, color: "#fbbf24", marginTop: "6px" }}>
            ⚠ Phoneme timing for this clip was estimated, not measured (the forced aligner could not run).
            Lip sync will be loose.
          </div>
        )}
        {!canRender && (
          <div style={{ ...small, marginTop: "6px" }}>
            {!selected?.usable
              ? "Select a usable avatar first."
              : "Generate speech with “return alignment” on; the timestamps drive the mouth."}
          </div>
        )}

        {job && (
          <div style={{ marginTop: "10px" }}>
            <div style={small}>
              {job.jobId} · <strong id="render-status" style={{ color: job.status === "FAILED" ? "#f87171" : "#e2e8f0" }}>
                {job.status}</strong>{job.engine ? ` · ${job.engine}` : ""}
            </div>
            {rendering && (
              <div style={{ height: "6px", background: "#1e293b", borderRadius: "3px", marginTop: "6px" }}>
                <div style={{
                  height: "100%", width: `${Math.round((job.progress || 0) * 100)}%`,
                  background: "#38bdf8", borderRadius: "3px", transition: "width 0.3s",
                }} />
              </div>
            )}
            {job.status === "FAILED" && <div className="alert error" style={{ marginTop: "8px" }}>{job.error}</div>}
            {job.status === "COMPLETED" && job.videoUrl && (
              <>
                <video id="avatar-video" controls crossOrigin="anonymous" src={`${apiBase}${job.videoUrl}?t=${job.jobId}`}
                  style={{ width: "100%", borderRadius: "8px", marginTop: "8px", background: "#000" }} />
                {result && (
                  <div style={{ ...small, marginTop: "6px" }}>
                    {result.frameCount} frames · {result.width}×{result.height} @ {result.fps} fps ·
                    rendered in {result.renderSeconds}s ({result.realtimeFactor}× real time)
                    {result.peakVramMb != null ? ` · peak VRAM ${result.peakVramMb} MiB` : ""}
                    {result.background ? ` · background: ${result.background}` : ""}
                  </div>
                )}
                {result?.watermark && (
                  <div id="provenance-line" style={{ ...small, marginTop: "4px", color: result.watermark.applied ? "#6ee7b7" : "#fbbf24" }}>
                    {result.watermark.applied
                      ? `🔒 invisible watermark verified (${result.watermark.tagBitsMatching}/128 bits)`
                      : `⚠ not watermarked: ${result.watermark.reason}`}
                    {result.manifest?.url && (
                      <> · <a id="manifest-link" href={`${apiBase}${result.manifest.url}`} target="_blank" rel="noreferrer"
                        style={{ color: "#38bdf8" }}>signed manifest</a></>
                    )}
                  </div>
                )}
                {result?.warnings?.map((w) => (
                  <div key={w} style={{ ...small, color: "#fbbf24", marginTop: "4px" }}>⚠ {w}</div>
                ))}
                <button type="button" className="secondary" id="lipsync-score-btn" style={{ marginTop: "8px" }}
                  disabled={scoring} onClick={handleScore}>
                  {scoring ? "Scoring…" : "Measure lip sync (SyncNet)"}
                </button>
                {score && (
                  <div id="lipsync-score" style={{ ...small, marginTop: "6px", color: "#e2e8f0" }}>
                    LSE-C {score.lseC} (higher is better) · LSE-D {score.lseD} (lower is better) · offset{" "}
                    {score.offsetFrames} frames
                    <div style={small}>{score.method}</div>
                    {score.warnings?.map((w) => (
                      <div key={w} style={{ color: "#fbbf24" }}>⚠ {w}</div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {error ? <div className="alert error">{error}</div> : null}
    </div>
  );
}
