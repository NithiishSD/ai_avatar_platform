import { useCallback, useEffect, useState } from "react";

/*
 * Step 1 of "Create video": choose the face. Registered faces are shown as portrait cards
 * (GET /api/v1/avatar/faces); a face whose consent record does not permit use is shown but disabled.
 *
 * Adding a face is kept out of the way, behind "Manage faces":
 *   - Add a photo    -> POST /api/v1/avatar/faces (a real person's photo needs their name and a consent basis)
 *   - Generate a face -> POST /api/v1/avatar/generate (Stable Diffusion; depicts nobody)
 *   - Restyle         -> POST /api/v1/avatar/stylize (a copy of the chosen face in another style)
 * Generation and restyling run for minutes on a CPU, so they are polled through the same task route.
 */

function detailOf(payload, fallback) {
  const detail = payload?.detail;
  if (!detail) return fallback;
  return typeof detail === "string" ? detail : detail.message || JSON.stringify(detail);
}

const RUNNING = ["QUEUED", "PROCESSING"];

export default function FacePicker({ apiBase, avatarId, setAvatarId, onEngines, onFaces }) {
  const [avatars, setAvatars] = useState([]);
  const [consentBases, setConsentBases] = useState([]);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(""); // which manage form is showing: "" | "upload" | "generate" | "restyle"

  const [upload, setUpload] = useState({ avatarId: "", subject: "", consentBasis: "", licence: "" });
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const [genOptions, setGenOptions] = useState(null);
  const [gen, setGen] = useState({ avatarId: "", age: "adult", presentation: "person", hair: "short-dark", glasses: false });
  const [styles, setStyles] = useState(null);
  const [restyle, setRestyle] = useState({ style: "painting", newAvatarId: "" });
  // One task at a time for generate and restyle: both are polled through GET /avatar/generate/{taskId}.
  const [task, setTask] = useState(null);

  /* Fetching and applying are separate so the mount effect can drop a reply that lands after
     unmounting, and set state only from the promise callback, never synchronously in the effect. */
  const fetchFaces = useCallback(async () => {
    const res = await fetch(`${apiBase}/api/v1/avatar/faces`);
    const payload = await res.json();
    if (!res.ok) throw new Error(detailOf(payload, "Could not list faces"));
    return payload;
  }, [apiBase]);

  /* Show a face list; keep the current choice when it is still usable, else pick the first usable face. */
  const apply = useCallback((payload) => {
    setAvatars(payload.avatars);
    setConsentBases(payload.consentBases);
    onEngines?.(payload.renderEngines);
    onFaces?.(payload.avatars);
    setAvatarId((current) => (current && payload.avatars.some((a) => a.avatarId === current && a.usable)
      ? current : payload.avatars.find((a) => a.usable)?.avatarId || ""));
  }, [setAvatarId, onEngines, onFaces]);

  const refresh = useCallback(async () => apply(await fetchFaces()), [apply, fetchFaces]);

  useEffect(() => {
    let ignore = false;
    fetchFaces().then((payload) => !ignore && apply(payload)).catch((err) => !ignore && setError(err.message));
    // The fixed choices for generation and restyling come from the API, not from this file.
    fetch(`${apiBase}/api/v1/avatar/generate/options`).then((r) => (r.ok ? r.json() : null)).then((o) => !ignore && o && setGenOptions(o)).catch(() => {});
    fetch(`${apiBase}/api/v1/avatar/styles`).then((r) => (r.ok ? r.json() : null)).then((s) => !ignore && s && setStyles(s)).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase, fetchFaces, apply]);

  /* Poll a generate / restyle task every 3 s; when it completes, show and select the new face. */
  useEffect(() => {
    if (!task || !RUNNING.includes(task.status)) return undefined;
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/api/v1/avatar/generate/${task.taskId}`);
        const body = await res.json();
        if (!res.ok) return;
        setTask(body);
        if (body.status === "COMPLETED") {
          await refresh();
          setAvatarId(body.result.avatarId);
        }
      } catch { /* transient: keep polling */ }
    }, 3000);
    return () => clearInterval(timer);
  }, [apiBase, task, refresh, setAvatarId]);

  const post = async (path, body, fallback) => {
    setError("");
    const res = await fetch(`${apiBase}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const payload = await res.json();
    if (!res.ok) throw new Error(detailOf(payload, fallback));
    return payload;
  };

  const handleGenerate = (event) => {
    event.preventDefault();
    post("/api/v1/avatar/generate", gen, "Generation was refused").then(setTask).catch((err) => setError(err.message));
  };

  const handleRestyle = (event) => {
    event.preventDefault();
    post("/api/v1/avatar/stylize", { avatarId, ...restyle }, "Restyling was refused").then(setTask).catch((err) => setError(err.message));
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
      if (!res.ok) throw new Error(detailOf(body, "The photo was refused"));
      setOpen("");
      setUploadFile(null);
      await refresh();
      setAvatarId(body.avatar.avatarId);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const busy = RUNNING.includes(task?.status);
  const toggle = (name) => setOpen((current) => (current === name ? "" : name));
  const idPattern = "[A-Za-z0-9][A-Za-z0-9_\\-]{0,63}";

  return (
    <div className="card">
      <h2><span className="step">1</span> Face</h2>
      <div className="faces" id="faces" data-selected={avatarId}>
        {avatars.map((a) => (
          <button key={a.avatarId} type="button" className="face" data-avatar={a.avatarId} aria-pressed={a.avatarId === avatarId}
            disabled={!a.usable} title={a.usable ? a.avatarId : a.usabilityReason} onClick={() => setAvatarId(a.avatarId)}>
            <img src={`${apiBase}${a.imageUrl}`} alt={a.avatarId} loading="lazy" />
            <span>{a.avatarId}</span>
          </button>
        ))}
      </div>
      {avatars.length === 0 && <p className="hint">No faces yet: add a photo or generate one.</p>}

      <div className="manage">
        <button type="button" id="upload-toggle" onClick={() => toggle("upload")}>Add a photo</button>
        <button type="button" id="generate-face-toggle" onClick={() => toggle("generate")}>Generate a face</button>
        <button type="button" id="restyle-toggle" disabled={!avatarId} onClick={() => toggle("restyle")}>Restyle this face</button>
      </div>

      {open === "upload" && (
        <form className="subform" onSubmit={handleUpload}>
          <p className="hint">Only add a face whose owner agreed to it being animated: one person, facing the camera.</p>
          <label className="field"><span>Photo</span>
            <input id="avatar-file" type="file" accept="image/png,image/jpeg,image/webp" required onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
          </label>
          <div className="row">
            <label className="field"><span>Name for this face</span>
              <input type="text" required pattern={idPattern} value={upload.avatarId} onChange={(e) => setUpload({ ...upload, avatarId: e.target.value })} />
            </label>
            <label className="field"><span>Who is in the photo</span>
              <input type="text" required value={upload.subject} onChange={(e) => setUpload({ ...upload, subject: e.target.value })} />
            </label>
          </div>
          <label className="field"><span>Their consent</span>
            <select required value={upload.consentBasis} onChange={(e) => setUpload({ ...upload, consentBasis: e.target.value })}>
              <option value="">choose…</option>
              {consentBases.map((basis) => <option key={basis} value={basis}>{basis}</option>)}
            </select>
          </label>
          <button type="submit" className="primary" disabled={uploading}>{uploading ? "Checking the photo…" : "Add face"}</button>
        </form>
      )}

      {open === "generate" && (
        <form className="subform" onSubmit={handleGenerate}>
          <p className="hint">Creates a face that depicts nobody. Takes about two minutes without a GPU.
            {genOptions && !genOptions.available && <span className="warn"> {genOptions.detail}</span>}</p>
          <label className="field"><span>Name for this face</span>
            <input id="generate-avatar-id" type="text" required pattern={idPattern} value={gen.avatarId} onChange={(e) => setGen({ ...gen, avatarId: e.target.value })} />
          </label>
          <div className="row">
            {["age", "presentation", "hair"].map((key) => (
              <label key={key} className="field"><span>{key}</span>
                <select value={gen[key]} onChange={(e) => setGen({ ...gen, [key]: e.target.value })}>
                  {(genOptions?.[key] ?? [gen[key]]).map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
            ))}
          </div>
          <label className="check"><input type="checkbox" checked={gen.glasses} onChange={(e) => setGen({ ...gen, glasses: e.target.checked })} /> glasses</label>
          <button type="submit" id="generate-face-btn" className="primary" disabled={genOptions?.available === false || busy}>
            {busy ? "Generating…" : "Generate face"}
          </button>
        </form>
      )}

      {open === "restyle" && (
        <form className="subform" id="restyle-form" onSubmit={handleRestyle}>
          <p className="hint">Makes a new face from “{avatarId}” in another style. It keeps the original&apos;s consent record and reports how much
            it still looks like the same person.{styles && !styles.available && <span className="warn"> {styles.detail}</span>}</p>
          <div className="row">
            <label className="field"><span>Style</span>
              <select id="restyle-style" value={restyle.style} onChange={(e) => setRestyle({ ...restyle, style: e.target.value })}>
                {Object.keys(styles?.styles ?? { [restyle.style]: null }).map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <label className="field"><span>Name for the new face</span>
              <input id="restyle-avatar-id" type="text" required pattern={idPattern} value={restyle.newAvatarId} onChange={(e) => setRestyle({ ...restyle, newAvatarId: e.target.value })} />
            </label>
          </div>
          <button type="submit" id="restyle-btn" className="primary" disabled={styles?.available === false || busy}>{busy ? "Working…" : "Restyle"}</button>
          {task?.result?.identity && (
            <p id="restyle-identity" className="hint">{task.result.style}: {task.result.identity.percent}% similar to the original
              {task.result.identity.samePerson ? " (same person)" : " (no longer recognised as the same person)"}</p>
          )}
        </form>
      )}

      {task && (
        <p id="generate-status" className="hint">
          {task.status}{task.status === "COMPLETED" && task.result?.seed !== undefined ? ` · seed ${task.result.seed}` : ""}
        </p>
      )}
      {task?.status === "FAILED" && <div className="alert error">{task.error}</div>}
      {error && <div className="alert error">{error}</div>}
    </div>
  );
}
