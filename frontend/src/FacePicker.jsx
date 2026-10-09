import { useCallback, useEffect, useState } from "react"; // useCallback keeps the same function object between renders

/*
 * Step 1 of "Create video": choose the face. Registered faces are shown as portrait cards
 * (GET /api/v1/avatar/faces); a face whose consent record does not permit use is shown but disabled.
 *
 * Adding a face is kept out of the way, behind "Manage faces":
 *   - Add a photo    -> POST /api/v1/avatar/faces (a real person's photo needs their name and a consent basis)
 *   - Generate a face -> POST /api/v1/avatar/generate (Stable Diffusion; depicts nobody)
 *   - Restyle         -> POST /api/v1/avatar/stylize (a copy of the chosen face in another style)
 *
 * Props from CreateVideo: the chosen avatarId and its setter (the parent owns the choice, this
 * component only changes it), plus onEngines / onFaces callbacks that pass the face list reply upward.
 * Generation and restyling run for minutes on a CPU, so they are polled through the same task route.
 */

// The server's error reason as text. FastAPI puts it in `detail`, either a string or an object;
// fall back to a fixed message when there is none.
function detailOf(payload, fallback) {
  const detail = payload?.detail;
  if (!detail) return fallback;
  return typeof detail === "string" ? detail : detail.message || JSON.stringify(detail); // objects are shown by their message, else as raw JSON
}

// Task states that mean "not finished yet": while a task is in one of these, keep polling.
const RUNNING = ["QUEUED", "PROCESSING"];

// `?.` in calls such as onEngines?.(x) below means "call it only if the parent passed it".
export default function FacePicker({ apiBase, avatarId, setAvatarId, onEngines, onFaces }) {
  const [avatars, setAvatars] = useState([]); // the face cards, from GET /api/v1/avatar/faces
  const [consentBases, setConsentBases] = useState([]); // the allowed consent choices, also from the server
  const [error, setError] = useState("");
  const [open, setOpen] = useState(""); // which manage form is showing: "" | "upload" | "generate" | "restyle"

  const [upload, setUpload] = useState({ avatarId: "", subject: "", consentBasis: "", licence: "" }); // form fields for "Add a photo", sent as-is
  const [uploadFile, setUploadFile] = useState(null); // the picked image File
  const [uploading, setUploading] = useState(false); // true while the photo is being checked and stored

  const [genOptions, setGenOptions] = useState(null); // choices for age, presentation, hair, plus availability
  const [gen, setGen] = useState({ avatarId: "", age: "adult", presentation: "person", hair: "short-dark", glasses: false }); // the generate form; defaults until the user picks
  const [styles, setStyles] = useState(null); // restyle choices and availability
  const [restyle, setRestyle] = useState({ style: "painting", newAvatarId: "" }); // the restyle form
  // One task at a time for generate and restyle: both are polled through GET /avatar/generate/{taskId}.
  const [task, setTask] = useState(null);

  /* Fetching and applying are separate so the mount effect can drop a reply that lands after
     unmounting, and set state only from the promise callback, never synchronously in the effect. */
  // useCallback(fn, deps) returns the same function until a dependency changes. That matters
  // because these functions are effect dependencies: a new function every render would re-run the effect.
  const fetchFaces = useCallback(async () => {
    const res = await fetch(`${apiBase}/api/v1/avatar/faces`); // list registered faces
    const payload = await res.json();
    if (!res.ok) throw new Error(detailOf(payload, "Could not list faces")); // the caller decides how to show it
    return payload;
  }, [apiBase]);

  /* Show a face list; keep the current choice when it is still usable, else pick the first usable face. */
  const apply = useCallback((payload) => {
    setAvatars(payload.avatars); // each has avatarId, imageUrl, usable, usabilityReason
    setConsentBases(payload.consentBases);
    onEngines?.(payload.renderEngines); // which lip-sync engines are installed, for the video step
    onFaces?.(payload.avatars); // CreateVideo needs the faces to match a voice to the chosen one
    // A setter given a function receives the current value, so the choice is never read stale.
    setAvatarId((current) => (current && payload.avatars.some((a) => a.avatarId === current && a.usable)
      ? current : payload.avatars.find((a) => a.usable)?.avatarId || ""));
  }, [setAvatarId, onEngines, onFaces]);

  const refresh = useCallback(async () => apply(await fetchFaces()), [apply, fetchFaces]); // reload after adding a face; errors go to the caller

  useEffect(() => {
    let ignore = false;
    // The "ignore" flag: the cleanup below sets it, so a reply that arrives after unmount (or after
    // StrictMode's test unmount in development) changes nothing.
    fetchFaces().then((payload) => !ignore && apply(payload)).catch((err) => !ignore && setError(err.message));
    // The fixed choices for generation and restyling come from the API, not from this file.
    fetch(`${apiBase}/api/v1/avatar/generate/options`).then((r) => (r.ok ? r.json() : null)).then((o) => !ignore && o && setGenOptions(o)).catch(() => {});
    fetch(`${apiBase}/api/v1/avatar/styles`).then((r) => (r.ok ? r.json() : null)).then((s) => !ignore && s && setStyles(s)).catch(() => {});
    return () => { ignore = true; }; // cleanup
  }, [apiBase, fetchFaces, apply]); // runs on mount and again only if one of these changes

  /* Poll a generate / restyle task every 3 s; when it completes, show and select the new face. */
  useEffect(() => {
    if (!task || !RUNNING.includes(task.status)) return undefined; // nothing running: no timer
    const timer = setInterval(async () => { // the polling loop: ask again every 3000 ms
      try {
        const res = await fetch(`${apiBase}/api/v1/avatar/generate/${task.taskId}`); // generate and restyle share this status route
        const body = await res.json();
        if (!res.ok) return; // skip this tick; the next one asks again
        setTask(body); // a new task object re-runs this effect, which replaces the timer
        if (body.status === "COMPLETED") {
          await refresh(); // load the new face into the list first
          setAvatarId(body.result.avatarId); // then select it
        }
      } catch { /* transient: keep polling */ }
    }, 3000);
    return () => clearInterval(timer); // cleanup stops the old timer, so only one loop ever runs
  }, [apiBase, task, refresh, setAvatarId]);

  // POST a JSON body and return the parsed reply, or throw with the server's reason.
  const post = async (path, body, fallback) => {
    setError("");
    const res = await fetch(`${apiBase}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); // JSON.stringify turns the object into the request text
    const payload = await res.json();
    if (!res.ok) throw new Error(detailOf(payload, fallback));
    return payload;
  };

  // Both forms start a background task on the server; the reply ({taskId, status}) becomes `task`,
  // which starts the polling effect above.
  const handleGenerate = (event) => {
    event.preventDefault();
    post("/api/v1/avatar/generate", gen, "Generation was refused").then(setTask).catch((err) => setError(err.message)); // Stable Diffusion; depicts nobody
  };

  const handleRestyle = (event) => {
    event.preventDefault();
    post("/api/v1/avatar/stylize", { avatarId, ...restyle }, "Restyling was refused").then(setTask).catch((err) => setError(err.message)); // the chosen face plus style and new name
  };

  // A photo is a file, so it goes as multipart FormData, not JSON. The browser sets the
  // Content-Type with its boundary itself.
  const handleUpload = async (event) => {
    event.preventDefault(); // keep the page from reloading on submit
    if (!uploadFile) return;
    setUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", uploadFile); // the image bytes
    Object.entries(upload).forEach(([key, value]) => form.append(key, value)); // then avatarId, subject, consentBasis, licence as text fields
    try {
      const res = await fetch(`${apiBase}/api/v1/avatar/faces`, { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(detailOf(body, "The photo was refused"));
      setOpen(""); // close the form on success
      setUploadFile(null);
      await refresh(); // show the new face
      setAvatarId(body.avatar.avatarId); // and select it
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false); // finally: re-enable the button either way
    }
  };

  const busy = RUNNING.includes(task?.status); // a generate or restyle task is still running
  const toggle = (name) => setOpen((current) => (current === name ? "" : name)); // clicking the open form's button again closes it
  const idPattern = "[A-Za-z0-9][A-Za-z0-9_\\-]{0,63}"; // the face-name rule from backend/contracts.py, checked by the browser before sending

  return (
    <div className="card">
      <h2><span className="step">1</span> Face</h2>
      {/* The face grid. aria-pressed marks the chosen card; a face without usable consent is disabled
         and its tooltip (title) says why. */}
      <div className="faces" id="faces" data-selected={avatarId}>
        {avatars.map((a) => (
          <button key={a.avatarId} type="button" className="face" data-avatar={a.avatarId} aria-pressed={a.avatarId === avatarId}
            disabled={!a.usable} title={a.usable ? a.avatarId : a.usabilityReason} onClick={() => setAvatarId(a.avatarId)}>
            {/* loading="lazy" lets the browser fetch off-screen portraits later. */}
            <img src={`${apiBase}${a.imageUrl}`} alt={a.avatarId} loading="lazy" />
            <span>{a.avatarId}</span>
          </button>
        ))}
      </div>
      {avatars.length === 0 && <p className="hint">No faces yet: add a photo or generate one.</p>}

      {/* "Manage faces": each button opens one sub-form below; restyle needs a chosen face. */}
      <div className="manage">
        <button type="button" id="upload-toggle" onClick={() => toggle("upload")}>Add a photo</button>
        <button type="button" id="generate-face-toggle" onClick={() => toggle("generate")}>Generate a face</button>
        <button type="button" id="restyle-toggle" disabled={!avatarId} onClick={() => toggle("restyle")}>Restyle this face</button>
      </div>

      {open === "upload" && (
        // `open === "upload" && (...)` renders the form only while it is the open one.
        <form className="subform" onSubmit={handleUpload}>
          <p className="hint">Only add a face whose owner agreed to it being animated: one person, facing the camera.</p>
          <label className="field"><span>Photo</span>
            {/* `accept` filters the file dialog; the server still checks the image. */}
            <input id="avatar-file" type="file" accept="image/png,image/jpeg,image/webp" required onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
          </label>
          <div className="row">
            <label className="field"><span>Name for this face</span>
              {/* Controlled input: value comes from state and every keystroke writes state back.
                 `{ ...upload, avatarId: x }` copies the object with one field changed; state is never mutated. */}
              <input type="text" required pattern={idPattern} value={upload.avatarId} onChange={(e) => setUpload({ ...upload, avatarId: e.target.value })} />
            </label>
            <label className="field"><span>Who is in the photo</span>
              <input type="text" required value={upload.subject} onChange={(e) => setUpload({ ...upload, subject: e.target.value })} />
            </label>
          </div>
          <label className="field"><span>Their consent</span>
            <select required value={upload.consentBasis} onChange={(e) => setUpload({ ...upload, consentBasis: e.target.value })}>
              <option value="">choose…</option>
              {/* The choices come from the server, so the list matches what it accepts. */}
              {consentBases.map((basis) => <option key={basis} value={basis}>{basis}</option>)}
            </select>
          </label>
          <button type="submit" className="primary" disabled={uploading}>{uploading ? "Checking the photo…" : "Add face"}</button>
        </form>
      )}

      {open === "generate" && (
        // Generate: the button stays disabled while the server says the model is unavailable or a task runs.
        <form className="subform" onSubmit={handleGenerate}>
          <p className="hint">Creates a face that depicts nobody. Takes about two minutes without a GPU.
            {genOptions && !genOptions.available && <span className="warn"> {genOptions.detail}</span>}</p>
          <label className="field"><span>Name for this face</span>
            <input id="generate-avatar-id" type="text" required pattern={idPattern} value={gen.avatarId} onChange={(e) => setGen({ ...gen, avatarId: e.target.value })} />
          </label>
          <div className="row">
            {/* One select per option. Before the options load, each shows only its current value.
               `[key]: value` is a computed property name: the field name comes from the variable. */}
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
        // Restyle: a copy of the chosen face in another style, with a similarity score when done.
        <form className="subform" id="restyle-form" onSubmit={handleRestyle}>
          <p className="hint">Makes a new face from “{avatarId}” in another style. It keeps the original&apos;s consent record and reports how much
            it still looks like the same person.{styles && !styles.available && <span className="warn"> {styles.detail}</span>}</p>
          <div className="row">
            <label className="field"><span>Style</span>
              <select id="restyle-style" value={restyle.style} onChange={(e) => setRestyle({ ...restyle, style: e.target.value })}>
                {/* Style names are the keys of the server's styles object; until it loads, only the default shows. */}
                {Object.keys(styles?.styles ?? { [restyle.style]: null }).map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <label className="field"><span>Name for the new face</span>
              <input id="restyle-avatar-id" type="text" required pattern={idPattern} value={restyle.newAvatarId} onChange={(e) => setRestyle({ ...restyle, newAvatarId: e.target.value })} />
            </label>
          </div>
          <button type="submit" id="restyle-btn" className="primary" disabled={styles?.available === false || busy}>{busy ? "Working…" : "Restyle"}</button>
          {/* After a restyle: how close the new face is to the original person. */}
          {task?.result?.identity && (
            <p id="restyle-identity" className="hint">{task.result.style}: {task.result.identity.percent}% similar to the original
              {task.result.identity.samePerson ? " (same person)" : " (no longer recognised as the same person)"}</p>
          )}
        </form>
      )}

      {/* Task status line for generate and restyle; the seed lets a generated face be reproduced. */}
      {task && (
        <p id="generate-status" className="hint">
          {task.status}{task.status === "COMPLETED" && task.result?.seed !== undefined ? ` · seed ${task.result.seed}` : ""}
        </p>
      )}
      {/* Two error sources: a failed background task, and a refused request (error state). */}
      {task?.status === "FAILED" && <div className="alert error">{task.error}</div>}
      {error && <div className="alert error">{error}</div>}
    </div>
  );
}
