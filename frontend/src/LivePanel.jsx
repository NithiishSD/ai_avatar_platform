import { useCallback, useEffect, useRef, useState } from "react"; // hooks; useRef is used heavily here, see below
import { recordedGender, rememberVoice, voiceForFace } from "./voices"; // same face-to-voice matching as the create screen

/*
 * Live avatar panel: open a WebSocket session, type something, and watch the avatar
 * say it while it is still being generated (WS /api/v1/live, see docs/05-API.md).
 *
 * The server sends, per sentence, an `audio` message and then binary media: PCM16
 * audio chunks and JPEG frames, each with a 13-byte header ending in
 * `presentationMs`, the time it is due on the session's timeline. Playing by that
 * time, not by arrival, is what keeps the mouth in step with the sound:
 *
 *   - audio is decoded to an AudioBuffer and scheduled on the Web Audio clock;
 *   - frames are decoded to bitmaps and drawn when the SAME clock reaches their time.
 *
 * Counters on the panel (`data-*` and the visible text) are the observable record of
 * what arrived; the E2E test reads them.
 *
 * Microphone (R-41): instead of text, the user's own voice can drive the face. The mic is
 * captured with Web Audio, cut into 0.5 s PCM16 chunks and sent as binary messages after an
 * `audio_start` that states the consent basis. Only frames come back (the user already hears
 *
 * Other routes: GET /api/v1/avatar/faces and GET /api/v1/audio/voices fill the two selects.
 * themself); the mouth follows loudness, an estimate the panel says out loud.
 */

// Binary header (docs/05-API.md): kind u8 | chunk u32 | index u32 | presentationMs u32, big-endian.
// Byte offsets: kind at 0, chunk at 1, index at 5, presentationMs at 9; the payload starts at 13.
const HEADER_BYTES = 13;
const KIND_AUDIO = 1; // payload is PCM16 mono audio at the `ready` sampleRate
const KIND_FRAME = 2; // payload is one JPEG frame
// Scheduling lead: audio is booked this far ahead of "now" so the first chunk is
// not already late by the time it is decoded.
const LEAD_S = 0.15;
// Microphone audio goes to the server in chunks of this length; frames for a chunk come back after it.
const MIC_CHUNK_S = 0.5;

// Inline style objects shared by the elements below.
const card = { background: "#0f172a", border: "1px solid #334155", borderRadius: "8px", padding: "12px 14px", marginBottom: "12px" };
const small = { fontSize: "0.78rem", color: "#94a3b8" };
const field = { width: "100%", padding: "6px 8px", marginTop: "4px", background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: "6px" };

// http://host -> ws://host and https://host -> wss://host: the WebSocket scheme for the same server.
function wsUrl(apiBase) {
  return `${apiBase.replace(/^http/, "ws")}/api/v1/live`;
}

export default function LivePanel({ apiBase }) {
  const [avatars, setAvatars] = useState([]);
  const [avatarId, setAvatarId] = useState("");
  // Screen state (useState): changing any of these re-renders the panel.
  const [state, setState] = useState("idle"); // idle | connecting | ready | speaking | closed
  const [error, setError] = useState("");
  const [text, setText] = useState("Hello! I am speaking live. Each sentence reaches you as soon as it is ready.");
  const [stats, setStats] = useState({ chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null, done: null }); // a copy of countersRef for display
  const [info, setInfo] = useState(null); // the server's `ready` message: size, fps, model
  // Speaker voice, matched to the face as on the create screen (voices.js).
  const [voices, setVoices] = useState([]);
  const [chosenSpeaker, setChosenSpeaker] = useState({});
  const [motion, setMotion] = useState(1.0); // head-and-shoulder movement, as on the create screen

  // Refs (useRef): values the audio and drawing code reads many times a second. Writing a ref does
  // not re-render, and callbacks always read the current value instead of the one from an old render.
  const socketRef = useRef(null); // the open WebSocket, or null
  const canvasRef = useRef(null); // the <canvas> element; React fills it via ref={canvasRef}
  const audioRef = useRef(null);       // AudioContext
  const anchorRef = useRef(0);         // audio-clock time at which timeline 0 plays
  const readyRef = useRef(null);       // the server's `ready` message
  const framesRef = useRef([]);        // decoded frames waiting for their time: {due, bitmap}
  const sayAtRef = useRef(0); // performance.now() when "Speak" was pressed, for first-audio / first-frame times
  const countersRef = useRef({ chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null }); // counted in the ref, copied to `stats` by publish()
  // Every audio chunk already handed to Web Audio and not yet finished. A chunk is scheduled ahead of
  // time, so stopping the server is not enough: Interrupt must also stop what is queued here, or the
  // voice keeps talking after the picture has stopped (the owner's M-05 finding).
  const sourcesRef = useRef(new Set());
  const [playing, setPlaying] = useState(0); // how many audio chunks are queued or playing (data-playing, read by the E2E test)
  const [micBasis, setMicBasis] = useState(""); // consent basis for the microphone, sent in audio_start
  const [micOn, setMicOn] = useState(false); // is the microphone streaming?
  const micRef = useRef(null);          // {stream, source, processor, sink, pending: Float32Array[]}
  const micAnchorRef = useRef(false);   // true until the first mic frame sets the drawing clock

  useEffect(() => {
    let ignore = false;
    // Load the faces once. The `ignore` flag, set by the cleanup, drops a reply that lands after unmount.
    fetch(`${apiBase}/api/v1/avatar/faces`)
      .then((res) => (res.ok ? res.json() : null)) // an error reply leaves the list empty
      .then((payload) => {
        if (ignore || !payload) return;
        setAvatars(payload.avatars);
        setAvatarId((current) => current || payload.avatars.find((a) => a.usable)?.avatarId || ""); // keep a choice, else the first usable face
      })
      .catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  useEffect(() => {
    let ignore = false;
    // Speaker voices; if this fails the speaker select is simply not shown.
    fetch(`${apiBase}/api/v1/audio/voices`).then((r) => (r.ok ? r.json() : null)).then((d) => !ignore && d && setVoices(d.voices ?? [])).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);
  const face = avatars.find((a) => a.avatarId === avatarId);
  const speaker = chosenSpeaker[avatarId] || voiceForFace(face, voices, "af_heart"); // chosen here, else remembered, else matched to the face

  // Copy the counters into state so the visible numbers update. `done` is cleared because a new
  // utterance is in progress. useCallback keeps one function so effects and callbacks that list it stay stable.
  const publish = useCallback(() => setStats({ ...countersRef.current, done: null }), []);

  // One binary message from the server. The socket was opened with binaryType "arraybuffer", so
  // `buffer` is raw bytes. A DataView reads numbers of a chosen size at a byte offset; it reads
  // big-endian unless told otherwise, which matches the header.
  const handleMedia = useCallback((buffer) => {
    const view = new DataView(buffer);
    const kind = view.getUint8(0); // 1 = audio, 2 = frame
    const presentationMs = view.getUint32(9); // when this media is due on the session timeline
    const payload = buffer.slice(HEADER_BYTES); // everything after the header, as a new ArrayBuffer
    const audio = audioRef.current;
    const ready = readyRef.current;
    if (!audio || !ready) return; // no clock or no `ready` yet: nothing to schedule against
    const c = countersRef.current; // short name; mutated in place, published at the end

    if (kind === KIND_AUDIO) {
      // Re-anchor if the timeline's "now" has slipped into the past (the user waited
      // between two `say`s), so a late chunk is played, not skipped.
      if (anchorRef.current === 0 || anchorRef.current + presentationMs / 1000 < audio.currentTime) {
        anchorRef.current = audio.currentTime + LEAD_S - presentationMs / 1000; // so this chunk plays LEAD_S from now
      }
      const samples = new Int16Array(payload, 0, Math.floor(payload.byteLength / 2)); // view the bytes as 16-bit samples (2 bytes each)
      const floats = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i += 1) floats[i] = samples[i] / 32768; // Web Audio wants floats in -1..1
      const buf = audio.createBuffer(1, floats.length, ready.sampleRate); // mono, at the rate the server announced
      buf.copyToChannel(floats, 0);
      const source = audio.createBufferSource(); // a one-shot player for that buffer
      source.buffer = buf;
      source.connect(audio.destination); // destination = the speakers
      sourcesRef.current.add(source); // track it so silence() can stop it
      source.onended = () => { // finished or stopped: forget it
        sourcesRef.current.delete(source);
        setPlaying(sourcesRef.current.size);
      };
      setPlaying(sourcesRef.current.size);
      // The Web Audio clock (audio.currentTime, in seconds) is sample-accurate and keeps running while
      // JavaScript is busy. start(t) books the chunk for time t on that clock, so playback follows
      // presentationMs, not the moment the message arrived.
      source.start(anchorRef.current + presentationMs / 1000);
      c.chunks += 1;
      c.audioBytes += payload.byteLength;
      if (c.firstAudioMs === null) c.firstAudioMs = Math.round(performance.now() - sayAtRef.current);
    } else if (kind === KIND_FRAME) {
      // Mic frames have no audio to anchor to: the first one sets the clock, a little ahead of now.
      if (micAnchorRef.current) {
        anchorRef.current = audio.currentTime + 0.1 - presentationMs / 1000;
        micAnchorRef.current = false;
      }
      const due = anchorRef.current + presentationMs / 1000; // the frame's time on the same audio clock
      // createImageBitmap decodes the JPEG asynchronously into a bitmap that drawImage can paint
      // quickly. Decoding is async, so the frame joins the queue when it is ready.
      createImageBitmap(new Blob([payload], { type: "image/jpeg" })).then((bitmap) => {
        framesRef.current.push({ due, bitmap }); // the draw loop below paints it when due
      });
      c.frames += 1;
      if (c.firstFrameMs === null) c.firstFrameMs = Math.round(performance.now() - sayAtRef.current);
    }
    publish();
  }, [publish]); // publish never changes, so this callback is made once

  /* Silence everything already scheduled and forget the timeline, so the next utterance starts fresh. */
  const silence = useCallback(() => {
    for (const source of sourcesRef.current) { // a Set is iterable
      try { source.stop(); } catch { /* never started or already ended */ }
    }
    sourcesRef.current.clear(); // stopped sources may still fire onended; deleting from an empty Set is harmless
    setPlaying(0);
    framesRef.current = []; // drop frames not yet shown
    anchorRef.current = 0; // 0 means "no anchor": the next audio chunk sets one
  }, []);

  // End the session: silence locally, tell the server, close the socket.
  const stop = useCallback(() => {
    silence();
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) { // only an open socket can carry the stop message
      try { socket.send(JSON.stringify({ type: "stop" })); } catch { /* already closing */ }
    }
    socket?.close(); // the onclose handler set in start() then marks the state "closed"
    socketRef.current = null;
  }, [silence]);

  /* Draw whichever decoded frame is due on the audio clock; one loop for the panel's life. */
  useEffect(() => {
    let id = 0; // the pending animation frame, so cleanup can cancel it
    // requestAnimationFrame calls drawDue before each screen repaint (about 60 times a second).
    function drawDue() {
      const audio = audioRef.current;
      const canvas = canvasRef.current;
      if (audio && canvas) {
        const queue = framesRef.current;
        let latest = null;
        while (queue.length && queue[0].due <= audio.currentTime) { // the queue is in decode order; take every frame whose time has come
          if (latest) latest.bitmap.close(); // a later frame is also due: skip this one and free its memory
          latest = queue.shift();
        }
        if (latest) {
          canvas.getContext("2d").drawImage(latest.bitmap, 0, 0, canvas.width, canvas.height); // paint only the newest due frame
          latest.bitmap.close(); // release the decoded image now that it is drawn
        }
      }
      id = requestAnimationFrame(drawDue); // schedule the next pass
    }
    id = requestAnimationFrame(drawDue); // start the loop
    return () => cancelAnimationFrame(id); // cleanup: stop the loop on unmount
  }, []);

  // An effect that only returns a cleanup: nothing on mount; on unmount release the microphone,
  // the socket and the AudioContext.
  useEffect(() => () => {
    micRef.current?.stream.getTracks().forEach((track) => track.stop());
    stop();
    audioRef.current?.close();
  }, [stop]);

  // "Start live session": reset counters and the timeline, make a fresh AudioContext, open the socket.
  const start = () => {
    setError("");
    setState("connecting");
    countersRef.current = { chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null };
    publish();
    framesRef.current = [];
    anchorRef.current = 0;
    // Created inside the click handler: browsers only let audio start from a user gesture.
    audioRef.current?.close();
    audioRef.current = new AudioContext();
    audioRef.current.resume();

    const socket = new WebSocket(wsUrl(apiBase)); // connects in the background; onopen fires when ready
    socket.binaryType = "arraybuffer"; // binary messages arrive as ArrayBuffer, not Blob, so DataView can read them at once
    socketRef.current = socket;
    // The first message must be `start`. The voice is only sent when the voice list loaded.
    socket.onopen = () => socket.send(JSON.stringify({ type: "start", avatarId, fps: 25, maxSide: 384, motionIntensity: motion, ...(voices.length ? { voice: speaker } : {}) }));
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") return handleMedia(event.data); // binary = media; text = a JSON control message
      const message = JSON.parse(event.data);
      if (message.type === "ready") { // session accepted: size, fps, sampleRate, model
        readyRef.current = message; // handleMedia needs sampleRate
        setInfo(message);
        if (canvasRef.current) {
          canvasRef.current.width = message.width; // the canvas pixel size matches the frames
          canvasRef.current.height = message.height;
        }
        setState("ready");
      } else if (message.type === "done") { // all sentences sent, with totals
        setStats((s) => ({ ...s, done: message }));
        setState("ready");
      } else if (message.type === "interrupted") { // the server stopped speaking
        silence();
        setState("ready");
      } else if (message.type === "error") {
        setError(`${message.code}: ${message.detail}`);
        setState((s) => (s === "speaking" ? "ready" : s)); // an error mid-speech returns to ready; the session may carry on
      }
    };
    socket.onerror = () => setError("Could not reach the live endpoint. Is the backend running?"); // browsers give no detail on WebSocket errors
    socket.onclose = () => {
      setState((s) => (s === "idle" ? s : "closed")); // an updater function reads the latest state, not the one this closure saw
      socketRef.current = null;
    };
  };

  // Send one `say`. Audio and frames then arrive through handleMedia.
  const speak = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    countersRef.current = { chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null };
    publish();
    sayAtRef.current = performance.now(); // start the first-audio / first-frame stopwatch
    audioRef.current?.resume(); // resume() is a no-op when already running
    socket.send(JSON.stringify({ type: "say", text }));
    setState("speaking");
  };

  const interrupt = () => {
    socketRef.current?.send(JSON.stringify({ type: "interrupt" })); // the session stays open
    silence(); // at once, not when the server's reply arrives
  };

  /* Send the samples gathered so far as one PCM16 binary message. */
  const flushMic = (socket, mic) => {
    const total = mic.pending.reduce((n, part) => n + part.length, 0); // samples gathered across callbacks
    if (!total) return;
    const pcm = new Int16Array(total); // Int16Array uses the machine byte order: little-endian on common hardware, as the server expects
    let offset = 0;
    for (const part of mic.pending) {
      for (let i = 0; i < part.length; i += 1) pcm[offset + i] = Math.max(-1, Math.min(1, part[i])) * 32767; // clamp, then float to 16-bit
      offset += part.length;
    }
    mic.pending = [];
    socket.send(pcm.buffer); // an ArrayBuffer is sent as a binary message
  };

  // Microphone: ask for it, announce `audio_start`, then stream 0.5 s PCM16 chunks.
  const startMic = async () => {
    const socket = socketRef.current;
    const audio = audioRef.current;
    if (!socket || !audio) return;
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } }); // the browser asks the user for permission here
      socket.send(JSON.stringify({ type: "audio_start", sampleRate: audio.sampleRate, consentBasis: micBasis })); // consent is stated before any audio
      const source = audio.createMediaStreamSource(stream); // the mic as a Web Audio node
      // ScriptProcessor is old but everywhere and enough for 0.5 s chunks; a silent gain keeps it
      // running without playing the user's own voice back to them.
      const processor = audio.createScriptProcessor(4096, 1, 1); // calls onaudioprocess with 4096-sample blocks, 1 channel in and out
      const sink = audio.createGain();
      sink.gain.value = 0; // volume 0: nothing is heard
      const mic = { stream, source, processor, sink, pending: [] };
      const chunkSamples = Math.round(audio.sampleRate * MIC_CHUNK_S); // samples in 0.5 s at the context rate
      processor.onaudioprocess = (event) => {
        mic.pending.push(new Float32Array(event.inputBuffer.getChannelData(0))); // copy: the browser reuses its buffer
        if (mic.pending.reduce((n, part) => n + part.length, 0) >= chunkSamples) flushMic(socket, mic);
      };
      source.connect(processor); // mic -> processor -> silent gain -> speakers
      processor.connect(sink);
      sink.connect(audio.destination);
      micRef.current = mic;
      micAnchorRef.current = true; // the first returned frame sets the drawing clock
      countersRef.current = { chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null };
      publish();
      sayAtRef.current = performance.now();
      audio.resume();
      setMicOn(true);
      setState("speaking");
    } catch (err) {
      setError(`Microphone unavailable: ${err.message}`); // e.g. permission denied or no device
    }
  };

  // Disconnect the audio graph, release the device, send what is left, then `audio_end`.
  const stopMic = () => {
    const mic = micRef.current;
    const socket = socketRef.current;
    if (!mic) return;
    mic.processor.disconnect();
    mic.source.disconnect();
    mic.sink.disconnect();
    mic.stream.getTracks().forEach((track) => track.stop()); // turns off the browser's recording indicator
    if (socket && socket.readyState === WebSocket.OPEN) {
      flushMic(socket, mic); // the last partial chunk
      socket.send(JSON.stringify({ type: "audio_end" }));
    }
    micRef.current = null;
    setMicOn(false);
  };

  const live = state === "ready" || state === "speaking"; // a session is open

  return (
    // data-* attributes expose state and counts for the E2E test.
    <div id="live-panel" data-state={state} data-chunks={stats.chunks} data-frames={stats.frames} data-playing={playing}>
      <h2 style={{ marginTop: "1.5rem" }}>Live Avatar</h2>
      <div style={card}>
        <label style={small}>
          Avatar face
          {/* Face, speaker and motion are fixed at `start`, so they lock while a session runs. */}
          <select id="live-avatar-select" style={field} value={avatarId} disabled={live || state === "connecting"}
            onChange={(e) => setAvatarId(e.target.value)}>
            {avatars.length === 0 && <option value="">no avatars registered</option>}
            {avatars.map((a) => (
              <option key={a.avatarId} value={a.avatarId} disabled={!a.usable}>{a.avatarId}{a.usable ? "" : " - blocked"}</option>
            ))}
          </select>
        </label>
        {/* Shown only when the voice list loaded. */}
        {voices.length > 0 && (
          <label style={{ ...small, display: "block", marginTop: "8px" }}>
            Speaker{face && recordedGender(face) ? ` (this face: ${recordedGender(face)})` : ""}
            {/* Remembered per face, as on the create screen. */}
            <select id="live-speaker-select" style={field} value={speaker} disabled={live || state === "connecting"}
              onChange={(e) => { setChosenSpeaker((c) => ({ ...c, [avatarId]: e.target.value })); rememberVoice(avatarId, e.target.value); }}>
              {voices.map((v) => <option key={v.id} value={v.id} disabled={!v.present}>{v.name} ({v.gender})</option>)}
            </select>
          </label>
        )}
        <label style={{ ...small, display: "block", marginTop: "8px" }}>
          Head movement
          <select id="live-motion-select" style={field} value={motion} disabled={live || state === "connecting"} onChange={(e) => setMotion(Number(e.target.value))}>
            <option value={0}>Still</option>
            <option value={0.6}>Subtle</option>
            <option value={1}>Natural</option>
            <option value={1.5}>Lively</option>
          </select>
        </label>
        {/* Session controls. */}
        <div style={{ display: "flex", gap: "8px", marginTop: "10px" }}>
          <button type="button" id="live-start" disabled={!avatarId || live || state === "connecting"} onClick={start}>
            {state === "connecting" ? "Connecting…" : "Start live session"}
          </button>
          <button type="button" id="live-stop" className="secondary" disabled={!live} onClick={stop}>Stop</button>
        </div>
        <div id="live-status" style={{ ...small, marginTop: "8px" }}>
          State: <strong>{state}</strong>
          {info ? ` · ${info.width}×${info.height} @ ${info.fps} fps · ${info.model}` : ""}
        </div>
        {error && <div className="alert error" id="live-error" style={{ marginTop: "8px" }}>{error}</div>}

        {/* ref={canvasRef} hands the DOM element to canvasRef.current. The draw loop paints frames here;
           the width and height attributes are reset from `ready`. */}
        <canvas ref={canvasRef} id="live-canvas" width={384} height={384}
          style={{ width: "100%", borderRadius: "8px", display: live ? "block" : "none", marginTop: "10px", background: "#020617" }} />

        <label style={{ ...small, display: "block", marginTop: "10px" }}>
          Say
          <textarea id="live-text" rows={2} style={field} value={text} disabled={!live} onChange={(e) => setText(e.target.value)} />
        </label>
        {/* Speak sends `say`; Interrupt stops the current speech and its queued audio. */}
        <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
          <button type="button" id="live-speak" disabled={state !== "ready" || !text.trim()} onClick={speak}>
            {state === "speaking" ? "Speaking…" : "Speak"}
          </button>
          <button type="button" id="live-interrupt" className="secondary" disabled={state !== "speaking"} onClick={interrupt}>Interrupt</button>
        </div>
        <div style={{ ...small, marginTop: "10px" }}>
          …or speak into your microphone. The mouth follows how loud you are: it is an estimate, not lip-read.
          <select id="live-mic-consent" style={field} value={micBasis} disabled={micOn} onChange={(e) => setMicBasis(e.target.value)}>
            <option value="">whose voice is it?</option>
            <option value="speaker-recorded">my own voice</option>
            <option value="written-consent">someone who consented in writing</option>
          </select>
        </div>
        {/* One button toggles the mic. While on it is always clickable; to start it needs a ready
           session and a consent choice. */}
        <button type="button" id="live-mic" style={{ marginTop: "8px" }}
          disabled={micOn ? false : state !== "ready" || !micBasis} onClick={micOn ? stopMic : startMic}>
          {micOn ? "Stop microphone" : "Speak with my microphone"}
        </button>
        {/* What arrived. The microphone run reports its drive ("audio-energy"); text runs report sentences. */}
        <div id="live-stats" style={{ ...small, marginTop: "8px" }}>
          audio chunks {stats.chunks} · frames {stats.frames}
          {stats.firstAudioMs !== null ? ` · first audio ${stats.firstAudioMs} ms` : ""}
          {stats.firstFrameMs !== null ? ` · first frame ${stats.firstFrameMs} ms` : ""}
          {stats.done && stats.done.drive ? ` · microphone: ${stats.done.chunks} chunks (${stats.done.drive})` : ""}
          {stats.done && !stats.done.drive ? ` · finished: ${stats.done.chunks} sentences in ${Math.round(stats.done.totalMs)} ms` : ""}
        </div>
      </div>
    </div>
  );
}
