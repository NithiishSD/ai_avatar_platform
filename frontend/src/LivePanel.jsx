import { useCallback, useEffect, useRef, useState } from "react";
import { recordedGender, rememberVoice, voiceForFace } from "./voices";

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
 * themself); the mouth follows loudness, an estimate the panel says out loud.
 */

const HEADER_BYTES = 13;
const KIND_AUDIO = 1;
const KIND_FRAME = 2;
// Scheduling lead: audio is booked this far ahead of "now" so the first chunk is
// not already late by the time it is decoded.
const LEAD_S = 0.15;
// Microphone audio goes to the server in chunks of this length; frames for a chunk come back after it.
const MIC_CHUNK_S = 0.5;

const card = { background: "#0f172a", border: "1px solid #334155", borderRadius: "8px", padding: "12px 14px", marginBottom: "12px" };
const small = { fontSize: "0.78rem", color: "#94a3b8" };
const field = { width: "100%", padding: "6px 8px", marginTop: "4px", background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155", borderRadius: "6px" };

function wsUrl(apiBase) {
  return `${apiBase.replace(/^http/, "ws")}/api/v1/live`;
}

export default function LivePanel({ apiBase }) {
  const [avatars, setAvatars] = useState([]);
  const [avatarId, setAvatarId] = useState("");
  const [state, setState] = useState("idle"); // idle | connecting | ready | speaking | closed
  const [error, setError] = useState("");
  const [text, setText] = useState("Hello! I am speaking live. Each sentence reaches you as soon as it is ready.");
  const [stats, setStats] = useState({ chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null, done: null });
  const [info, setInfo] = useState(null);
  // Speaker voice, matched to the face as on the create screen (voices.js).
  const [voices, setVoices] = useState([]);
  const [chosenSpeaker, setChosenSpeaker] = useState({});

  const socketRef = useRef(null);
  const canvasRef = useRef(null);
  const audioRef = useRef(null);       // AudioContext
  const anchorRef = useRef(0);         // audio-clock time at which timeline 0 plays
  const readyRef = useRef(null);       // the server's `ready` message
  const framesRef = useRef([]);        // decoded frames waiting for their time: {due, bitmap}
  const sayAtRef = useRef(0);
  const countersRef = useRef({ chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null });
  // Every audio chunk already handed to Web Audio and not yet finished. A chunk is scheduled ahead of
  // time, so stopping the server is not enough: Interrupt must also stop what is queued here, or the
  // voice keeps talking after the picture has stopped (the owner's M-05 finding).
  const sourcesRef = useRef(new Set());
  const [playing, setPlaying] = useState(0);
  const [micBasis, setMicBasis] = useState("");
  const [micOn, setMicOn] = useState(false);
  const micRef = useRef(null);          // {stream, source, processor, sink, pending: Float32Array[]}
  const micAnchorRef = useRef(false);   // true until the first mic frame sets the drawing clock

  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/api/v1/avatar/faces`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (ignore || !payload) return;
        setAvatars(payload.avatars);
        setAvatarId((current) => current || payload.avatars.find((a) => a.usable)?.avatarId || "");
      })
      .catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);

  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/api/v1/audio/voices`).then((r) => (r.ok ? r.json() : null)).then((d) => !ignore && d && setVoices(d.voices ?? [])).catch(() => {});
    return () => { ignore = true; };
  }, [apiBase]);
  const face = avatars.find((a) => a.avatarId === avatarId);
  const speaker = chosenSpeaker[avatarId] || voiceForFace(face, voices, "af_heart");

  const publish = useCallback(() => setStats({ ...countersRef.current, done: null }), []);

  const handleMedia = useCallback((buffer) => {
    const view = new DataView(buffer);
    const kind = view.getUint8(0);
    const presentationMs = view.getUint32(9);
    const payload = buffer.slice(HEADER_BYTES);
    const audio = audioRef.current;
    const ready = readyRef.current;
    if (!audio || !ready) return;
    const c = countersRef.current;

    if (kind === KIND_AUDIO) {
      // Re-anchor if the timeline's "now" has slipped into the past (the user waited
      // between two `say`s), so a late chunk is played, not skipped.
      if (anchorRef.current === 0 || anchorRef.current + presentationMs / 1000 < audio.currentTime) {
        anchorRef.current = audio.currentTime + LEAD_S - presentationMs / 1000;
      }
      const samples = new Int16Array(payload, 0, Math.floor(payload.byteLength / 2));
      const floats = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i += 1) floats[i] = samples[i] / 32768;
      const buf = audio.createBuffer(1, floats.length, ready.sampleRate);
      buf.copyToChannel(floats, 0);
      const source = audio.createBufferSource();
      source.buffer = buf;
      source.connect(audio.destination);
      sourcesRef.current.add(source);
      source.onended = () => {
        sourcesRef.current.delete(source);
        setPlaying(sourcesRef.current.size);
      };
      setPlaying(sourcesRef.current.size);
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
      const due = anchorRef.current + presentationMs / 1000;
      createImageBitmap(new Blob([payload], { type: "image/jpeg" })).then((bitmap) => {
        framesRef.current.push({ due, bitmap });
      });
      c.frames += 1;
      if (c.firstFrameMs === null) c.firstFrameMs = Math.round(performance.now() - sayAtRef.current);
    }
    publish();
  }, [publish]);

  /* Silence everything already scheduled and forget the timeline, so the next utterance starts fresh. */
  const silence = useCallback(() => {
    for (const source of sourcesRef.current) {
      try { source.stop(); } catch { /* never started or already ended */ }
    }
    sourcesRef.current.clear();
    setPlaying(0);
    framesRef.current = [];
    anchorRef.current = 0;
  }, []);

  const stop = useCallback(() => {
    silence();
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      try { socket.send(JSON.stringify({ type: "stop" })); } catch { /* already closing */ }
    }
    socket?.close();
    socketRef.current = null;
  }, [silence]);

  /* Draw whichever decoded frame is due on the audio clock; one loop for the panel's life. */
  useEffect(() => {
    let id = 0;
    function drawDue() {
      const audio = audioRef.current;
      const canvas = canvasRef.current;
      if (audio && canvas) {
        const queue = framesRef.current;
        let latest = null;
        while (queue.length && queue[0].due <= audio.currentTime) {
          if (latest) latest.bitmap.close();
          latest = queue.shift();
        }
        if (latest) {
          canvas.getContext("2d").drawImage(latest.bitmap, 0, 0, canvas.width, canvas.height);
          latest.bitmap.close();
        }
      }
      id = requestAnimationFrame(drawDue);
    }
    id = requestAnimationFrame(drawDue);
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => () => {
    micRef.current?.stream.getTracks().forEach((track) => track.stop());
    stop();
    audioRef.current?.close();
  }, [stop]);

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

    const socket = new WebSocket(wsUrl(apiBase));
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    socket.onopen = () => socket.send(JSON.stringify({ type: "start", avatarId, fps: 25, maxSide: 384, ...(voices.length ? { voice: speaker } : {}) }));
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") return handleMedia(event.data);
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        readyRef.current = message;
        setInfo(message);
        if (canvasRef.current) {
          canvasRef.current.width = message.width;
          canvasRef.current.height = message.height;
        }
        setState("ready");
      } else if (message.type === "done") {
        setStats((s) => ({ ...s, done: message }));
        setState("ready");
      } else if (message.type === "interrupted") {
        silence();
        setState("ready");
      } else if (message.type === "error") {
        setError(`${message.code}: ${message.detail}`);
        setState((s) => (s === "speaking" ? "ready" : s));
      }
    };
    socket.onerror = () => setError("Could not reach the live endpoint. Is the backend running?");
    socket.onclose = () => {
      setState((s) => (s === "idle" ? s : "closed"));
      socketRef.current = null;
    };
  };

  const speak = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    countersRef.current = { chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null };
    publish();
    sayAtRef.current = performance.now();
    audioRef.current?.resume();
    socket.send(JSON.stringify({ type: "say", text }));
    setState("speaking");
  };

  const interrupt = () => {
    socketRef.current?.send(JSON.stringify({ type: "interrupt" }));
    silence(); // at once, not when the server's reply arrives
  };

  /* Send the samples gathered so far as one PCM16 binary message. */
  const flushMic = (socket, mic) => {
    const total = mic.pending.reduce((n, part) => n + part.length, 0);
    if (!total) return;
    const pcm = new Int16Array(total);
    let offset = 0;
    for (const part of mic.pending) {
      for (let i = 0; i < part.length; i += 1) pcm[offset + i] = Math.max(-1, Math.min(1, part[i])) * 32767;
      offset += part.length;
    }
    mic.pending = [];
    socket.send(pcm.buffer);
  };

  const startMic = async () => {
    const socket = socketRef.current;
    const audio = audioRef.current;
    if (!socket || !audio) return;
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
      socket.send(JSON.stringify({ type: "audio_start", sampleRate: audio.sampleRate, consentBasis: micBasis }));
      const source = audio.createMediaStreamSource(stream);
      // ScriptProcessor is old but everywhere and enough for 0.5 s chunks; a silent gain keeps it
      // running without playing the user's own voice back to them.
      const processor = audio.createScriptProcessor(4096, 1, 1);
      const sink = audio.createGain();
      sink.gain.value = 0;
      const mic = { stream, source, processor, sink, pending: [] };
      const chunkSamples = Math.round(audio.sampleRate * MIC_CHUNK_S);
      processor.onaudioprocess = (event) => {
        mic.pending.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        if (mic.pending.reduce((n, part) => n + part.length, 0) >= chunkSamples) flushMic(socket, mic);
      };
      source.connect(processor);
      processor.connect(sink);
      sink.connect(audio.destination);
      micRef.current = mic;
      micAnchorRef.current = true;
      countersRef.current = { chunks: 0, frames: 0, audioBytes: 0, firstAudioMs: null, firstFrameMs: null };
      publish();
      sayAtRef.current = performance.now();
      audio.resume();
      setMicOn(true);
      setState("speaking");
    } catch (err) {
      setError(`Microphone unavailable: ${err.message}`);
    }
  };

  const stopMic = () => {
    const mic = micRef.current;
    const socket = socketRef.current;
    if (!mic) return;
    mic.processor.disconnect();
    mic.source.disconnect();
    mic.sink.disconnect();
    mic.stream.getTracks().forEach((track) => track.stop());
    if (socket && socket.readyState === WebSocket.OPEN) {
      flushMic(socket, mic);
      socket.send(JSON.stringify({ type: "audio_end" }));
    }
    micRef.current = null;
    setMicOn(false);
  };

  const live = state === "ready" || state === "speaking";

  return (
    <div id="live-panel" data-state={state} data-chunks={stats.chunks} data-frames={stats.frames} data-playing={playing}>
      <h2 style={{ marginTop: "1.5rem" }}>Live Avatar</h2>
      <div style={card}>
        <label style={small}>
          Avatar face
          <select id="live-avatar-select" style={field} value={avatarId} disabled={live || state === "connecting"}
            onChange={(e) => setAvatarId(e.target.value)}>
            {avatars.length === 0 && <option value="">no avatars registered</option>}
            {avatars.map((a) => (
              <option key={a.avatarId} value={a.avatarId} disabled={!a.usable}>{a.avatarId}{a.usable ? "" : " - blocked"}</option>
            ))}
          </select>
        </label>
        {voices.length > 0 && (
          <label style={{ ...small, display: "block", marginTop: "8px" }}>
            Speaker{face && recordedGender(face) ? ` (this face: ${recordedGender(face)})` : ""}
            <select id="live-speaker-select" style={field} value={speaker} disabled={live || state === "connecting"}
              onChange={(e) => { setChosenSpeaker((c) => ({ ...c, [avatarId]: e.target.value })); rememberVoice(avatarId, e.target.value); }}>
              {voices.map((v) => <option key={v.id} value={v.id} disabled={!v.present}>{v.name} ({v.gender})</option>)}
            </select>
          </label>
        )}
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

        <canvas ref={canvasRef} id="live-canvas" width={384} height={384}
          style={{ width: "100%", borderRadius: "8px", display: live ? "block" : "none", marginTop: "10px", background: "#020617" }} />

        <label style={{ ...small, display: "block", marginTop: "10px" }}>
          Say
          <textarea id="live-text" rows={2} style={field} value={text} disabled={!live} onChange={(e) => setText(e.target.value)} />
        </label>
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
        <button type="button" id="live-mic" style={{ marginTop: "8px" }}
          disabled={micOn ? false : state !== "ready" || !micBasis} onClick={micOn ? stopMic : startMic}>
          {micOn ? "Stop microphone" : "Speak with my microphone"}
        </button>
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
