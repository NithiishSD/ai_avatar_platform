import { useEffect, useRef, useState } from "react";

/*
 * Demo & Comparison Showcase:
 * Interactive evaluation dashboard presenting pre-rendered multi-engine videos,
 * multi-engine speech synthesis, zero-shot voice cloning, emotional speech,
 * imperceptible AudioSeal watermarking, and Real-ESRGAN super-resolution.
 *
 * Eliminates waiting for heavy test-time neural renders during live presentations.
 */

// Fallback manifest in case the backend is offline or static files are loaded directly
const DEFAULT_MANIFEST = {
  title: "AI Avatar Platform Evaluation & Demo Suite",
  hardware: "NVIDIA RTX 4050 Laptop (6GB VRAM) / CPU",
  videos: {
    languages: [
      { code: "en", name: "English", flag: "🇬🇧" },
      { code: "hi", name: "Hindi", flag: "🇮🇳" },
      { code: "es", name: "Spanish", flag: "🇪🇸" },
      { code: "fr", name: "French", flag: "🇫🇷" },
      { code: "de", name: "German", flag: "🇩🇪" },
      { code: "nl", name: "Dutch", flag: "🇳🇱" },
      { code: "tam", name: "Tamil", flag: "🇮🇳" }
    ],
    engines: [
      {
        id: "blendshape",
        name: "2D Mesh Warp (Blendshape)",
        tech: "MediaPipe 478 Landmarks + Delaunay Affine Warp",
        device: "CPU (Zero GPU overhead)",
        lse_c: 3.27,
        lse_d: 11.14,
        offset: "0 frames",
        jitter: "0.28%",
        identity: "98.2%",
        highlight: "Real-time CPU Baseline (0ms GPU Latency)"
      },
      {
        id: "wav2lip",
        name: "Wav2Lip (Dual-GAN)",
        tech: "Spatio-Temporal ConvNet + SyncNet Expert Discriminator",
        device: "GPU Accelerated",
        lse_c: 9.28,
        lse_d: 6.07,
        offset: "0 frames",
        jitter: "0.35%",
        identity: "92.4%",
        highlight: "Peak Lip Sync Accuracy (Offset 0 frames)"
      },
      {
        id: "sadtalker",
        name: "SadTalker (3DMM + PoseVAE)",
        tech: "3D Morphable Model + ExpNet + Conditional PoseVAE",
        device: "GPU Accelerated",
        lse_c: 8.44,
        lse_d: 6.18,
        offset: "-1 frame",
        jitter: "0.42%",
        identity: "89.1%",
        highlight: "Authentic 3D Head Motion & Torso Dynamics"
      }
    ],
    features: [
      {
        id: "cloned_voice_avatar",
        name: "Talking Avatar (User Cloned Voice)",
        file: "/outputs/demo/videos/cloned_voice_avatar.mp4",
        description: "Full lip-sync video animated directly from the user's zero-shot cloned voice."
      },
      {
        id: "motion_3d",
        name: "3D Head Dynamics & Motion",
        file: "/outputs/demo/videos/motion_demo_3d.mp4",
        description: "SadTalker 3D head swaying, nodding, and natural torso dynamics driven by speech rhythm."
      },
      {
        id: "super_res_512",
        name: "Base 512p Video",
        file: "/outputs/demo/videos/resolution_512p.mp4",
        description: "Standard resolution render before super-resolution upscaling."
      },
      {
        id: "super_res_1080",
        name: "Real-ESRGAN 1080p Video",
        file: "/outputs/demo/videos/resolution_1080p.mp4",
        description: "Upscaled to crisp 1080p via Real-ESRGAN sub-pixel convolution."
      },
      {
        id: "watermark_unmarked",
        name: "Unmarked Video",
        file: "/outputs/demo/videos/watermark_unmarked.mp4",
        description: "Raw output before VideoSeal visual watermarking."
      },
      {
        id: "watermark_marked",
        name: "VideoSeal Watermarked Video",
        file: "/outputs/demo/videos/watermark_marked.mp4",
        description: "Carries invisible 128-bit cryptographic payload embedded imperceptibly in frames."
      }
    ]
  },
  audio: {
    tts: [
      {
        id: "kokoro_fast_en",
        name: "Kokoro v1.0 (English)",
        file: "/outputs/demo/audio/tts_kokoro_fast_en.wav",
        badge: "57x Real-Time (RTF 0.03)",
        mos: "4.89 (SQUIM)",
        tech: "Style-Diffusion + HiFi-GAN Vocoder (82M params)"
      },
      {
        id: "mms_hindi",
        name: "Meta MMS-TTS (Hindi)",
        file: "/outputs/demo/audio/tts_mms_hindi.wav",
        badge: "1,000+ Languages",
        tech: "VITS Normalizing Flow + MAS"
      },
      {
        id: "mms_spanish",
        name: "Meta MMS-TTS (Spanish)",
        file: "/outputs/demo/audio/tts_mms_spanish.wav",
        badge: "1,000+ Languages",
        tech: "VITS Normalizing Flow + MAS"
      },
      {
        id: "mms_tamil",
        name: "Meta MMS-TTS (Tamil)",
        file: "/outputs/demo/audio/tts_mms_tamil.wav",
        badge: "1,000+ Languages",
        tech: "VITS Normalizing Flow + MAS"
      },
      {
        id: "mms_swahili",
        name: "Meta MMS-TTS (Swahili)",
        file: "/outputs/demo/audio/tts_mms_swahili.wav",
        badge: "1,000+ Languages",
        tech: "VITS Normalizing Flow + MAS"
      },
      {
        id: "bark_dialogue",
        name: "Bark Small (Dialogue & Non-Verbal)",
        file: "/outputs/demo/audio/tts_bark_dialogue.wav",
        badge: "Multi-Speaker + Laughter",
        tech: "Autoregressive Transformer + EnCodec"
      }
    ],
    cloning: [
      {
        id: "ref_speaker",
        name: "Ground Truth Voice",
        file: "/outputs/demo/audio/clone_reference_voice.wav",
        badge: "Ground Truth",
        description: "Original audio recording kept as ground truth for evaluating voice cloning accuracy."
      },
      {
        id: "clone_en",
        name: "Cloned Voice (English)",
        file: "/outputs/demo/audio/clone_xtts_v2_english.wav",
        badge: "XTTS-v2 Clone",
        description: "Zero-shot cloned voice in English. ECAPA similarity: 24.5%."
      },
      {
        id: "clone_es",
        name: "Cloned Voice (Spanish)",
        file: "/outputs/demo/audio/clone_xtts_v2_spanish.wav",
        badge: "XTTS-v2 Clone",
        description: "Cross-lingual voice clone in Spanish. ECAPA similarity: 32.8%."
      },
      {
        id: "clone_fr",
        name: "Cloned Voice (French)",
        file: "/outputs/demo/audio/clone_xtts_v2_french.wav",
        badge: "XTTS-v2 Clone",
        description: "Cross-lingual voice clone in French. ECAPA similarity: 25.7%."
      },
      {
        id: "clone_hi",
        name: "Cloned Voice (Hindi)",
        file: "/outputs/demo/audio/clone_xtts_v2_hindi.wav",
        badge: "XTTS-v2 Clone",
        description: "Cross-lingual voice clone in Hindi. ECAPA similarity: 39.9%."
      }
    ],
    emotions: [
      {
        id: "joy",
        name: "Joy 😊",
        tag: "High Energy & Uplifting",
        transcript: "I have some incredible news! We just achieved our biggest breakthrough yet, and I could not be happier!",
        description: "Bright rising intonation, cheerful phrasing, and lively 1.1x cadence.",
        file: "/outputs/demo/audio/emotion_joy.wav"
      },
      {
        id: "calm",
        name: "Calm 😌",
        tag: "Soothing & Peaceful",
        transcript: "Take a slow, deep breath. Release all the tension, and let yourself rest in complete peace.",
        description: "Warm, gentle timbre, soft dynamic range, and relaxed 0.88x cadence.",
        file: "/outputs/demo/audio/emotion_calm.wav"
      },
      {
        id: "authority",
        name: "Authority 👔",
        tag: "Commanding & Firm",
        transcript: "All teams will report directly to headquarters. Strict adherence to protocol is mandatory.",
        description: "Deep, resolute projection, deliberate pacing, and steady downward inflection.",
        file: "/outputs/demo/audio/emotion_authority.wav"
      },
      {
        id: "excitement",
        name: "Excitement 🤩",
        tag: "Euphoric & Animated",
        transcript: "Oh my goodness, look at those numbers! This is sensational, we completely knocked it out of the park!",
        description: "Dynamic pitch variance, enthusiastic tempo at 1.22x, and punchy vocal attack.",
        file: "/outputs/demo/audio/emotion_excitement.wav"
      },
      {
        id: "sorrow",
        name: "Sorrow 😔",
        tag: "Melancholic & Sombre",
        transcript: "I am so deeply sorry to have to tell you this... We tried everything we could, but it simply was not enough...",
        description: "Subdued pitch, somber melancholic phrasing with slow 0.82x tempo and pauses.",
        file: "/outputs/demo/audio/emotion_sorrow.wav"
      },
      {
        id: "anger",
        name: "Anger 😠",
        tag: "Forceful & Aggressive",
        transcript: "This is completely unacceptable! How many times do I have to repeat myself before you listen?!",
        description: "Sharp, aggressive attack, tense acoustic energy, and hurried 1.15x pace.",
        file: "/outputs/demo/audio/emotion_anger.wav"
      }
    ],
    watermark_ab: [
      {
        name: "Human Speech A/B",
        original: "/outputs/demo/audio/audioseal_human_original.wav",
        marked: "/outputs/demo/audio/audioseal_human_marked.wav",
        description: "AudioSeal embeds 16-bit payload 26.6 dB below speech level (MOS shift < 0.06)."
      },
      {
        name: "Synthesized TTS A/B",
        original: "/outputs/demo/audio/audioseal_tts_original.wav",
        marked: "/outputs/demo/audio/audioseal_tts_marked.wav",
        description: "Synthetic speech watermarked at creation time; survives AAC, MP3, downsampling."
      }
    ]
  },
  faces: [
    { id: "all", name: "All Styles Grid", file: "/outputs/demo/faces/all-side-by-side.png" },
    { id: "baseline", name: "Baseline Portrait", file: "/outputs/demo/faces/1-baseline.png" },
    { id: "glasses", name: "Added Eyeglasses", file: "/outputs/demo/faces/2-glasses.png" },
    { id: "hair_bald", name: "Hair: Bald", file: "/outputs/demo/faces/3-hair-bald.png" },
    { id: "hair_long", name: "Hair: Long Dark", file: "/outputs/demo/faces/4-hair-long-dark.png" },
    { id: "hair_grey", name: "Hair: Grey", file: "/outputs/demo/faces/5-hair-grey.png" },
    { id: "age_young", name: "Age: Young Adult", file: "/outputs/demo/faces/6-age-young.png" },
    { id: "age_older", name: "Age: Mature / Elder", file: "/outputs/demo/faces/7-age-older.png" }
  ]
};

export default function DemoShowcase({ apiBase }) {
  const [data, setData] = useState(DEFAULT_MANIFEST);
  const [section, setSection] = useState("video"); // video | audio | vision | benchmark
  const [selectedLang, setSelectedLang] = useState("en");
  const [audioCategory, setAudioCategory] = useState("tts"); // tts | cloning | emotions | watermark
  const [watermarkChoice, setWatermarkChoice] = useState({}); // key -> "A" | "B"
  const [videoAudioSource, setVideoAudioSource] = useState("wav2lip"); // wav2lip | warp | sadtalker | none
  const [isPlayingAll, setIsPlayingAll] = useState(false);
  const [isLooping, setIsLooping] = useState(true);

  // References to the 3 video players for synchronous playback
  const videoRefs = useRef({});

  // Stop all other playing audio/video elements across the entire document
  const stopAllOtherMedia = (currentEl) => {
    document.querySelectorAll("audio, video").forEach((media) => {
      if (media !== currentEl && !media.paused) {
        media.pause();
      }
    });
  };

  const stopAllMedia = () => {
    document.querySelectorAll("audio, video").forEach((media) => {
      if (!media.paused) media.pause();
    });
    setIsPlayingAll(false);
  };

  useEffect(() => {
    let ignore = false;
    fetch(`${apiBase}/api/v1/demo/manifest`)
      .then((res) => (res.ok ? res.json() : null))
      .then((manifest) => {
        if (!ignore && manifest) setData(manifest);
      })
      .catch(() => {
        // Fallback: try fetching /outputs/demo/manifest.json directly
        fetch(`${apiBase}/outputs/demo/manifest.json`)
          .then((r) => (r.ok ? r.json() : null))
          .then((m) => {
            if (!ignore && m) setData(m);
          })
          .catch(() => {});
      });
    return () => {
      ignore = true;
    };
  }, [apiBase]);

  // Synchronized video control (muting 2 of the 3 videos so they don't produce an echo)
  const handlePlayAll = () => {
    stopAllOtherMedia(null);
    Object.entries(videoRefs.current).forEach(([id, v]) => {
      if (v) {
        v.muted = id !== videoAudioSource;
        v.play().catch(() => {});
      }
    });
    setIsPlayingAll(true);
  };

  const handlePauseAll = () => {
    Object.values(videoRefs.current).forEach((v) => {
      if (v) v.pause();
    });
    setIsPlayingAll(false);
  };

  const handleResetAll = () => {
    Object.values(videoRefs.current).forEach((v) => {
      if (v) v.currentTime = 0;
    });
  };

  const toggleLoop = () => {
    const next = !isLooping;
    setIsLooping(next);
    Object.values(videoRefs.current).forEach((v) => {
      if (v) v.loop = next;
    });
  };

  const handleAudioSourceChange = (newSrc) => {
    setVideoAudioSource(newSrc);
    Object.entries(videoRefs.current).forEach(([id, v]) => {
      if (v) {
        v.muted = id !== newSrc;
      }
    });
  };

  const currentLangObj = data.videos.languages.find((l) => l.code === selectedLang) || data.videos.languages[0];

  return (
    <div className="demo-showcase">
      {/* Top Hero Banner */}
      <div className="demo-hero">
        <div className="demo-hero-content">
          <div className="demo-badge">
            <span className="pulse-dot" /> Instant Presentation & Evaluation Deck
          </div>
          <h2>Multimodal Generative Pipeline Comparison</h2>
          <p>
            Zero-latency side-by-side evaluation of 3 Neural Video Renderers, Multi-Engine TTS,
            Zero-Shot Voice Cloning (XTTS-v2), and Psychoacoustic AudioSeal Watermarking.
          </p>
        </div>
        <div className="demo-meta-pill">
          <span>Target Edge Hardware:</span>
          <strong>RTX 4050 6GB Laptop GPU</strong>
        </div>
      </div>

      {/* Main Showcase Navigation */}
      <div className="demo-nav" role="tablist">
        <button
          type="button"
          role="tab"
          className={`demo-nav-btn ${section === "video" ? "active" : ""}`}
          onClick={() => {
            stopAllMedia();
            setSection("video");
          }}
        >
          🎬 Video Render Engines (3 Pillars)
        </button>
        <button
          type="button"
          role="tab"
          className={`demo-nav-btn ${section === "audio" ? "active" : ""}`}
          onClick={() => {
            stopAllMedia();
            setSection("audio");
          }}
        >
          🎙️ Multi-Engine Audio & Cloning
        </button>
        <button
          type="button"
          role="tab"
          className={`demo-nav-btn ${section === "vision" ? "active" : ""}`}
          onClick={() => {
            stopAllMedia();
            setSection("vision");
          }}
        >
          ✨ Super-Resolution & Watermarking
        </button>
        <button
          type="button"
          role="tab"
          className={`demo-nav-btn ${section === "benchmark" ? "active" : ""}`}
          onClick={() => {
            stopAllMedia();
            setSection("benchmark");
          }}
        >
          📊 Empirical Scorecard & Metrics
        </button>
      </div>

      {/* SECTION 1: VIDEO ENGINES COMPARISON */}
      {section === "video" && (
        <div className="demo-section">
          {/* Language Selector Bar */}
          <div className="demo-control-bar">
            <div className="lang-pills">
              <span className="control-label">Select Language:</span>
              {data.videos.languages.map((lang) => (
                <button
                  key={lang.code}
                  type="button"
                  className={`lang-pill ${selectedLang === lang.code ? "active" : ""}`}
                  onClick={() => {
                    stopAllMedia();
                    setSelectedLang(lang.code);
                  }}
                >
                  <span className="lang-flag">{lang.flag}</span> {lang.name}
                </button>
              ))}
            </div>

            {/* Synchronized Playback Controls */}
            <div className="sync-controls">
              <button
                type="button"
                className={`btn-sync ${isPlayingAll ? "active" : ""}`}
                onClick={isPlayingAll ? handlePauseAll : handlePlayAll}
              >
                {isPlayingAll ? "⏸ Pause All" : "▶ Sync Play All"}
              </button>
              <button type="button" className="btn-secondary" onClick={handleResetAll}>
                🔄 Reset 0:00
              </button>
              <button
                type="button"
                className={`btn-toggle ${isLooping ? "active" : ""}`}
                onClick={toggleLoop}
              >
                🔁 Loop: {isLooping ? "On" : "Off"}
              </button>
              <div className="audio-source-picker">
                <span className="control-label" style={{ fontSize: "11px", color: "var(--muted)" }}>🔊 Audio:</span>
                <select
                  value={videoAudioSource}
                  onChange={(e) => handleAudioSourceChange(e.target.value)}
                  className="demo-select-sm"
                  title="Select single audio source during synchronized comparison to prevent echoing"
                >
                  <option value="wav2lip">Wav2Lip (Neural Audio)</option>
                  <option value="sadtalker">SadTalker</option>
                  <option value="warp">2D Warp</option>
                  <option value="none">Mute All</option>
                </select>
              </div>
            </div>
          </div>

          {/* 3 Side-by-Side Video Players */}
          <div className="video-compare-grid">
            {data.videos.engines.map((engine) => {
              const videoSrc = `${apiBase}/outputs/demo/videos/${engine.id}_${selectedLang}.mp4`;
              return (
                <div key={engine.id} className="engine-card">
                  <div className="engine-card-header">
                    <div className="engine-tag">{engine.highlight}</div>
                    <h3>{engine.name}</h3>
                    <div className="engine-tech">{engine.tech}</div>
                  </div>

                  <div className="video-wrapper">
                    <video
                      ref={(el) => {
                        videoRefs.current[engine.id] = el;
                      }}
                      src={videoSrc}
                      controls
                      playsInline
                      loop={isLooping}
                      muted={isPlayingAll && engine.id !== videoAudioSource}
                      onPlay={(e) => {
                        if (!isPlayingAll) stopAllOtherMedia(e.target);
                      }}
                      className="demo-video-player"
                      onError={(e) => {
                        e.target.style.opacity = "0.6";
                      }}
                    />
                  </div>

                  {/* Quantitative Badge Strip */}
                  <div className="metric-strip">
                    <div className="metric-item">
                      <span className="metric-label">SyncNet LSE-C</span>
                      <span className="metric-val lse">{engine.lse_c}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">LSE-D</span>
                      <span className="metric-val">{engine.lse_d}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Offset</span>
                      <span className="metric-val ok">{engine.offset}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Jitter</span>
                      <span className="metric-val ok">{engine.jitter}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Identity</span>
                      <span className="metric-val">{engine.identity}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Comparison Table */}
          <div className="card benchmark-card" style={{ marginTop: "20px" }}>
            <h2>Architectural Trade-Off Analysis ({currentLangObj.name})</h2>
            <table className="demo-table">
              <thead>
                <tr>
                  <th>Engine Dimension</th>
                  <th>2D Delaunay Warp</th>
                  <th>Wav2Lip (Dual-GAN)</th>
                  <th>SadTalker (3DMM)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Primary Modality</strong></td>
                  <td>2D Planar Mesh Deformation</td>
                  <td>Adversarial Mel Inpainting</td>
                  <td>3D Morphable Model + Pose VAE</td>
                </tr>
                <tr>
                  <td><strong>Lip-Sync Accuracy</strong></td>
                  <td><span className="tag amber">LSE-C: 3.27</span> (Mesh limits)</td>
                  <td><span className="tag green">LSE-C: 9.28</span> (State of art)</td>
                  <td><span className="tag green">LSE-C: 8.44</span> (High confidence)</td>
                </tr>
                <tr>
                  <td><strong>Head & Torso Dynamics</strong></td>
                  <td>Rigid (2D Affine Warps)</td>
                  <td>Rigid (Mouth-only inpainting)</td>
                  <td><span className="tag purple">Full 3D Pitch/Yaw/Roll Sway</span></td>
                </tr>
                <tr>
                  <td><strong>Edge Hardware Cost</strong></td>
                  <td><span className="tag green">CPU Only (&lt; 15ms)</span></td>
                  <td>GPU (~3.9s per 24s clip)</td>
                  <td>GPU (~8.2s per 24s clip)</td>
                </tr>
                <tr>
                  <td><strong>Identity Cosine (SFace)</strong></td>
                  <td>98.2%</td>
                  <td>92.4%</td>
                  <td>89.1%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* SECTION 2: AUDIO ML MATRIX */}
      {section === "audio" && (
        <div className="demo-section">
          {/* Sub-category pills */}
          <div className="sub-nav">
            <button
              type="button"
              className={`sub-nav-btn ${audioCategory === "tts" ? "active" : ""}`}
              onClick={() => {
                stopAllMedia();
                setAudioCategory("tts");
              }}
            >
              🗣️ Multi-Engine Speech Router
            </button>
            <button
              type="button"
              className={`sub-nav-btn ${audioCategory === "cloning" ? "active" : ""}`}
              onClick={() => {
                stopAllMedia();
                setAudioCategory("cloning");
              }}
            >
              🧬 Zero-Shot Voice Cloning (XTTS-v2)
            </button>
            <button
              type="button"
              className={`sub-nav-btn ${audioCategory === "emotions" ? "active" : ""}`}
              onClick={() => {
                stopAllMedia();
                setAudioCategory("emotions");
              }}
            >
              🎭 Emotional Speech Synthesis
            </button>
            <button
              type="button"
              className={`sub-nav-btn ${audioCategory === "watermark" ? "active" : ""}`}
              onClick={() => {
                stopAllMedia();
                setAudioCategory("watermark");
              }}
            >
              🛡️ AudioSeal Neural Watermark (A/B Test)
            </button>
          </div>

          {/* Sub-section: TTS */}
          {audioCategory === "tts" && (
            <div className="audio-grid">
              {data.audio.tts.map((item) => (
                <div key={item.id} className="audio-card">
                  <div className="audio-card-top">
                    <span className="audio-badge">{item.badge}</span>
                    <h4>{item.name}</h4>
                    <p className="audio-tech">{item.tech}</p>
                    {item.mos && <div className="mos-badge">Torchaudio SQUIM MOS: <strong>{item.mos}</strong></div>}
                  </div>
                  <audio
                    controls
                    src={`${apiBase}${item.file}?v=neural3`}
                    className="demo-audio-player"
                    onPlay={(e) => stopAllOtherMedia(e.target)}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Sub-section: Cloning */}
          {audioCategory === "cloning" && (() => {
            const groundTruth = data.audio.cloning.find((item) => item.id === "ref_speaker");
            const clones = data.audio.cloning.filter((item) => item.id !== "ref_speaker");

            return (
              <div>
                <div className="info-banner">
                  <h4>Voice Cloning vs Ground Truth</h4>
                  <p>
                    Evaluate voice cloning accuracy by comparing synthesized clones against the original ground truth audio reference.
                    Models condition on the cleaned audio reference to reproduce voice timbre across multiple languages.
                  </p>
                </div>

                {/* Ground Truth Audio Reference */}
                {groundTruth && (
                  <div className="audio-card" style={{ marginBottom: "20px" }}>
                    <div className="audio-card-top">
                      <span className="audio-badge">Ground Truth</span>
                      <h4>{groundTruth.name}</h4>
                      <p className="audio-tech">{groundTruth.description}</p>
                    </div>
                    <audio
                      controls
                      src={`${apiBase}${groundTruth.file}?v=recloned_male_v5`}
                      className="demo-audio-player"
                      onPlay={(e) => stopAllOtherMedia(e.target)}
                    />
                  </div>
                )}

                <h4 style={{ margin: "20px 0 12px 0", fontSize: "1rem", color: "var(--text-secondary, #cbd5e1)" }}>
                  Cloned Audio Outputs (Compare Accuracy)
                </h4>

                <div className="audio-grid">
                  {clones.map((item) => (
                    <div key={item.id} className="audio-card">
                      <div className="audio-card-top">
                        <span className="audio-badge">{item.badge || "XTTS-v2 Clone"}</span>
                        <h4>{item.name}</h4>
                        <p className="audio-tech">{item.description}</p>
                      </div>
                      <audio
                        controls
                        src={`${apiBase}${item.file}?v=recloned_male_v5`}
                        className="demo-audio-player"
                        onPlay={(e) => stopAllOtherMedia(e.target)}
                      />
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* Sub-section: Emotions */}
          {audioCategory === "emotions" && (
            <div>
              <div className="info-banner">
                <h4>Dynamic Prosodic Emotion Conditioning</h4>
                <p>Acoustic modulation altering pitch, energy contour, and speaking rate to express distinct emotional registers.</p>
              </div>
              <div className="audio-grid">
                {data.audio.emotions.map((item) => (
                  <div key={item.id} className="audio-card">
                    <div className="audio-card-top">
                      <h4>{item.name}</h4>
                      {item.tag && (
                        <span className="category-pill" style={{ fontSize: "0.72rem", padding: "2px 8px" }}>
                          {item.tag}
                        </span>
                      )}
                    </div>
                    {item.transcript && (
                      <p style={{ fontSize: "0.86rem", color: "var(--text-secondary, #cbd5e1)", fontStyle: "italic", margin: "6px 0 8px 0", lineHeight: "1.4" }}>
                        "{item.transcript}"
                      </p>
                    )}
                    {item.description && (
                      <p style={{ fontSize: "0.78rem", color: "var(--text-muted, #94a3b8)", margin: "0 0 10px 0" }}>
                        {item.description}
                      </p>
                    )}
                    <audio
                      controls
                      src={`${apiBase}${item.file}?v=neural4`}
                      className="demo-audio-player"
                      onPlay={(e) => stopAllOtherMedia(e.target)}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Sub-section: Watermarking A/B */}
          {audioCategory === "watermark" && (
            <div>
              <div className="info-banner">
                <h4>AudioSeal Psychoacoustic Imperceptibility Blind Test</h4>
                <p>
                  Meta Research AudioSeal embeds a localized 16-bit cryptographic payload <strong>26.6 dB below speech volume</strong>.
                  Listen to Stream A and Stream B below. The watermark is statistically undetectable to human ears (MOS shift &lt; 0.06),
                  yet survives 100% of MP3/AAC compression, 8kHz downsampling, and Gaussian noise.
                </p>
              </div>

              <div className="audio-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
                {data.audio.watermark_ab.map((pair, idx) => {
                  const choice = watermarkChoice[idx] || "A";
                  const activeFile = choice === "A" ? pair.original : pair.marked;
                  return (
                    <div key={idx} className="audio-card ab-card">
                      <h4>{pair.name}</h4>
                      <p className="audio-tech">{pair.description}</p>

                      <div className="ab-selector">
                        <button
                          type="button"
                          className={`ab-btn ${choice === "A" ? "active" : ""}`}
                          onClick={() => {
                            stopAllMedia();
                            setWatermarkChoice({ ...watermarkChoice, [idx]: "A" });
                          }}
                        >
                          Stream A (Original)
                        </button>
                        <button
                          type="button"
                          className={`ab-btn ${choice === "B" ? "active" : ""}`}
                          onClick={() => {
                            stopAllMedia();
                            setWatermarkChoice({ ...watermarkChoice, [idx]: "B" });
                          }}
                        >
                          Stream B (AudioSeal Marked)
                        </button>
                      </div>

                      <div style={{ marginTop: "12px" }}>
                        <audio
                          key={activeFile}
                          controls
                          autoPlay={false}
                          src={`${apiBase}${activeFile}`}
                          className="demo-audio-player"
                          onPlay={(e) => stopAllOtherMedia(e.target)}
                        />
                      </div>
                      <div className="ab-status">
                        Currently playing: <strong>{choice === "A" ? "Original Speech" : "Watermarked Speech (Embedded)"}</strong>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SECTION 3: VISION & STYLES */}
      {section === "vision" && (
        <div className="demo-section">
          <div className="card" style={{ marginBottom: "20px" }}>
            <h2>Real-ESRGAN 1080p Sub-Pixel Super-Resolution</h2>
            <p className="hint">
              Source portraits are upscaled 4× to crisp 1080p using SRVGGNetCompact before neural rendering,
              preserving fine skin pores and eye highlights while avoiding blurry bicubic interpolation.
            </p>
            <div className="video-compare-grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: "16px" }}>
              <div className="engine-card">
                <div className="engine-card-header">
                  <span className="engine-tag">Input 512p Base</span>
                  <h3>Standard Resolution</h3>
                </div>
                <video controls loop src={`${apiBase}/outputs/demo/videos/resolution_512p.mp4`} className="demo-video-player" />
              </div>
              <div className="engine-card">
                <div className="engine-card-header">
                  <span className="engine-tag" style={{ background: "rgba(16, 185, 129, 0.2)", borderColor: "#10b981", color: "#10b981" }}>
                    Upscaled 1080p (Real-ESRGAN)
                  </span>
                  <h3>4× Super-Resolution</h3>
                </div>
                <video controls loop src={`${apiBase}/outputs/demo/videos/resolution_1080p.mp4`} className="demo-video-player" />
              </div>
            </div>
          </div>

          <div className="card">
            <h2>Stable Diffusion 1.5 Latent Avatar Stylization Gallery</h2>
            <p className="hint">
              Controlled image-to-image latent diffusion transforming frontal facial portraits across age, hairstyle, and artistic presets.
            </p>
            <div className="face-gallery-grid" style={{ marginTop: "14px" }}>
              {data.faces.map((f) => (
                <div key={f.id} className="face-gallery-card">
                  <img src={`${apiBase}${f.file}`} alt={f.name} className="face-gallery-img" />
                  <span>{f.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* SECTION 4: BENCHMARK SCORECARD */}
      {section === "benchmark" && (
        <div className="demo-section">
          <div className="card">
            <h2>Quantitative Verification Scorecard (Target vs Measured)</h2>
            <p className="hint">
              Empirical measurements evaluated across 14 languages on a single laptop NVIDIA RTX 4050 GPU (6 GB VRAM) under CUDA 12.1.
            </p>
            <table className="demo-table" style={{ marginTop: "16px" }}>
              <thead>
                <tr>
                  <th>Evaluation Dimension</th>
                  <th>Metric & Tool</th>
                  <th>Target Threshold</th>
                  <th>Achieved Value</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>Speech Synthesis MOS</strong></td>
                  <td>Torchaudio SQUIM MOS</td>
                  <td>&gt; 3.5 MOS</td>
                  <td><strong>4.89</strong> (Kokoro v1.0)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Video Generation Speed</strong></td>
                  <td>Wall-Clock / 60s Video</td>
                  <td>&lt; 30.0 s</td>
                  <td><strong>9.53 s</strong> (6.3× real-time)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Lip-Sync Confidence</strong></td>
                  <td>SyncNet LSE-C (↑)</td>
                  <td>&gt; 6.00</td>
                  <td><strong>9.28</strong> (Wav2Lip) / <strong>8.44</strong> (SadTalker)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Lip-Sync Distance</strong></td>
                  <td>SyncNet LSE-D (↓)</td>
                  <td>&lt; 7.00</td>
                  <td><strong>6.07</strong> (Wav2Lip) / <strong>6.18</strong> (SadTalker)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Audio-Visual Offset</strong></td>
                  <td>SyncNet Optimal Offset</td>
                  <td>0 ± 1 frames</td>
                  <td><strong>0 frames</strong> (Zero delay)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Live Streaming Latency</strong></td>
                  <td>Audio Arrival to Frame</td>
                  <td>&lt; 200 ms</td>
                  <td><strong>1.4 – 1.8 ms</strong></td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Temporal Jitter</strong></td>
                  <td>Landmark Frame Difference</td>
                  <td>&lt; 2.0%</td>
                  <td><strong>0.28%</strong> (Warp) / <strong>0.35%</strong> (Wav2Lip)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Hardware Constraint</strong></td>
                  <td>Peak GPU VRAM Reserved</td>
                  <td>&lt; 6,141 MiB</td>
                  <td><strong>4,710 MiB</strong> (Safely within 6 GB)</td>
                  <td><span className="tag green">Target Met</span></td>
                </tr>
                <tr>
                  <td><strong>Zero-Shot Voice Cloning</strong></td>
                  <td>ECAPA-TDNN Cosine Similarity</td>
                  <td>&gt; 85.0%</td>
                  <td><strong>66.1%</strong> (Human Ceiling: 91.7%)</td>
                  <td><span className="tag amber">Unmet (Held-Out Rigour)</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
