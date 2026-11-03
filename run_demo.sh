#!/usr/bin/env bash
# ==============================================================================
# AI Avatar Platform - Quick Demo Launcher
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================================"
echo "  🎬 AI AVATAR PLATFORM - DEMO & EVALUATION SUITE"
echo "================================================================================"

PY="$DIR/backend/.conda/bin/python"
if [ ! -x "$PY" ]; then
    echo "❌ Error: Python not found at $PY"
    exit 1
fi

# 1. Health check or start backend
echo "🔍 Checking backend service on port 8000..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is already running on http://localhost:8000"
else
    echo "🚀 Starting backend server on http://localhost:8000..."
    cd "$DIR/backend"
    FRONTEND_DIST="$DIR/frontend/dist" DEMO_MODE=true PYTHONPATH=. "$PY" -m uvicorn app:app --port 8000 --host 0.0.0.0 &
    BACKEND_PID=$!
    cd "$DIR"
    
    # Wait for backend to be ready
    echo "⏳ Waiting for backend to respond..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Backend is UP and READY (PID $BACKEND_PID)"
            break
        fi
        sleep 1
    done
fi

# 2. Display Demo Access Points
echo ""
echo "================================================================================"
echo "  🌟 READY FOR PRESENTATION & DEMO!"
echo "================================================================================"
echo ""
echo "  🖥️  Evaluation & Demo Mode (3-Way Video/Audio Showcase):"
echo "      👉 http://localhost:8000/?demo=true"
echo ""
echo "  🚀 Actual Core Production Application (Create / Live / Verify):"
echo "      👉 http://localhost:8000"
echo "      (Or toggle the 'Demo Mode: ON/OFF' switch directly in the studio header!)"
echo ""
echo "  📑 Interactive Swagger API Docs:"
echo "      👉 http://localhost:8000/docs"
echo ""
echo "  📊 Available Demo Tabs:"
echo "      1. 'Demo & Compare':"
echo "         - 🎬 3 Video Engines side-by-side (2D Warp vs Wav2Lip vs SadTalker)"
echo "         - 🌐 Multi-language comparisons (English, Hindi, Spanish, French, etc.)"
echo "         - 🎙️ Multi-Engine TTS (Kokoro 57x, MMS-TTS 1,000+ languages, Bark)"
echo "         - 🧬 Zero-Shot Voice Cloning (XTTS-v2 held-out similarity)"
echo "         - 🛡️ AudioSeal Neural Watermark A/B imperceptibility test"
echo "         - ✨ Real-ESRGAN 1080p Super-Resolution and Style Variations"
echo "      2. 'Live Avatar':"
echo "         - 🚀 1-Click Live Avatar Demo (Zero-lag streaming avatar)"
echo "         - ⚡ 1.4 ms packet-to-frame real-time response"
echo "         - 🎭 1-Click speech presets & dynamic facial expressions"
echo ""
echo "================================================================================"
echo "Press Ctrl+C to stop servers when finished."

# Optional: start Vite dev server if running interactively with argument --dev
if [ "$1" == "--dev" ]; then
    echo "Starting frontend Vite dev server on http://localhost:5173..."
    cd "$DIR/frontend"
    npm run dev -- --port 5173
fi
