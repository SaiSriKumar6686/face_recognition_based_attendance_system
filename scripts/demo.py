"""
demo.py — Standalone Demo Launcher
════════════════════════════════════
One-command launcher for the Face Attendance System demo.

This script initialises the entire pipeline and starts the web server:
  1. Loads InsightFace buffalo_l (ResNet-50) recognition model
  2. Loads the FAISS gallery of enrolled students
  3. Starts the Flask web application on http://localhost:5000

Usage:
    python scripts/demo.py
    python scripts/demo.py --port 8080
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import log
from src.utils.db import init_db


def main():
    parser = argparse.ArgumentParser(description="Face Attendance System — Standalone Demo")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Face Guard AI — Attendance System Demo         ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Model:     InsightFace buffalo_l (ResNet-50)    ║")
    print("  ║  Detector:  RetinaFace (det_10g)                 ║")
    print("  ║  Matching:  FAISS IndexFlatIP (cosine)           ║")
    print("  ║  Pipeline:  CLAHE → Denoise → Embed → Match     ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # Pre-warm critical singletons
    log.info("Initialising database...")
    init_db()

    log.info("Loading recognition model (this may take a few seconds)...")
    from src.inference.embedder import get_embedder
    embedder = get_embedder()
    log.info(f"Embedder ready: {type(embedder).__name__}")

    log.info("Loading FAISS index...")
    from src.inference.matcher import get_matcher
    matcher = get_matcher()
    log.info(f"Gallery: {matcher.index.ntotal} enrolled embeddings")

    print()
    print(f"  ✓ System ready!")
    print(f"  ✓ Open your browser at: http://localhost:{args.port}/demo")
    print(f"  ✓ Press Ctrl+C to stop")
    print()

    # Start Flask
    from web.app import create_app
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
