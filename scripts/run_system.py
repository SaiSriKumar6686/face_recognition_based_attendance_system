"""
run_system.py
──────────────
Master launcher — starts all three background services:
  1.  LiveInference       — CCTV frame processing loop
  2.  RetrainTrigger      — continual learning scheduler
  3.  Flask web app       — admin dashboard + review queue

Usage
─────
    python scripts/run_system.py
    python scripts/run_system.py --source rtsp://192.168.1.100/stream1
    python scripts/run_system.py --source 0 --fps 1.0 --port 5000
"""

import argparse
import signal
import sys
import time

from src.utils.logger import log
from src.utils.db import init_db
from src.inference.live_inference import LiveInference
from src.continual_learning.retrain_trigger import RetrainTrigger

def main():
    parser = argparse.ArgumentParser(description="Face Attendance System — run all services")
    parser.add_argument("--source",  default=0,      help="CCTV source (int or RTSP URL)")
    parser.add_argument("--fps",     type=float, default=2.0, help="Max inference FPS")
    parser.add_argument("--port",    type=int,   default=5000, help="Flask web port")
    parser.add_argument("--no-web",  action="store_true",     help="Disable web server")
    parser.add_argument("--no-retrain", action="store_true",  help="Disable retrain trigger")
    args = parser.parse_args()

    # convert source to int if numeric
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    init_db()
    log.info("Database initialised.")

    # ── Start inference loop ─────────────────────────────────────────
    inference = LiveInference(source=source, fps_limit=args.fps)
    inference.start()

    # ── Start retrain trigger ────────────────────────────────────────
    retrain = None
    if not args.no_retrain:
        retrain = RetrainTrigger(poll_interval_sec=60)
        retrain.start()

    # ── Start web server ─────────────────────────────────────────────
    if not args.no_web:
        try:
            from web.app import create_app
            app = create_app()
            # run in a separate thread so we can handle SIGINT cleanly
            import threading
            web_thread = threading.Thread(
                target=lambda: app.run(host="0.0.0.0", port=args.port, debug=False),
                daemon=True,
            )
            web_thread.start()
            log.info(f"Web dashboard: http://localhost:{args.port}")
        except ImportError as e:
            log.warning(f"Web server could not start: {e}")

    # ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────
    def _shutdown(sig, frame):
        log.info("Shutting down…")
        inference.stop()
        if retrain:
            retrain.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("System running. Press Ctrl+C to stop.")
    while True:
        time.sleep(5)


if __name__ == "__main__":
    main()
