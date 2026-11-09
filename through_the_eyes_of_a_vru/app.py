"""Start the viewer: ``uv run app.py`` then open the printed address."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from through_the_eyes_of_a_vru.viewer.server import create_app

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "samples"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    app = create_app(args.root)
    url = f"http://{args.host}:{args.port}"
    print(f"Viewer running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
