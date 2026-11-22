"""Inline exported runs into the twin viewer: web/twin_viewer_src.html -> web/twin_viewer.html.
    python web/build_viewer.py [run.json ...]      (default: every file in web/runs/)
The viewer also loads run files dropped onto it, so embedding is only a convenience."""
import json, sys
from pathlib import Path

W = Path(__file__).resolve().parent
files = [Path(f) for f in sys.argv[1:]] or sorted((W / "runs").glob("*.json"))
runs = [json.loads(f.read_text()) for f in files]
html = (W / "twin_viewer_src.html").read_text()
assert "/*RUNS*/[]" in html
html = html.replace("/*RUNS*/[]", json.dumps(runs, separators=(",", ":")))
out = W / "twin_viewer.html"
out.write_text(html)
print(out, round(len(html) / 1024), "KB,", len(runs), "runs:", ", ".join(f.name for f in files))
