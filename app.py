"""Read-only experiment report server, standard library only.
Usage: python app.py --run runs/<run-directory>
"""
import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

def make_page(run):
    reports = []
    for name in ["manifest.json","validation_metrics.json","outer_metrics.json","final_metrics.json"]:
        path = run/name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            reports.append("<h2>"+html.escape(name)+"</h2><pre>"+
                           html.escape(json.dumps(value,indent=2))+"</pre>")
    if not reports:
        raise ValueError("Choose a completed experiment directory containing JSON reports.")
    return ("<!doctype html><meta charset='utf-8'><title>FoG experiment</title>"
            "<h1>FoG research report</h1><p>Development results do not establish "
            "clinical effectiveness. See the experiment mode and policy status.</p>"+
            "".join(reports)).encode()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--port",type=int,default=8000)
    args=parser.parse_args()
    page=make_page(args.run)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)
    print(f"Report: http://127.0.0.1:{args.port}")
    HTTPServer(("127.0.0.1",args.port),Handler).serve_forever()

if __name__=="__main__":
    main()

