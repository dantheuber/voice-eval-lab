"""A tiny model server for supervisor tests. Takes a port, and optional behavior flags.

--ready-after S    answer /health with 503 for S seconds first
--crash-marker P   exit with code 3 on the first start, if P doesn't exist yet
--pid-file P       write this process's pid to P
"""

import argparse
import http.server
import signal
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--ready-after", type=float, default=0.0)
    parser.add_argument("--crash-marker")
    parser.add_argument("--pid-file")
    arguments = parser.parse_args()

    if arguments.pid_file:
        import os

        with open(arguments.pid_file, "a") as file:
            file.write(f"{os.getpid()}\n")
    if arguments.crash_marker and not Path(arguments.crash_marker).exists():
        Path(arguments.crash_marker).touch()
        print("crashing on purpose", flush=True)
        sys.exit(3)

    ready_at = time.monotonic() + arguments.ready_after

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802  (http.server's name)
            ready = time.monotonic() >= ready_at
            self.send_response(200 if ready else 503)
            self.end_headers()

        def log_message(self, *args):
            pass

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print("listening", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
