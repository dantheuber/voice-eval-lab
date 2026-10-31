"""A worker a fake server leaves behind. Writes its pid to a file, then sleeps."""

import signal
import sys
import time
from pathlib import Path

if "--ignore-term" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(__import__("os").getpid()))
time.sleep(60)
