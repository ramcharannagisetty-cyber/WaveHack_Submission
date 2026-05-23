#!/usr/bin/env python3
"""
PetriAI startup script.
Usage:  python3 start.py
Then open:  http://localhost:5050
"""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).parent
APP  = ROOT / "backend" / "app.py"

print("""
╔══════════════════════════════════════════╗
║          🧫  PetriAI v1.0               ║
║   AI-Powered Microbial Analysis          ║
╠══════════════════════════════════════════╣
║  Open browser →  http://localhost:5050  ║
║  Press  Ctrl+C  to stop the server      ║
╚══════════════════════════════════════════╝
""")

os.execv(sys.executable, [sys.executable, str(APP)])
