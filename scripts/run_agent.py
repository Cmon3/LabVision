"""Run agent on one image: python scripts/run_agent.py IMAGE "request" """
from __future__ import annotations
import sys
from pathlib import Path

# load env from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from labvision.agent import run_agent

def main() -> int:
    if len(sys.argv) < 3:
        print('Usage: python scripts/run_agent.py IMAGE "request"')
        return 1
    img = sys.argv[1]
    req = sys.argv[2]
    out = run_agent(img, req)
    print(out["report"])
    print("\n--- structured ---")
    print(f"count={out['count']} qc={out['qc']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
