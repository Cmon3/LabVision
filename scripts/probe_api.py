"""First probe of the Meta Model API with one image.

Usage: python scripts/probe_api.py path/to/image.jpg

Reads META_API_KEY, META_API_BASE_URL and META_MODEL from the environment
(or a local .env). The key is never printed or committed.

TODO: adapt the request body to the exact format in the Meta Model API
quickstart. The payload below is a placeholder, not the documented format.
"""
import base64
import os
import sys

import requests
from dotenv import load_dotenv


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    load_dotenv()
    key = os.environ.get("META_API_KEY")
    base = os.environ.get("META_API_BASE_URL")
    model = os.environ.get("META_MODEL")
    if not (key and base and model):
        print("Set META_API_KEY, META_API_BASE_URL and META_MODEL (see .env.example).")
        return 1

    with open(sys.argv[1], "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    # Placeholder payload: replace with the documented request format.
    payload = {"model": model, "image": image_b64, "prompt": "colonies"}
    r = requests.post(
        base,
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=120,
    )
    print("status:", r.status_code)
    print(r.text[:2000])
    return 0 if r.ok else 2


if __name__ == "__main__":
    sys.exit(main())
