"""Resumable official downloads, checked against Google Storage MD5 metadata."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
]
PREFIX = "v1.0/connectome-data/flat-connectome/"

def digest(path):
    with path.open("rb") as f:
        return base64.b64encode(hashlib.file_digest(f, "md5").digest()).decode()

def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        key = PREFIX + name
        meta_url = "https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o/" + urllib.parse.quote(key, safe="")
        with urllib.request.urlopen(meta_url, timeout=60) as response:
            meta = json.load(response)
        url = "https://storage.googleapis.com/flyem-male-cns/" + key
        path = raw / name
        if path.exists():
            if path.stat().st_size != int(meta["size"]) or digest(path) != meta["md5Hash"]:
                raise RuntimeError(f"Existing file fails verification: {path}")
            print("Verified cached", name, flush=True)
        else:
            partial = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url], check=True)
            if partial.stat().st_size != int(meta["size"]) or digest(partial) != meta["md5Hash"]:
                raise RuntimeError(f"Downloaded file fails verification: {partial}")
            partial.rename(path)
        manifest[name] = {"url": url, "bytes": int(meta["size"]), "md5": meta["md5Hash"], "generation": meta["generation"]}
        (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("All three official files verified", flush=True)

if __name__ == "__main__":
    main()
