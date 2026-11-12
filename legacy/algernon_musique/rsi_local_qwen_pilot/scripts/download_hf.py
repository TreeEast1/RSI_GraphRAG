#!/usr/bin/env python3
"""Download a public Hugging Face repo with curl.

The system huggingface_hub 0.29 client rejects the 2026 resolve-cache / xet
redirects. curl follows them. Existing files with the expected size are skipped.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def list_files(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "rsi-local-qwen-pilot"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    files = []
    for row in data:
        if row.get("type") not in (None, "file"):
            continue
        path = row.get("path") or row.get("rfilename")
        if not path or path.endswith("/"):
            continue
        files.append({"path": path, "size": row.get("size")})
    if not files:
        raise RuntimeError(f"no files listed for {repo}")
    return files


def fetch(repo: str, dest: str, item: dict) -> str:
    path = item["path"]
    size = item.get("size")
    out = os.path.join(dest, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.isfile(out) and size and os.path.getsize(out) == size:
        return f"skip {path}"
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    tmp = out + ".partial"
    cmd = [
        "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "2",
        "--retry-all-errors", "-C", "-", "-o", tmp, url,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        raise RuntimeError(f"curl failed for {path}: {tail}")
    os.replace(tmp, out)
    got = os.path.getsize(out)
    if size and got != size:
        raise RuntimeError(f"size mismatch {path}: got {got} expected {size}")
    return f"ok {path} {got}"


def main() -> None:
    repo = sys.argv[1]
    dest = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(dest, exist_ok=True)
    files = list_files(repo)
    total = sum(x.get("size") or 0 for x in files)
    print(f"repo={repo} files={len(files)} bytes={total}", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch, repo, dest, item): item["path"] for item in files}
        for fut in as_completed(futs):
            path = futs[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
                print(f"ERR {path}: {exc}", flush=True)
    manifest = {
        "repo": repo,
        "dest": dest,
        "n_files": len(files),
        "total_bytes": total,
        "files": files,
        "errors": errors,
    }
    with open(os.path.join(dest, "_download_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if errors:
        raise SystemExit(f"{len(errors)} download errors")
    print("DONE", repo, flush=True)


if __name__ == "__main__":
    main()
