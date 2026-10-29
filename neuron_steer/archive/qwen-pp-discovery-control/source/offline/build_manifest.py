"""Hash this completed evidence bundle and check its local Markdown links."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "manifest.json"


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    checks = [read_json(p) for p in (
        "verification.json",
        "generation-audit/verification.json",
        "condolence-rescue/verification.json",
    )]
    assert all(check["passed"] for check in checks)
    assert read_json("text-audit.json")["passed"]
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path == DEST:
            continue
        if path.stat().st_size > 10_000_000:
            print("Hashing", path.relative_to(ROOT), flush=True)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"path": str(path.relative_to(ROOT)),
                      "bytes": path.stat().st_size,
                      "sha256": digest.hexdigest()})
    missing = []
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            target = target.strip("<>").split("#")[0]
            if not target or "://" in target:
                continue
            resolved = (path.parent / target).resolve()
            if resolved != DEST and not resolved.exists():
                missing.append({"file": str(path.relative_to(ROOT)), "target": target})
    completions = [check["completion"] for check in checks]
    launch = read_json("launch.json")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_forward_calls": sum(c["calls"] for c in completions),
        "smoke_forward_calls": launch["smoke_forward_calls"],
        "backward_calls": sum(c["backwards"] for c in completions),
        "model_run_seconds": sum(c["seconds"] for c in completions),
        "pid_correlated_gpu_samples": sum(c["inference_pid_gpu_samples"] for c in checks),
        "all_verifications_passed": True,
        "gpu_idle_after_completion": launch["gpu_idle_after_all_runs"],
        "local_markdown_links_passed": not missing,
        "missing_links": missing,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "exclusions": ["manifest.json itself", "__pycache__ directories"],
        "files": files,
    }
    DEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    assert not missing, missing


if __name__ == "__main__":
    main()
