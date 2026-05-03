import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_MODULES = ["k_proj", "o_proj", "in_proj", "q_proj", "up_proj", "v_proj", "down_proj", "out_proj", "lm_head"]
BAD_PATTERNS = [".experts.w1.", ".experts.w2.", ".experts.w3.", ".gate_proj.", ".x_proj."]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def load_tensors(path: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    out: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = tuple(f.get_slice(k).get_shape())
    return out


def module_from_key(key: str) -> str:
    for m in EXPECTED_MODULES + ["gate_proj", "x_proj"]:
        if f".{m}." in key:
            return m
    return "other"


def rank_from_pair(shape_a: tuple[int, ...], _shape_b: tuple[int, ...]) -> int | None:
    return shape_a[0] if len(shape_a) == 2 else None


def audit_adapter(adapter_dir: Path) -> dict:
    cpath = adapter_dir / "adapter_config.json"
    mpath = adapter_dir / "adapter_model.safetensors"
    if not cpath.exists() or not mpath.exists():
        raise FileNotFoundError(f"missing files in {adapter_dir}")

    cfg = json.loads(cpath.read_text())
    tensors = load_tensors(mpath)

    module_count = Counter()
    ranks = []
    zero = []
    unexpected = []
    has_backbone = False
    has_model_model = False
    keyset = set(tensors)

    for key, shape in tensors.items():
        has_backbone |= "base_model.model.backbone" in key
        has_model_model |= "base_model.model.model" in key
        if 0 in shape:
            zero.append({"key": key, "shape": list(shape)})
        if any(p in key for p in BAD_PATTERNS):
            unexpected.append(key)

        module_count[module_from_key(key)] += 1

        if key.endswith(".lora_A.weight"):
            bkey = key.replace(".lora_A.weight", ".lora_B.weight")
            if bkey in keyset:
                r = rank_from_pair(shape, tensors[bkey])
                ranks.append({"key": key, "rank": r})

    rank_dist = Counter([r["rank"] for r in ranks if r["rank"] is not None])
    max_rank = max(rank_dist) if rank_dist else None
    rank_violations = [r for r in ranks if r["rank"] and r["rank"] > 32]

    present_mod = {k.split(".")[-3] for k in tensors if k.endswith(".lora_A.weight")}
    missing = [m for m in EXPECTED_MODULES if m not in present_mod]

    return {
        "adapter_config": cfg,
        "target_modules": cfg.get("target_modules", []),
        "lora_alpha": cfg.get("lora_alpha"),
        "r": cfg.get("r"),
        "inference_mode": cfg.get("inference_mode"),
        "tensor_key_count": len(tensors),
        "module_tensor_count": dict(module_count),
        "rank_distribution": {str(k): v for k, v in sorted(rank_dist.items())},
        "max_rank": max_rank,
        "rank_violations": rank_violations,
        "zero_shape_tensors": zero,
        "unexpected_keys": unexpected,
        "missing_expected_modules": missing,
        "prefix_check": {"contains_backbone": has_backbone, "contains_model_model": has_model_model},
        "adapter_model_size_bytes": mpath.stat().st_size,
        "adapter_model_sha256": sha256_file(mpath),
    }


def zip_inventory(zpath: Path) -> dict:
    if not zpath.exists():
        return {"exists": False}
    with zipfile.ZipFile(zpath) as zf:
        entries = [i.filename for i in zf.infolist()]
    return {
        "exists": True,
        "path": str(zpath),
        "sha256": sha256_file(zpath),
        "entries": entries,
        "size_bytes": zpath.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Nemotron adapter conversion artifacts.")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"))
    parser.add_argument("--adapter-path", default=os.getenv("ADAPTER_PATH", "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20"))
    parser.add_argument("--baseline-submission-path", default=os.getenv("BASELINE_SUBMISSION_PATH", "/kaggle/working/submission.zip"))
    parser.add_argument("--outdir", default="reports/conversion_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    adapter_path = Path(args.adapter_path)
    baseline_submission = Path(args.baseline_submission_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": now(),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "baseline_submission_path": str(baseline_submission),
    }

    report["baseline_inventory"] = (
        audit_adapter(adapter_path) if adapter_path.exists() else {"error": "adapter path not found in this environment"}
    )
    report["zip_inventory"] = zip_inventory(baseline_submission)

    stage_names = [
        "raw adapter load",
        "prefix rename",
        "expert unfuse",
        "Mamba gate/x merge",
        "SVD compression",
        "target_modules reconciliation",
        "adapter_config rewrite",
        "final safetensors export",
        "submission.zip export",
    ]
    report["conversion_stage_log"] = [
        {"stage": s, "status": "not_executed_in_this_environment", "notes": "instrumentation scaffold only"}
        for s in stage_names
    ]
    report["svd_error_report"] = {
        "status": "not_computed",
        "reason": "requires conversion run with original and compressed factors",
    }

    (outdir / "baseline_adapter_inventory.json").write_text(json.dumps({"report": report}, indent=2, ensure_ascii=False))
    (outdir / "conversion_stage_log.json").write_text(json.dumps(report["conversion_stage_log"], indent=2, ensure_ascii=False))
    (outdir / "svd_error_report.json").write_text(json.dumps(report["svd_error_report"], indent=2, ensure_ascii=False))

    (outdir / "baseline_adapter_inventory.md").write_text(
        "# Baseline Adapter Inventory\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n"
    )
    (outdir / "conversion_stage_log.md").write_text(
        "# Conversion Stage Log\n\n"
        + "\n".join([f"- {s['stage']}: {s['status']} ({s['notes']})" for s in report["conversion_stage_log"]])
    )
    (outdir / "svd_error_report.md").write_text(
        "# SVD Error Report\n\n- status: not_computed\n- reason: requires conversion run with original and compressed factors\n"
    )


if __name__ == "__main__":
    main()
