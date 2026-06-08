"""Prompt2 pre-run copy for B3 gate/x asymmetric split planning.

Source Golden notebook: b3-nemotron-svd-26042701.ipynb
Experiment: B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT

This file is a copied patch target only. It was not executed during Prompt2.
Do not run it until Prompt3+ explicitly authorizes adapter generation / zip creation.
"""

import json
import os
import re
import glob
import shutil
import zipfile
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, UTC
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import kagglehub

# ============================================================
# B3_INPROJ_SPLIT_GATE_X_16_16 + clean submission.zip + SAFE RANK_MAP
# ============================================================

ADAPTER_PATH = os.environ.get(
    "ADAPTER_PATH",
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
)

MODEL_PATH = os.environ.get("MODEL_PATH") or kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKING_ADAPTER_DIR = Path(os.environ.get("WORKING_ADAPTER_DIR", str(EXPERIMENT_DIR / "adapter")))
DIAG_DIR = Path(os.environ.get("DIAG_DIR", str(EXPERIMENT_DIR / "diagnostics")))
SUBMISSION_ZIP = Path(os.environ.get("SUBMISSION_ZIP", str(EXPERIMENT_DIR / "submission.zip")))

# ============================================================
# Prompt3 static split constants for the copied Mamba merge block.
# Golden source used split_rank = 16 for both components.
# Prompt3 wires these constants into the copied Mamba merge block only,
# leaving Golden untouched.
# ============================================================
EXPERIMENT_NAME = "B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT"
GATE_COMPONENT_RANK = 15
X_COMPONENT_RANK = 17
IN_PROJ_TOTAL_RANK = 32
assert GATE_COMPONENT_RANK + X_COMPONENT_RANK == IN_PROJ_TOTAL_RANK

# === 【最重要】Kaggle制限(2.5GB)クリアのための黄金比 RANK_MAP ===
# === 再調整版 RANK_MAP (LB 0.85超えを狙う平準化設定) ===
RANK_MAP = {
    "o_proj": 32,
    "out_proj": 32,
    "k_proj": 32,
    "in_proj": 32,
    "down_proj": 22, # [変更] 16 -> 22: 削りすぎたため元に戻す（24に近い値へ）
    "up_proj": 22,   # [変更] 24 -> 22: down_projとのバランスを取る
    "q_proj": 32,    # テンソル数が少ないため上限32をキープ
    "v_proj": 32,    # 同上
    "gate_proj": 16,
    "x_proj": 16,
    "default": 24,
}

CANONICAL_TARGET_MODULES = [
    "k_proj",
    "o_proj",
    "in_proj",
    "q_proj",
    "up_proj",
    "v_proj",
    "down_proj",
    "out_proj",
    "lm_head",
]

BAD_PATTERNS = [
    ".experts.w1.",
    ".experts.w2.",
    ".experts.w3.",
    ".gate_proj.",
    ".x_proj.",
]

GOOD_PATTERNS = [
    ".in_proj.",
    ".out_proj.",
    ".up_proj.",
    ".down_proj.",
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".o_proj.",
    ".lm_head.",
]

def now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def get_target_rank(name: str) -> int:
    for key, rank in RANK_MAP.items():
        if key in name:
            return rank
    return RANK_MAP["default"]

def trained_adapter_key_rename(key: str) -> str:
    return key.replace("base_model.model.model", "base_model.model.backbone")

def rel_fro_error(
    original_B: torch.Tensor,
    original_A: torch.Tensor,
    new_B: torch.Tensor,
    new_A: torch.Tensor,
) -> dict:
    original_delta = original_B.float() @ original_A.float()
    new_delta = new_B.float() @ new_A.float()
    original_norm = torch.linalg.norm(original_delta).item()
    abs_error = torch.linalg.norm(original_delta - new_delta).item()
    rel_error = abs_error / max(original_norm, 1e-12)
    return {
        "original_delta_norm": original_norm,
        "abs_fro_error": abs_error,
        "rel_fro_error": rel_error,
    }

def append_compression_log(
    compression_log: list[dict],
    *,
    module: str,
    reason: str,
    before_rank: int,
    after_rank: int,
    original_B: torch.Tensor,
    original_A: torch.Tensor,
    new_B: torch.Tensor,
    new_A: torch.Tensor,
) -> None:
    err = rel_fro_error(original_B.float(), original_A.float(), new_B.float(), new_A.float())
    compression_log.append(
        {
            "module": module,
            "reason": reason,
            "before_rank": int(before_rank),
            "after_rank": int(after_rank),
            "original_shape_B": list(original_B.shape),
            "original_shape_A": list(original_A.shape),
            "compressed_shape_B": list(new_B.shape),
            "compressed_shape_A": list(new_A.shape),
            **err,
        }
    )

def compress_lora_fast(
    B: torch.Tensor,
    A: torch.Tensor,
    target_rank: int,
    *,
    module: str,
    reason: str,
    compression_log: list[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    before_rank = int(A.shape[0])
    if before_rank <= target_rank:
        return B.contiguous(), A.contiguous()

    Bf = B.float()
    Af = A.float()

    q_b, r_b = torch.linalg.qr(Bf)
    q_a, r_a = torch.linalg.qr(Af.T)
    core = r_b @ r_a.T
    u, s, vh = torch.linalg.svd(core, full_matrices=False)

    new_B = (q_b @ u[:, :target_rank]) * s[:target_rank].unsqueeze(0)
    new_A = vh[:target_rank, :] @ q_a.T

    append_compression_log(
        compression_log,
        module=module,
        reason=reason,
        before_rank=before_rank,
        after_rank=target_rank,
        original_B=Bf,
        original_A=Af,
        new_B=new_B,
        new_A=new_A,
    )

    return new_B.to(B.dtype).contiguous(), new_A.to(A.dtype).contiguous()

def load_model_key_shapes(model_path: str | Path) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for model_safetensors in glob.glob(str(Path(model_path) / "*.safetensors")):
        with safe_open(model_safetensors, framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    if not shapes:
        raise FileNotFoundError(f"No model safetensors found under {model_path}")
    return shapes

def load_adapter_tensors(adapter_path: str | Path) -> dict[str, torch.Tensor]:
    adapter_model_path = Path(adapter_path) / "adapter_model.safetensors"
    if not adapter_model_path.exists():
        raise FileNotFoundError(adapter_model_path)

    tensors = {}
    with safe_open(str(adapter_model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def find_mamba_merge_layers(base_names: set[str]) -> tuple[dict[str, dict[str, str]], set[str]]:
    layers = {}
    for base in base_names:
        for proj in ("gate_proj", "x_proj"):
            if f".{proj}" in base:
                layers.setdefault(base.rsplit(f".{proj}", 1)[0], {})[proj] = base

    incomplete = {
        layer: projs
        for layer, projs in layers.items()
        if set(projs) != {"gate_proj", "x_proj"}
    }
    if incomplete:
        raise ValueError(f"Incomplete gate/x mamba merge pairs: {sorted(incomplete.items())[:5]}")

    merge_bases = {base for projs in layers.values() for base in projs.values()}
    return layers, merge_bases

def infer_target_modules_from_tensors(tensors: dict[str, torch.Tensor]) -> list[str]:
    found = set()
    for key in tensors:
        if key.endswith(".lora_A.weight"):
            found.add(key[: -len(".lora_A.weight")].split(".")[-1])
    return [m for m in CANONICAL_TARGET_MODULES if m in found]

def write_adapter_config(adapter_dir: Path, target_modules: list[str]) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["target_modules"] = target_modules
    config["inference_mode"] = True
    config["lora_alpha"] = 32
    config["lora_dropout"] = 0

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return config

def summarize_compression_log(compression_log: list[dict]) -> dict:
    if not compression_log:
        return {
            "exists": True,
            "num_logged_compressions": 0,
            "max_rel_fro_error": None,
            "mean_rel_fro_error": None,
            "worst_modules": [],
        }

    rels = [float(row["rel_fro_error"]) for row in compression_log]
    worst = sorted(compression_log, key=lambda row: row["rel_fro_error"], reverse=True)[:20]

    return {
        "exists": True,
        "num_logged_compressions": len(compression_log),
        "max_rel_fro_error": max(rels),
        "mean_rel_fro_error": sum(rels) / len(rels),
        "worst_modules": worst,
    }

def build_b3_adapter() -> tuple[dict, dict]:
    print(f"ADAPTER_PATH: {ADAPTER_PATH}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"WORKING_ADAPTER_DIR: {WORKING_ADAPTER_DIR}")

    if WORKING_ADAPTER_DIR.exists():
        shutil.rmtree(WORKING_ADAPTER_DIR)

    WORKING_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ADAPTER_PATH, WORKING_ADAPTER_DIR, dirs_exist_ok=True)

    model_key_shapes = load_model_key_shapes(MODEL_PATH)
    adapter_tensors = load_adapter_tensors(ADAPTER_PATH)

    print(f"Loaded model key shapes: {len(model_key_shapes)}")
    print(f"Loaded raw adapter tensors: {len(adapter_tensors)}")

    base_names = {re.sub(r"\.lora_[AB]\.weight$", "", key) for key in adapter_tensors}
    mamba_merge_layers, mamba_merge_bases = find_mamba_merge_layers(base_names)

    print(f"Mamba merge layers: {len(mamba_merge_layers)}")

    output_tensors = {}
    compression_log = []

    for base in sorted(base_names):
        lora_A = adapter_tensors[f"{base}.lora_A.weight"]
        lora_B = adapter_tensors[f"{base}.lora_B.weight"]
        renamed = trained_adapter_key_rename(base)

        if ".experts.w3" in base and lora_A.numel() == 0:
            continue

        if base in mamba_merge_bases:
            continue

        if ".experts.w1" in base or ".experts.w2" in base:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

            num_experts = lora_A.shape[0]
            proj_name = "up_proj" if ".w1" in base else "down_proj"

            for expert_idx in range(num_experts):
                exp_renamed = re.sub(
                    r"\.experts\.w[12]",
                    f".experts.{expert_idx}.{proj_name}",
                    renamed,
                )

                new_B, new_A = compress_lora_fast(
                    lora_B[expert_idx].contiguous(),
                    lora_A[expert_idx].contiguous(),
                    get_target_rank(proj_name),
                    module=exp_renamed,
                    reason=f"expert_unfuse_{proj_name}_compression",
                    compression_log=compression_log,
                )

                output_tensors[f"{exp_renamed}.lora_A.weight"] = new_A
                output_tensors[f"{exp_renamed}.lora_B.weight"] = new_B

            continue

        new_B, new_A = compress_lora_fast(
            lora_B,
            lora_A,
            get_target_rank(renamed),
            module=renamed,
            reason="standard_projection_compression",
            compression_log=compression_log,
        )

        output_tensors[f"{renamed}.lora_A.weight"] = new_A
        output_tensors[f"{renamed}.lora_B.weight"] = new_B

    for layer_path, projs in sorted(mamba_merge_layers.items()):
        renamed_layer = trained_adapter_key_rename(layer_path)
        in_proj_base = f"{renamed_layer}.in_proj"

        model_in_proj_key = renamed_layer.replace("base_model.model.", "") + ".in_proj.weight"
        if model_in_proj_key not in model_key_shapes:
            raise KeyError(f"Missing model shape for {model_in_proj_key}")

        in_proj_dim = model_key_shapes[model_in_proj_key][0]

        gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()

        if gate_A.shape[0] != x_A.shape[0]:
            raise ValueError(f"Rank mismatch in {layer_path}: gate={gate_A.shape[0]} x={x_A.shape[0]}")
        if gate_A.shape[1] != x_A.shape[1]:
            raise ValueError(f"Input dim mismatch in {layer_path}: gate={gate_A.shape} x={x_A.shape}")

        raw_rank = gate_A.shape[0]
        gate_target_rank = GATE_COMPONENT_RANK
        x_target_rank = X_COMPONENT_RANK
        assert gate_target_rank + x_target_rank == IN_PROJ_TOTAL_RANK

        gate_B15, gate_A15 = compress_lora_fast(
            gate_B,
            gate_A,
            gate_target_rank,
            module=f"{in_proj_base}.gate_component",
            reason="B3_gate_component_rank32_to_15_before_in_proj_merge",
            compression_log=compression_log,
        )

        x_B17, x_A17 = compress_lora_fast(
            x_B,
            x_A,
            x_target_rank,
            module=f"{in_proj_base}.x_component",
            reason="B3_x_component_rank32_to_17_before_in_proj_merge",
            compression_log=compression_log,
        )

        gate_rank = int(gate_A15.shape[0])
        x_rank = int(x_A17.shape[0])
        assert gate_rank == GATE_COMPONENT_RANK
        assert x_rank == X_COMPONENT_RANK
        assert gate_rank + x_rank == IN_PROJ_TOTAL_RANK

        merged_A_raw = torch.cat([gate_A, x_A], dim=0)
        merged_B_raw = torch.zeros(in_proj_dim, 2 * raw_rank, dtype=gate_B.dtype)
        merged_B_raw[: gate_B.shape[0], :raw_rank] = gate_B
        merged_B_raw[gate_B.shape[0] : gate_B.shape[0] + x_B.shape[0], raw_rank:] = x_B

        merged_A = torch.cat([gate_A15, x_A17], dim=0)
        merged_B = torch.zeros(in_proj_dim, IN_PROJ_TOTAL_RANK, dtype=gate_B15.dtype)
        merged_B[: gate_B15.shape[0], :gate_rank] = gate_B15
        merged_B[
            gate_B15.shape[0] : gate_B15.shape[0] + x_B17.shape[0],
            gate_rank : gate_rank + x_rank,
        ] = x_B17

        assert merged_A.shape[0] == IN_PROJ_TOTAL_RANK
        assert merged_B.shape[1] == IN_PROJ_TOTAL_RANK

        append_compression_log(
            compression_log,
            module=in_proj_base,
            reason="B3_split_gate_x_15_17_block_merge_to_in_proj_rank32",
            before_rank=2 * raw_rank,
            after_rank=IN_PROJ_TOTAL_RANK,
            original_B=merged_B_raw,
            original_A=merged_A_raw,
            new_B=merged_B,
            new_A=merged_A,
        )

        output_tensors[f"{in_proj_base}.lora_A.weight"] = merged_A.to(gate_A.dtype).contiguous()
        output_tensors[f"{in_proj_base}.lora_B.weight"] = merged_B.to(gate_B.dtype).contiguous()

    adapter_model_path = WORKING_ADAPTER_DIR / "adapter_model.safetensors"
    save_file(output_tensors, str(adapter_model_path))

    target_modules = infer_target_modules_from_tensors(output_tensors)
    config = write_adapter_config(WORKING_ADAPTER_DIR, target_modules)

    compression_log_path = DIAG_DIR / "svd_compression_error.jsonl"
    with open(compression_log_path, "w", encoding="utf-8") as f:
        for row in compression_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_compression_log(compression_log)

    print("Weight surgery complete.")
    print(f"Output tensors: {len(output_tensors)}")
    print(f"Output adapter_model.safetensors: {adapter_model_path}")

    # サイズチェックと警告
    output_size_bytes = adapter_model_path.stat().st_size
    output_size_gb = output_size_bytes / (1024**3)
    print(f"Output size: {output_size_bytes} bytes ({output_size_gb:.2f} GB)")

    if output_size_gb > 2.5:
        print("\n" + "="*60)
        print("⚠️ WARNING: 出力ファイルが2.5GBを超過しています！")
        print("Kaggleの提出サイズ制限によりSubmission Errorになる可能性があります。")
        print("="*60 + "\n")
    else:
        print("\n✅ SUCCESS: ファイルサイズはKaggleの制限(2.5GB)以内に収まっています。")

    return config, summary

def validate_adapter(adapter_model_path: Path) -> dict:
    bad_counts = Counter()
    good_counts = Counter()
    suffix_counts = Counter()
    rank_violations = []
    zero_shape_count = 0
    max_rank_seen = 0
    contains_model_model = False
    contains_backbone = False
    num_tensors = 0

    with safe_open(str(adapter_model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            num_tensors += 1
            shape = tuple(f.get_slice(key).get_shape())

            if "base_model.model.model" in key:
                contains_model_model = True
            if "base_model.model.backbone" in key:
                contains_backbone = True
            if 0 in shape:
                zero_shape_count += 1

            for pattern in BAD_PATTERNS:
                if pattern in key:
                    bad_counts[pattern] += 1
            for pattern in GOOD_PATTERNS:
                if pattern in key:
                    good_counts[pattern] += 1

            if key.endswith(".lora_A.weight"):
                module = key[: -len(".lora_A.weight")].split(".")[-1]
                suffix_counts[module] += 1
                rank = int(shape[0])
                max_rank_seen = max(max_rank_seen, rank)
                if rank > 32:
                    rank_violations.append({"key": key, "rank": rank})

    with open(WORKING_ADAPTER_DIR / "adapter_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    target_modules = set(config.get("target_modules", []))
    emitted_modules = set(suffix_counts)
    missing_target_modules = sorted(target_modules - emitted_modules)

    return {
        "missing_target_modules": missing_target_modules,
        "rank_violations_rank_gt_32": rank_violations,
        "max_rank_seen": max_rank_seen,
        "num_tensors": num_tensors,
        "contains_base_model_model_model": contains_model_model,
        "contains_base_model_model_backbone": contains_backbone,
        "bad_pattern_counts": dict(bad_counts),
        "good_pattern_counts": dict(good_counts),
        "zero_shape_tensor_count": zero_shape_count,
        "suffix_counts": dict(suffix_counts),
    }

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def create_clean_submission_zip(adapter_dir: Path, output_zip: Path) -> dict:
    required = ["adapter_model.safetensors", "adapter_config.json"]

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    for name in required:
        path = adapter_dir / name
        if not path.exists():
            raise FileNotFoundError(path)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in required:
            zf.write(adapter_dir / name, arcname=name)

    entries = []
    with zipfile.ZipFile(output_zip, "r") as zf:
        for info in zf.infolist():
            entries.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": info.CRC,
                }
            )

    unexpected = sorted(set(row["name"] for row in entries) - set(required))

    return {
        "path": str(output_zip),
        "size_bytes": output_zip.stat().st_size,
        "sha256": sha256_file(output_zip),
        "expected_exact_entries_present": sorted(row["name"] for row in entries) == sorted(required),
        "unexpected_entries": unexpected,
        "entries": entries,
    }

def main() -> None:
    _, compression_summary = build_b3_adapter()

    adapter_model_path = WORKING_ADAPTER_DIR / "adapter_model.safetensors"
    adapter_check = validate_adapter(adapter_model_path)
    zip_check = create_clean_submission_zip(WORKING_ADAPTER_DIR, SUBMISSION_ZIP)

    summary = {
        "timestamp_utc": now_utc(),
        "experiment": "B3_INPROJ_SPLIT_GATE_X_16_16_FINAL_OPTIMIZED",
        "adapter_check": adapter_check,
        "zip_check": zip_check,
        "svd_compression_summary": compression_summary,
        "diagnostic_files": {
            "svd_error_log": str(DIAG_DIR / "svd_compression_error.jsonl"),
            "adapter_check": str(DIAG_DIR / "adapter_consistency_check.json"),
            "zip_check": str(DIAG_DIR / "submission_zip_check.json"),
            "summary": str(DIAG_DIR / "diagnostic_summary.json"),
        },
    }

    with open(DIAG_DIR / "adapter_consistency_check.json", "w", encoding="utf-8") as f:
        json.dump(adapter_check, f, ensure_ascii=False, indent=2)
    with open(DIAG_DIR / "submission_zip_check.json", "w", encoding="utf-8") as f:
        json.dump(zip_check, f, ensure_ascii=False, indent=2)
    with open(DIAG_DIR / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Serving adapter validation ===")
    print(json.dumps(adapter_check, ensure_ascii=False, indent=2))

    print("\n=== Clean submission.zip validation ===")
    print(json.dumps(zip_check, ensure_ascii=False, indent=2))

    print("\n=== Offline diagnostics summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("DONE")

if __name__ == "__main__":
    main()
