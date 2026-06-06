#!/usr/bin/env python3
"""Offline Phase3 analysis scaffold for NVIDIA Nemotron reasoning validation logs.

This script intentionally does **not** train, alter adapters, create submission.zip, or
submit to Kaggle.  It builds the requested Phase3 artifacts from local validation and
prediction/logprob files when they are available, and emits explicit NOT_AVAILABLE
placeholders when they are not.

The implementation is dependency-light (Python standard library only) so it can run in
Kaggle Internet OFF environments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CATEGORY_MAP_COLUMNS = [
    "problem_id", "category", "subcategory", "confidence", "rule",
    "matched_keywords", "manual_review_required",
]
VALIDATION_LABELED_EXTRA = ["question", "gold_answer"]
PRED_COLUMNS = [
    "problem_id", "category", "subcategory", "question", "gold_answer", "pred_answer",
    "raw_output", "reasoning_text", "final_answer_text", "is_correct", "parse_success",
    "parse_error_type", "generation_token_count", "finish_reason",
]
OVERALL_SUMMARY_COLUMNS = [
    "scope", "category", "subcategory", "n_total", "n_correct", "accuracy",
    "n_parse_success", "parse_success_rate", "avg_generation_token_count",
]
LOGPROB_COLUMNS = [
    "problem_id", "category", "subcategory", "is_correct", "parse_success", "min_logprob",
    "mean_logprob", "answer_min_logprob", "answer_mean_logprob", "low_conf_token_count",
    "lowest_logprob_token", "lowest_logprob_context", "answer_low_conf_token_count",
]
CATEGORY_FAILURE_COLUMNS = [
    "category", "subcategory", "n", "correct", "accuracy", "avg_min_logprob",
    "avg_answer_min_logprob", "n_wrong_low_conf", "n_wrong_high_conf",
    "n_correct_low_conf", "priority_score", "priority_reason",
]
FAILURE_TYPE_COLUMNS = ["category", "failure_type", "n", "share", "priority_mean"]
CRYPT_COLUMNS = [
    "problem_id", "subcategory", "question", "gold_answer", "pred_answer", "is_correct",
    "min_logprob", "answer_min_logprob", "failure_type", "failure_reason",
    "solver_check_possible", "synthetic_generation_possible", "recommended_template",
    "example_priority",
]
BIT_NUM_COLUMNS = [
    "problem_id", "category", "subcategory", "question", "gold_answer", "pred_answer",
    "is_correct", "min_logprob", "answer_min_logprob", "failure_type", "failure_reason",
    "solver_check_possible", "synthetic_generation_possible", "recommended_template",
    "example_priority",
]

CRYPT_KEYWORDS = {
    "alphametic_addition": ["alphametic", "cryptarithm", "send", "more", "money", "+", "sum"],
    "alphametic_subtraction": ["alphametic", "cryptarithm", "subtract", "minus", "-"],
    "alphametic_multiplication": ["alphametic", "cryptarithm", "multiply", "product", "×", "*"],
    "digit_assignment": ["digit", "letter", "assign", "mapping", "different digits"],
    "carry_reasoning": ["carry", "column", "addition"],
    "leading_zero_constraint": ["leading zero", "first digit", "cannot be zero", "nonzero"],
}
BIT_KEYWORDS = {
    "xor": ["xor", "exclusive or", "^"],
    "and": ["bitwise and", "&"],
    "or": ["bitwise or", "|"],
    "shift_left": ["left shift", "<<", "shift left"],
    "shift_right": ["right shift", ">>", "shift right"],
    "mask": ["mask", "bitmask", "masked"],
    "signed_unsigned": ["signed", "unsigned", "two's complement", "2's complement"],
    "binary_arithmetic": ["binary arithmetic", "binary sum", "binary addition"],
}
NUMERAL_KEYWORDS = {
    "binary_to_decimal": ["binary to decimal", "base 2 to decimal", "convert 0b", "in decimal"],
    "decimal_to_binary": ["decimal to binary", "base 10 to binary", "in binary"],
    "hex_to_decimal": ["hex to decimal", "hexadecimal to decimal", "0x", "in decimal"],
    "decimal_to_hex": ["decimal to hex", "decimal to hexadecimal", "in hexadecimal", "in hex"],
    "base_n_conversion": ["base", "radix", "convert", "base-"],
    "roman_numeral": ["roman numeral", "roman"],
}
OTHER_CATEGORY_KEYWORDS = [
    ("cipher", "cipher", ["cipher", "encrypt", "decrypt", "caesar", "vigenere", "rot13"]),
    ("equation", "equation_solving", ["equation", "solve for", "linear", "quadratic"]),
    ("unit_conversion", "unit_conversion", ["convert", "meters", "kilograms", "miles", "fahrenheit", "celsius"]),
    ("rule_induction", "sequence_rule", ["sequence", "pattern", "next term", "rule"]),
    ("logic", "deduction", ["true", "false", "logic", "knights", "liars"]),
    ("arithmetic", "arithmetic", ["calculate", "compute", "sum", "product", "difference", "quotient"]),
]


def norm_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "correct"}:
        return True
    if s in {"false", "0", "no", "n", "wrong", "incorrect"}:
        return False
    return None


def read_table(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else obj.get("data", [])
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_split_key(row: Dict[str, Any], idx: int) -> str:
    """Return a stable semantic key for deterministic train.csv splitting."""
    for key in ("problem_id", "id", "question_id", "row_id", "uuid"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    question = detect_question(row)
    if question:
        return question
    return json.dumps(row, ensure_ascii=False, sort_keys=True) or f"row_{idx}"


def create_deterministic_validation_split(
    *,
    train_csv: Path,
    validation_output: Path,
    remainder_output: Optional[Path] = None,
    seed: str = "42",
    validation_fraction: float = 0.2,
    validation_size: Optional[int] = None,
    manifest_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a deterministic held-out validation CSV from Kaggle train.csv.

    The split is hash-based and does not depend on input row order for rows with
    stable IDs/questions.  All original columns are preserved.
    """
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    n_total = len(rows)
    if n_total == 0:
        n_valid = 0
    elif validation_size is not None:
        n_valid = max(1, min(n_total, int(validation_size)))
    else:
        frac = max(0.0, min(1.0, float(validation_fraction)))
        n_valid = max(1, min(n_total, int(round(n_total * frac))))
    ranked: List[Tuple[str, int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        key = stable_split_key(row, idx)
        digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest()
        ranked.append((digest, idx, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    valid_idx = {idx for _, idx, _ in ranked[:n_valid]}
    valid_rows = [row for idx, row in enumerate(rows) if idx in valid_idx]
    remainder_rows = [row for idx, row in enumerate(rows) if idx not in valid_idx]

    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with validation_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid_rows)
    if remainder_output:
        remainder_output.parent.mkdir(parents=True, exist_ok=True)
        with remainder_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(remainder_rows)

    manifest = {
        "source_train_csv": str(train_csv),
        "validation_output": str(validation_output),
        "remainder_output": str(remainder_output) if remainder_output else "",
        "seed": str(seed),
        "validation_fraction": validation_fraction,
        "validation_size": validation_size if validation_size is not None else "",
        "n_total": n_total,
        "n_validation": len(valid_rows),
        "n_remainder": len(remainder_rows),
        "method": "sha256(seed + stable_row_id_or_question), sorted ascending",
        "columns_preserved": fieldnames,
    }
    manifest_path = manifest_output or validation_output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def detect_id(row: Dict[str, Any], idx: int) -> str:
    for key in ("problem_id", "id", "question_id", "row_id", "uuid"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"validation_{idx:05d}"


def detect_question(row: Dict[str, Any]) -> str:
    for key in ("question", "prompt", "problem", "input", "text"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def detect_answer(row: Dict[str, Any]) -> str:
    for key in ("gold_answer", "answer", "target", "label", "expected", "output"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def keyword_hits(text: str, mapping: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    low = text.lower()
    best_sub = "unknown"
    best_hits: List[str] = []
    for sub, kws in mapping.items():
        hits = [kw for kw in kws if kw.lower() in low]
        if len(hits) > len(best_hits):
            best_sub, best_hits = sub, hits
    return best_sub, best_hits


def classify_question(question: str) -> Dict[str, Any]:
    q = question or ""
    low = q.lower()
    sub, hits = keyword_hits(q, CRYPT_KEYWORDS)
    crypt_signal = any(x in low for x in ["cryptarithm", "alphametic", "letter", "different digits", "distinct digits"])
    equation_like = bool(re.search(r"[A-Z]{2,}\s*[+\-*×]\s*[A-Z]{2,}\s*=\s*[A-Z]{2,}", q))
    if crypt_signal or equation_like:
        if sub == "unknown":
            sub = "digit_assignment"
        return {"category": "cryptarithm", "subcategory": sub, "confidence": 0.88 if hits else 0.72,
                "rule": "cryptarithm_keyword_or_letter_equation", "matched_keywords": "|".join(hits),
                "manual_review_required": False if hits or equation_like else True}

    sub, hits = keyword_hits(q, BIT_KEYWORDS)
    if hits or any(tok in q for tok in ["<<", ">>", "^", "&", "|"]):
        return {"category": "bit_manipulation", "subcategory": sub, "confidence": 0.9 if hits else 0.75,
                "rule": "bit_keyword_operator", "matched_keywords": "|".join(hits),
                "manual_review_required": False}

    sub, hits = keyword_hits(q, NUMERAL_KEYWORDS)
    base_pattern = bool(re.search(r"\bbase\s*-?\s*\d+\b|\b0x[0-9a-f]+\b|\b0b[01]+\b", low))
    if hits or base_pattern:
        return {"category": "numeral_conversion", "subcategory": sub, "confidence": 0.86 if hits else 0.7,
                "rule": "numeral_conversion_keyword_or_base_literal", "matched_keywords": "|".join(hits),
                "manual_review_required": False if hits else True}

    for cat, subcat, kws in OTHER_CATEGORY_KEYWORDS:
        hits = [kw for kw in kws if kw in low]
        if hits:
            return {"category": cat, "subcategory": subcat, "confidence": 0.62,
                    "rule": f"{cat}_keyword", "matched_keywords": "|".join(hits),
                    "manual_review_required": cat in {"arithmetic", "unit_conversion"}}

    return {"category": "other", "subcategory": "unknown", "confidence": 0.25,
            "rule": "no_rule_matched", "matched_keywords": "", "manual_review_required": True}


def build_category_rows(validation_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cat_rows, labeled = [], []
    for idx, row in enumerate(validation_rows):
        pid = detect_id(row, idx)
        question = detect_question(row)
        gold = detect_answer(row)
        c = classify_question(question)
        cat = {"problem_id": pid, **c}
        lab = {**cat, "question": question, "gold_answer": gold}
        cat_rows.append(cat)
        labeled.append(lab)
    return cat_rows, labeled


def normalize_prediction_rows(pred_rows: List[Dict[str, Any]], cat_by_id: Dict[str, Dict[str, Any]], val_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(pred_rows):
        pid = detect_id(row, idx)
        val = val_by_id.get(pid, {})
        cat = cat_by_id.get(pid, {})
        raw = str(row.get("raw_output", row.get("output", row.get("generation", ""))))
        pred = str(row.get("pred_answer", row.get("prediction", row.get("answer_pred", ""))))
        final_text = str(row.get("final_answer_text", pred or raw[-200:]))
        gold = str(row.get("gold_answer", val.get("gold_answer", detect_answer(row))))
        is_correct = norm_bool(row.get("is_correct"))
        if is_correct is None and pred and gold:
            is_correct = pred.strip() == gold.strip()
        parse_success = norm_bool(row.get("parse_success"))
        if parse_success is None:
            parse_success = bool(pred or final_text)
        out.append({
            "problem_id": pid,
            "category": row.get("category", cat.get("category", "other")),
            "subcategory": row.get("subcategory", cat.get("subcategory", "unknown")),
            "question": row.get("question", val.get("question", "")),
            "gold_answer": gold,
            "pred_answer": pred,
            "raw_output": raw,
            "reasoning_text": row.get("reasoning_text", raw),
            "final_answer_text": final_text,
            "is_correct": "" if is_correct is None else is_correct,
            "parse_success": "" if parse_success is None else parse_success,
            "parse_error_type": row.get("parse_error_type", "" if parse_success else "missing_final_answer"),
            "generation_token_count": row.get("generation_token_count", row.get("token_count", len(raw.split()) if raw else "")),
            "finish_reason": row.get("finish_reason", "unknown" if raw else "not_available"),
        })
    return out


def avg(vals: List[float]) -> str:
    vals = [v for v in vals if not math.isnan(v)]
    return "" if not vals else f"{sum(vals)/len(vals):.6f}"


def to_float(v: Any) -> float:
    try:
        if v in (None, ""):
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def summarize_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return [{"scope": "overall", "category": "ALL", "subcategory": "ALL", "n_total": 0, "n_correct": 0,
                 "accuracy": "NOT_AVAILABLE", "n_parse_success": 0, "parse_success_rate": "NOT_AVAILABLE",
                 "avg_generation_token_count": ""}]
    groups = {("overall", "ALL", "ALL"): rows}
    for r in rows:
        groups.setdefault(("category", r.get("category", "other"), r.get("subcategory", "unknown")), []).append(r)
    out = []
    for (scope, cat, sub), grp in groups.items():
        n = len(grp)
        correct = sum(1 for r in grp if norm_bool(r.get("is_correct")) is True)
        parse = sum(1 for r in grp if norm_bool(r.get("parse_success")) is True)
        toks = [to_float(r.get("generation_token_count")) for r in grp]
        out.append({"scope": scope, "category": cat, "subcategory": sub, "n_total": n, "n_correct": correct,
                    "accuracy": f"{correct/n:.6f}" if n else "NOT_AVAILABLE", "n_parse_success": parse,
                    "parse_success_rate": f"{parse/n:.6f}" if n else "NOT_AVAILABLE",
                    "avg_generation_token_count": avg(toks)})
    return out


def normalize_logprob_rows(log_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if log_rows:
        pred_by_id = {r["problem_id"]: r for r in pred_rows}
        out = []
        for idx, row in enumerate(log_rows):
            pid = detect_id(row, idx)
            p = pred_by_id.get(pid, {})
            out.append({c: row.get(c, p.get(c, "")) for c in LOGPROB_COLUMNS})
        return out
    out = []
    for r in pred_rows:
        out.append({
            "problem_id": r.get("problem_id", ""), "category": r.get("category", ""),
            "subcategory": r.get("subcategory", ""), "is_correct": r.get("is_correct", ""),
            "parse_success": r.get("parse_success", ""), "min_logprob": "", "mean_logprob": "",
            "answer_min_logprob": "", "answer_mean_logprob": "", "low_conf_token_count": "",
            "lowest_logprob_token": "", "lowest_logprob_context": "", "answer_low_conf_token_count": "",
        })
    return out


def build_category_failure(log_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in log_rows:
        groups[(r.get("category", "other"), r.get("subcategory", "unknown"))].append(r)
    out = []
    for (cat, sub), grp in groups.items():
        n = len(grp)
        correct = sum(1 for r in grp if norm_bool(r.get("is_correct")) is True)
        mins = [to_float(r.get("min_logprob")) for r in grp]
        ansmins = [to_float(r.get("answer_min_logprob")) for r in grp]
        low_threshold = -5.0
        high_threshold = -1.0
        wrong_low = sum(1 for r in grp if norm_bool(r.get("is_correct")) is False and to_float(r.get("answer_min_logprob") or r.get("min_logprob")) <= low_threshold)
        wrong_high = sum(1 for r in grp if norm_bool(r.get("is_correct")) is False and to_float(r.get("answer_min_logprob") or r.get("min_logprob")) >= high_threshold)
        correct_low = sum(1 for r in grp if norm_bool(r.get("is_correct")) is True and to_float(r.get("answer_min_logprob") or r.get("min_logprob")) <= low_threshold)
        accuracy = correct / n if n else 0.0
        score = 1 + int(n >= 10) + int(accuracy < 0.75) + int(wrong_low > 0) + int(cat in {"cryptarithm", "bit_manipulation", "numeral_conversion"})
        out.append({"category": cat, "subcategory": sub, "n": n, "correct": correct,
                    "accuracy": f"{accuracy:.6f}", "avg_min_logprob": avg(mins),
                    "avg_answer_min_logprob": avg(ansmins), "n_wrong_low_conf": wrong_low,
                    "n_wrong_high_conf": wrong_high, "n_correct_low_conf": correct_low,
                    "priority_score": min(5, max(1, score)),
                    "priority_reason": "high if count/low accuracy/low confidence/solver-friendly signals are present"})
    if not out:
        out.append({
            "category": "ALL",
            "subcategory": "NOT_AVAILABLE",
            "n": 0,
            "correct": 0,
            "accuracy": "NOT_AVAILABLE",
            "avg_min_logprob": "",
            "avg_answer_min_logprob": "",
            "n_wrong_low_conf": 0,
            "n_wrong_high_conf": 0,
            "n_correct_low_conf": 0,
            "priority_score": "",
            "priority_reason": "NOT_AVAILABLE: no validation prediction/logprob rows were provided",
        })
    return sorted(out, key=lambda r: (str(r.get("priority_score", "")), str(r.get("n", ""))), reverse=True)


def classify_failure(row: Dict[str, Any], category: str) -> Dict[str, Any]:
    text = " ".join(str(row.get(k, "")) for k in ["question", "raw_output", "reasoning_text", "final_answer_text", "pred_answer"])
    low = text.lower()
    if norm_bool(row.get("parse_success")) is False or not row.get("pred_answer"):
        ftype, reason = "final_parse_error", "prediction/final answer could not be parsed"
    elif category == "cryptarithm":
        if "leading zero" in low or "cannot be zero" in low:
            ftype, reason = "leading_zero_error", "leading-zero constraint appears in prompt/output"
        elif "carry" in low:
            ftype, reason = "carry_error", "carry reasoning appears central"
        elif "same digit" in low or "conflict" in low or "distinct" in low:
            ftype, reason = "mapping_conflict", "digit-letter distinctness or mapping conflict signal"
        elif "try" in low or "search" in low:
            ftype, reason = "incomplete_search", "search/enumeration signal in output"
        elif re.search(r"\d+\s*[+\-*]\s*\d+", low):
            ftype, reason = "arithmetic_error", "numeric arithmetic signal in reasoning"
        else:
            ftype, reason = "unknown", "rule-based classifier did not find a specific cryptarithm failure signal"
    elif category == "bit_manipulation":
        pairs = [("xor_error", ["xor", "exclusive or", "^"]), ("and_error", ["bitwise and", "&"]),
                 ("or_error", ["bitwise or", "|"]), ("shift_error", ["shift", "<<", ">>"]),
                 ("mask_error", ["mask"]), ("signed_unsigned_error", ["signed", "unsigned", "two's complement"]),
                 ("base_conversion_error", ["binary", "hex", "decimal"])]
        ftype, reason = "unknown", "rule-based classifier did not find a specific bit failure signal"
        for typ, kws in pairs:
            if any(k in low for k in kws):
                ftype, reason = typ, f"matched {typ} keyword"
                break
    else:
        pairs = [("binary_decimal_error", ["binary to decimal", "0b"]), ("decimal_binary_error", ["decimal to binary"]),
                 ("hex_decimal_error", ["hex to decimal", "0x"]), ("decimal_hex_error", ["decimal to hex"]),
                 ("roman_numeral_error", ["roman"]), ("base_n_place_value_error", ["base", "radix"]),
                 ("digit_order_error", ["order", "reverse"])]
        ftype, reason = "unknown", "rule-based classifier did not find a specific numeral failure signal"
        for typ, kws in pairs:
            if any(k in low for k in kws):
                ftype, reason = typ, f"matched {typ} keyword"
                break
    solver = category in {"cryptarithm", "bit_manipulation", "numeral_conversion"}
    template = {
        "cryptarithm": "column-wise constraint template with explicit carry and leading-zero checks",
        "bit_manipulation": "operator-by-operator binary workspace template with decimal cross-check",
        "numeral_conversion": "place-value expansion template with inverse conversion check",
    }.get(category, "final answer extraction template")
    priority = 5 if ftype not in {"unknown", "final_parse_error"} else (4 if ftype == "final_parse_error" else 3)
    return {"failure_type": ftype, "failure_reason": reason, "solver_check_possible": solver,
            "synthetic_generation_possible": solver, "recommended_template": template, "example_priority": priority}


def build_failure_cases(pred_rows: List[Dict[str, Any]], log_by_id: Dict[str, Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    rows = []
    for r in pred_rows:
        if r.get("category") != category:
            continue
        lp = log_by_id.get(r.get("problem_id", ""), {})
        is_wrong = norm_bool(r.get("is_correct")) is False
        low_correct = norm_bool(r.get("is_correct")) is True and to_float(lp.get("answer_min_logprob") or lp.get("min_logprob")) <= -5.0
        if not (is_wrong or low_correct):
            continue
        c = classify_failure(r, category)
        base = {**r, **lp, **c}
        rows.append(base)
    return rows


def build_failure_type_summary(*case_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    total_by_cat = Counter()
    for cases in case_lists:
        for r in cases:
            cat = r.get("category", "cryptarithm")
            if "category" not in r and "subcategory" in r:
                cat = "cryptarithm"
            typ = r.get("failure_type", "unknown")
            buckets[(cat, typ)].append(r)
            total_by_cat[cat] += 1
    out = []
    for (cat, typ), rows in buckets.items():
        n = len(rows)
        out.append({"category": cat, "failure_type": typ, "n": n,
                    "share": f"{n / total_by_cat[cat]:.6f}" if total_by_cat[cat] else "",
                    "priority_mean": avg([to_float(r.get("example_priority")) for r in rows])})
    if not out:
        out.append({"category": "ALL", "failure_type": "NOT_AVAILABLE", "n": 0, "share": "", "priority_mean": ""})
    return sorted(out, key=lambda r: (r.get("category", ""), int(r.get("n", 0) or 0)), reverse=True)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "NOT_AVAILABLE"


def write_docs(outdir: Path, args: argparse.Namespace, generated: List[str], summaries: Dict[str, Any]) -> None:
    measured = summaries.get("measured", False)
    accuracy = summaries.get("accuracy", "NOT_AVAILABLE")
    parse_rate = summaries.get("parse_rate", "NOT_AVAILABLE")
    n_total = summaries.get("n_total", 0)
    cat_table = summaries.get("category_failure_rows", [])[:10]
    fail_rows = summaries.get("failure_type_rows", [])[:10]

    readme = f"""# Phase3 Analysis

This directory contains the Phase3 analysis artifact set for the Golden Baseline failure-category workflow.

## Safety status

- Training/SFT: **not executed**.
- Golden adapter mutation: **not executed**.
- `adapter_model.safetensors` generation/overwrite: **not executed**.
- `adapter_config.json` mutation: **not executed**.
- `submission.zip` creation: **not executed**.
- Kaggle submission/Public LB check: **not executed**.

## Inputs observed by this run

- Golden notebook/script: `{args.golden_notebook or 'b3-nemotron-svd-26042701.ipynb'}`
- Adapter path: `{args.adapter_path or 'NOT_PROVIDED'}`
- Validation path: `{args.validation or 'NOT_PROVIDED'}`
- Prediction log path: `{args.predictions or 'NOT_PROVIDED'}`
- Logprob path: `{args.logprobs or 'NOT_PROVIDED'}`

## Offline one-file notebook

- `phase3_offline_analysis.ipynb` is the self-contained Kaggle Internet-OFF notebook version for RTX Pro5000 environments.
- It uses Python standard library only and embeds the Phase3 analysis code in one code cell; it does not require importing `phase3_make_recommendation.py`.
- The notebook is analysis-only: no training, no adapter mutation, no `submission.zip` creation, and no Kaggle submission.

## Golden Baseline input/log audit

See `phase3_analysis/golden_baseline_input_audit.md` for the attached notebook inspection.  The notebook exposes the Golden adapter path and adapter diagnostics, but it does not define a validation/held-out dataset path and does not create a row-level Golden prediction log.  If no external per-problem Golden validation prediction artifact exists, do not modify Golden Baseline code by default; run category-map/CV-design analysis first and keep measured prediction/logprob sections as `NOT_AVAILABLE`. A logging observer is a later HOLD option only if needed, and it must not change prompt, decoding, adapter, model, or submission behavior.

## Validation source clarification

- Kaggle `train.csv` is the labeled source from which a Phase3 validation/held-out set can be made.
- Prefer a deterministic held-out split derived from `train.csv`, for example `/kaggle/working/phase3_validation_split.csv`, rather than treating the whole training file as the validation set.
- The `--validation` file must contain gold answers; Kaggle `test.csv` is not sufficient for local accuracy/failure analysis.
- Kaggle Discussion search did not yield a verified public CV split in this environment; if none is confirmed, use a deterministic split from competition `train.csv` as the first option.

## Deterministic split creation

Create the first-pass Phase3 validation file from Kaggle `train.csv` without touching Golden Baseline inference:

```bash
python phase3_make_recommendation.py \
  --output-dir phase3_analysis \
  --create-validation-from-train /kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv \
  --validation-output /kaggle/working/phase3_validation_split.csv \
  --validation-fraction 0.2 \
  --split-seed 42
```

The script preserves all original columns and writes `/kaggle/working/phase3_validation_split.manifest.json` with seed, method, and row counts.

## Execution handoff

See `phase3_analysis/execution_handoff_plan.md` for the split between repository-side assistant work and Kaggle-side user execution. The first-pass recommendation is to run category-map/CV-design mode on a deterministic `train.csv`-derived split without modifying Golden Baseline logging.

## Classification rules

Categories are assigned by deterministic keyword/rule matching.  Cryptarithm, bit manipulation, and numeral conversion rules are evaluated before broader categories.
Ambiguous rows are marked `manual_review_required=True` instead of being silently treated as reliable `other` examples.

### High-priority category rules

- `cryptarithm`: alphametic/cryptarithm keywords, distinct digit/letter mapping language, or uppercase letter equations such as `SEND + MORE = MONEY`.
- `bit_manipulation`: bitwise operator keywords and symbols (`xor`, `&`, `|`, `^`, `<<`, `>>`, mask, signed/unsigned).
- `numeral_conversion`: explicit base/radix conversion, binary/hex literals, decimal/hex/binary/Roman conversion language.

## Measurement status

- Validation rows analyzed: `{n_total}`
- Golden validation accuracy: `{accuracy}`
- Parse success rate: `{parse_rate}`
- Status: `{'MEASURED_FROM_LOCAL_LOGS' if measured else 'NOT_MEASURED_NO_VALIDATION_OR_PREDICTION_LOGS'}`
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    run_commands = f"""# Phase3 Run Commands

## Commands executed

```bash
python phase3_make_recommendation.py --output-dir {outdir}
```

Actual command line captured by script:

```bash
{' '.join([repr(x) if ' ' in x else x for x in os.sys.argv])}
```

## Inputs

- Golden notebook/script: `{args.golden_notebook or 'b3-nemotron-svd-26042701.ipynb'}`
- Adapter path: `{args.adapter_path or 'NOT_PROVIDED'}`
- Validation path: `{args.validation or 'NOT_PROVIDED'}`
- Prediction path: `{args.predictions or 'NOT_PROVIDED'}`
- Logprob path: `{args.logprobs or 'NOT_PROVIDED'}`

## Repro settings

- Seed: `{args.seed}`
- Generation config: `{args.generation_config}`
- Script version: `phase3_make_recommendation.py stdlib-v1`
- Git hash: `{git_hash()}`
- Execution datetime UTC: `{datetime.now(timezone.utc).isoformat()}`

## Generated artifacts

""" + "\n".join(f"- `{p}`" for p in generated) + "\n"
    (outdir / "run_commands.md").write_text(run_commands, encoding="utf-8")

    repro = """# Phase3 Reproducibility Notes

## Conditions for identical reruns

- Use the same Golden notebook/script, Golden adapter path, validation/held-out data, prediction log, logprob log, seed, and generation config.
- If generating predictions before this script, keep the Golden Baseline decoding settings unchanged.
- Do not alter LoRA rank, target modules, dtype, adapter files, or prompt templates during Phase3 analysis.

## Remaining nondeterminism

- This script is deterministic for fixed input files.
- Model generation outside this script may remain nondeterministic if CUDA kernels, sampling, vLLM scheduling, or temperature settings are nondeterministic.

## Logprob availability

- If token logprobs are present in the input logprob file, `min_logprob_summary.csv` and priority fields use them.
- If logprobs are unavailable, logprob columns are left blank and report sections mark confidence-based conclusions as unmeasured.

## vLLM vs transformers

- vLLM and transformers can differ in token accounting, finish reasons, and logprob APIs.
- Compare only runs created with the same inference backend unless the backend change itself is the isolated experiment variable.

## Kaggle Internet OFF compatibility

- The script uses only Python standard-library modules and local files, so it is suitable for Kaggle Internet OFF analysis.
- It does not clone Huikang pipeline code, download models, train, mutate adapters, zip submissions, or call Kaggle APIs.

## Fallback if inference/logprob extraction fails

1. Prefer an existing Golden prediction JSONL via `--predictions` if it already exists; do not create new Golden Baseline logging just for the first Phase3 pass.
2. Provide a held-out/validation CSV/JSONL via `--validation` for category labeling and CV design.
3. Rerun this script to compute all summaries that can be computed from available logs.
4. If predictions/logprobs remain unavailable, keep measured accuracy/failure/logprob conclusions as `NOT_AVAILABLE` and mark logging as HOLD.
"""
    (outdir / "reproducibility_notes.md").write_text(repro, encoding="utf-8")

    md_cat = "\n".join(
        f"| {r.get('category','')} | {r.get('subcategory','')} | {r.get('n','')} | {r.get('accuracy','')} | {r.get('avg_min_logprob','')} | {r.get('n_wrong_low_conf','')} | {r.get('n_correct_low_conf','')} | {r.get('priority_score','')} |"
        for r in cat_table
    ) or "| NOT_AVAILABLE | NOT_AVAILABLE | 0 | NOT_AVAILABLE |  |  |  |  |"
    md_fail = "\n".join(
        f"| {r.get('category','')} | {r.get('failure_type','')} | {r.get('n','')} | {r.get('share','')} |"
        for r in fail_rows
    ) or "| NOT_AVAILABLE | NOT_AVAILABLE | 0 | |"
    recommendation = f"""# Phase3 Analysis Report

## 0. Current Operating Priority

- Do **not** modify the Golden Baseline notebook just to add logging in the first pass.
- First option: build a deterministic `train.csv`-derived held-out split and run category-map/CV-design analysis.
- If no existing Golden prediction/logprob artifact is available, measured accuracy/failure/logprob sections remain `NOT_AVAILABLE`.
- A logging-only validation runner is HOLD / low priority and should be added only later if measured evidence is worth the Golden Baseline touch risk.

## 0.5 Handoff Summary

- Assistant/repo side: maintain the offline analysis notebook/script, schemas, documentation, and post-run interpretation.
- User/Kaggle side: create the deterministic `train.csv`-derived held-out split, run the self-contained notebook in Internet-OFF mode, and return the generated `phase3_analysis/` outputs.
- First pass should not add Golden Baseline logging; prediction/logprob measurements stay `NOT_AVAILABLE` unless existing artifacts are supplied.

## 1. Summary

### Facts from local artifacts

- Validation件数: `{n_total}`
- Golden accuracy: `{accuracy}`
- Parse success rate: `{parse_rate}`
- Measurement status: `{'MEASURED_FROM_LOCAL_LOGS' if measured else 'NOT_MEASURED: validation/prediction/logprob inputs were not present in this repository'}`

### Unconfirmed / not measured in this run

- Public LB 0.86 is treated as user-provided context only; this script did not submit to Kaggle or query the Public LB.
- Category weakness rankings are data-driven only when validation predictions are supplied.  Without those logs, the next-experiment list below is a guarded hypothesis framework, not a measured claim.

## 2. Category Failure Summary

| category | subcategory | n | accuracy | avg_min_logprob | n_wrong_low_conf | n_correct_low_conf | priority_score |
|---|---:|---:|---:|---:|---:|---:|---:|
{md_cat}

## 3. Cryptarithm Findings

- Main failure types are computed in `failure_cases_cryptarithm.csv` when cryptarithm rows and Golden predictions are available.
- Solver check possible: yes for well-formed alphametic/digit-assignment tasks.
- Synthetic generation possible: yes, but should be isolated into one-variable +100-example experiments.
- Recommended template: column-wise constraint table with explicit carry, leading-zero, and mapping-conflict checks.

## 4. Bit Manipulation Findings

- Main failure types are computed in `failure_cases_bit_manipulation.csv` when local rows are available.
- Recommended next candidate: XOR/base-conversion examples with operator-by-operator binary workspace and decimal cross-check.

## 5. Numeral Conversion Findings

- Main failure types are computed in `failure_cases_numeral_conversion.csv` when local rows are available.
- Recommended next candidate: place-value conversion examples with inverse conversion checks.

## 6. Protected Cases

- Protected cases are defined as `is_correct=True` and low `answer_min_logprob`/`min_logprob`.
- In this run, protected cases are `{('available in min_logprob_summary.csv' if measured else 'NOT_AVAILABLE because no measured prediction/logprob file was provided')}`.

## 7. Next Experiment Candidates

1. cryptarithm carry-focused +100（推奨度：★★★★★）
   - 理由: carry mistakes are solver-checkable and can be trained with a narrow template.
   - baselineとの差分: add only 100 carry-focused cryptarithm examples.
   - 変更対象: next-phase training data only; no adapter structure change.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_carry --n 100`
   - 評価方法: rerun Phase3 validation and compare cryptarithm carry accuracy plus protected-case retention.
   - 採用条件: carry subset improves without reducing stable-correct categories.
   - rollback方法: remove the +100 carry data file and restore previous adapter checkpoint.
   - 失敗リスク: overfitting to addition wording or disturbing already-correct alphametics.

2. cryptarithm leading-zero +100（推奨度：★★★★☆）
   - 理由: leading-zero constraints are explicit, solver-checkable, and easy to isolate.
   - baselineとの差分: add only leading-zero constraint examples.
   - 変更対象: next-phase training data only.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_leading_zero --n 100`
   - 評価方法: leading-zero failure-type count and final accuracy.
   - 採用条件: fewer leading-zero errors with no parse regression.
   - rollback方法: remove leading-zero data and checkpoint.
   - 失敗リスク: model may over-mention leading-zero constraints when absent.

3. cryptarithm mapping-conflict +100（推奨度：★★★★☆）
   - 理由: distinct-digit mapping errors are common in alphametic reasoning and solver-verifiable.
   - baselineとの差分: add only mapping-conflict examples.
   - 変更対象: next-phase training data only.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_mapping_conflict --n 100`
   - 評価方法: mapping-conflict failure count and protected cryptarithm retention.
   - 採用条件: improvement on conflict cases without arithmetic/carry regression.
   - rollback方法: remove mapping-conflict data and checkpoint.
   - 失敗リスク: longer search traces may hurt concise final-answer parsing.

4. bit XOR/base conversion +100（推奨度：★★★☆☆）
   - 理由: bitwise XOR plus base conversion is solver-checkable and complements cryptarithm.
   - baselineとの差分: add only XOR/base conversion examples.
   - 変更対象: next-phase data only.
   - 実行コマンド: `python make_sft_data.py --type bit_xor_base --n 100`
   - 評価方法: bit_manipulation accuracy and parse success.
   - 採用条件: measurable bit subset gain with no cryptarithm regression.
   - rollback方法: remove XOR/base data and checkpoint.
   - 失敗リスク: base conversion artifacts can leak into numeral-only problems.

5. numeral conversion +100（推奨度：★★★☆☆）
   - 理由: deterministic place-value tasks are low-risk and solver-checkable.
   - baselineとの差分: add only numeral-conversion examples.
   - 変更対象: next-phase data only.
   - 実行コマンド: `python make_sft_data.py --type numeral_conversion --n 100`
   - 評価方法: numeral_conversion accuracy and protected-case retention.
   - 採用条件: improves numeral subset without answer-format degradation.
   - rollback方法: remove numeral data and checkpoint.
   - 失敗リスク: easy examples may not transfer to harder reasoning tasks.

6. answer format / final parse template only（推奨度：★★★☆☆）
   - 理由: targets parse failures only and avoids changing reasoning content.
   - baselineとの差分: prompt/template post-processing only; no SFT data change.
   - 変更対象: final answer extraction template.
   - 実行コマンド: `python run_validation.py --template final_parse_v2`
   - 評価方法: parse success and no accuracy loss on stable-correct cases.
   - 採用条件: parse_success improves and accuracy does not drop.
   - rollback方法: revert template file.
   - 失敗リスク: format changes can alter exact answer matching.

7. min logprob lower-tail replay candidate（推奨度：★★★☆☆）
   - 理由: replays low-confidence correct and wrong examples to protect fragile wins and target uncertain misses.
   - baselineとの差分: select only lower-tail logprob examples; no architecture change.
   - 変更対象: next-phase data sampling list.
   - 実行コマンド: `python select_lower_tail.py --input phase3_analysis/min_logprob_summary.csv`
   - 評価方法: retention of correct-low-confidence cases plus wrong-low-confidence recovery.
   - 採用条件: protects fragile correct examples while improving selected misses.
   - rollback方法: remove lower-tail replay list and checkpoint.
   - 失敗リスク: requires reliable logprobs; without logprobs this is HOLD.

## 8. ADOPT / HOLD / REJECT

### ADOPT

- Adopt only categories that have enough local failures, solver verification, synthetic generation, and no protected-case regression in Phase3 outputs.

### HOLD

- Hold all confidence/logprob claims when `min_logprob_summary.csv` lacks logprob values.
- Hold categories with low count, ambiguous classification, or validation skew.

### REJECT

- Reject adapter-rank/target-module/dtype changes, Golden adapter edits, public-LB-only feedback loops, and multi-variable experiments in the next phase.

## Failure Type Summary

| category | failure_type | n | share |
|---|---:|---:|---:|
{md_fail}
"""
    (outdir / "phase3_recommendation.md").write_text(recommendation, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline Phase3 Nemotron analysis artifacts from local logs.")
    p.add_argument("--output-dir", default="phase3_analysis")
    p.add_argument("--validation", type=Path, default=None, help="CSV/JSON/JSONL validation or held-out dataset")
    p.add_argument("--create-validation-from-train", type=Path, default=None, help="Optional Kaggle train.csv source used to create a deterministic held-out validation split")
    p.add_argument("--validation-output", type=Path, default=Path("/kaggle/working/phase3_validation_split.csv"), help="Output CSV path for --create-validation-from-train")
    p.add_argument("--train-remainder-output", type=Path, default=None, help="Optional CSV path for non-held-out rows")
    p.add_argument("--validation-fraction", type=float, default=0.2, help="Held-out fraction when --validation-size is not set")
    p.add_argument("--validation-size", type=int, default=None, help="Exact held-out row count; overrides --validation-fraction")
    p.add_argument("--split-seed", default="42", help="Seed string used in deterministic hash split")
    p.add_argument("--predictions", type=Path, default=None, help="Golden validation prediction JSONL/CSV")
    p.add_argument("--logprobs", type=Path, default=None, help="Token/logprob summary JSONL/CSV")
    p.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH", ""))
    p.add_argument("--golden-notebook", default="b3-nemotron-svd-26042701.ipynb")
    p.add_argument("--seed", default=os.environ.get("SEED", "NOT_PROVIDED"))
    p.add_argument("--generation-config", default=os.environ.get("GENERATION_CONFIG", "Golden Baseline unchanged; not executed by this script"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    split_manifest = None
    if args.create_validation_from_train and not args.validation:
        split_manifest = create_deterministic_validation_split(
            train_csv=args.create_validation_from_train,
            validation_output=args.validation_output,
            remainder_output=args.train_remainder_output,
            seed=args.split_seed,
            validation_fraction=args.validation_fraction,
            validation_size=args.validation_size,
        )
        args.validation = args.validation_output

    validation_rows = read_table(args.validation) if args.validation else []
    pred_input_rows = read_table(args.predictions) if args.predictions else []
    log_input_rows = read_table(args.logprobs) if args.logprobs else []

    cat_rows, labeled_rows = build_category_rows(validation_rows)
    cat_by_id = {r["problem_id"]: r for r in cat_rows}
    val_by_id = {r["problem_id"]: r for r in labeled_rows}
    pred_rows = normalize_prediction_rows(pred_input_rows, cat_by_id, val_by_id)

    log_rows = normalize_logprob_rows(log_input_rows, pred_rows)
    log_by_id = {r.get("problem_id", ""): r for r in log_rows}

    crypt_cases = build_failure_cases(pred_rows, log_by_id, "cryptarithm")
    bit_cases = build_failure_cases(pred_rows, log_by_id, "bit_manipulation")
    num_cases = build_failure_cases(pred_rows, log_by_id, "numeral_conversion")
    failure_type_rows = build_failure_type_summary(crypt_cases, bit_cases, num_cases)
    cat_failure_rows = build_category_failure(log_rows)
    summary_rows = summarize_predictions(pred_rows)

    write_csv(outdir / "category_map.csv", CATEGORY_MAP_COLUMNS, cat_rows)
    write_csv(outdir / "validation_set_labeled.csv", CATEGORY_MAP_COLUMNS + VALIDATION_LABELED_EXTRA, labeled_rows)
    write_jsonl(outdir / "golden_validation_predictions.jsonl", pred_rows)
    write_csv(outdir / "golden_validation_summary.csv", OVERALL_SUMMARY_COLUMNS, summary_rows)
    write_csv(outdir / "min_logprob_summary.csv", LOGPROB_COLUMNS, log_rows)
    write_csv(outdir / "category_failure_summary.csv", CATEGORY_FAILURE_COLUMNS, cat_failure_rows)
    write_csv(outdir / "failure_type_summary.csv", FAILURE_TYPE_COLUMNS, failure_type_rows)
    write_csv(outdir / "failure_cases_cryptarithm.csv", CRYPT_COLUMNS, crypt_cases)
    write_csv(outdir / "failure_cases_bit_manipulation.csv", BIT_NUM_COLUMNS, bit_cases)
    write_csv(outdir / "failure_cases_numeral_conversion.csv", BIT_NUM_COLUMNS, num_cases)

    overall = summary_rows[0] if summary_rows else {}
    generated = [str(p) for p in sorted(outdir.glob("*"))]
    write_docs(outdir, args, generated, {
        "measured": bool(pred_rows),
        "n_total": overall.get("n_total", 0),
        "accuracy": overall.get("accuracy", "NOT_AVAILABLE"),
        "parse_rate": overall.get("parse_success_rate", "NOT_AVAILABLE"),
        "category_failure_rows": cat_failure_rows,
        "failure_type_rows": failure_type_rows,
    })

    # Regenerate generated list after docs were written.
    generated = [str(p) for p in sorted(outdir.glob("*"))]
    run_text = (outdir / "run_commands.md").read_text(encoding="utf-8")
    if "phase3_recommendation.md" not in run_text:
        with (outdir / "run_commands.md").open("a", encoding="utf-8") as f:
            f.write("\n## Final artifact inventory\n\n")
            for pth in generated:
                f.write(f"- `{pth}`\n")

    print(json.dumps({"output_dir": str(outdir), "artifacts": generated, "measured_predictions": len(pred_rows), "split_manifest": split_manifest}, indent=2))


if __name__ == "__main__":
    main()
