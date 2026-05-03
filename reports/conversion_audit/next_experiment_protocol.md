# Next Experiment Protocol (Conversion-only)

## 目的
baseline 0.86 を壊さず、conversion/surgery/SVD 工程のみで改善余地を検証する。

## 前提
- 学習禁止: post-finetune / lm_head-only / residual / tensor swap は実施しない。
- 制約: rank <= 32, zero-shape なし, unexpected fused key なし。

## 実験セット

### Exp-A: Error-aware rank map
1. baseline 変換を実行し SVD 誤差を算出
2. 相対 Fro 誤差 上位 10% の module/layer に +2 rank（上限32）
3. 低誤差層から同量 rank を減算し総サイズ予算を固定
4. submission.zip を監査して LB を比較

### Exp-B: Mamba gate/x merge orientation/scaling 再検証
1. gate/x -> in_proj merge 前後で ΔW 再構成誤差を保存
2. A/B 向き・concat順序・scale の組み合わせを最小探索
3. key整合/shape整合監査を通過した候補のみ LB 比較

### Exp-C: SVD solver 比較
1. deterministic SVD / QR-SVD / randomized SVD を同 rank で実行
2. module別・layer別 relative Fro 誤差を比較
3. 最低誤差かつ制約通過の方式を採択

## 評価基準
- Hard constraints
  - max rank <= 32
  - rank violation = 0
  - zero-shape tensor = 0
  - missing target_modules = 0
  - bad patterns (`.experts.w1/.w2/.w3`, `.gate_proj`, `.x_proj`) = 0
- Soft targets
  - SVD relative error の上位層誤差を baseline 比で低減
  - LB が 0.86 非劣化（理想: 改善）

## 実行コマンド（Kaggle資産マウント環境）
```bash
python conversion_audit.py \
  --adapter-path /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
  --baseline-submission-path /kaggle/working/submission.zip
```

## Go / No-Go
- Go:
  - Hard constraints 全通過
  - baseline 比で誤差統計改善
  - LB >= 0.86
- No-Go:
  - Hard constraints 1つでも失敗
  - LB < 0.86

## 出力物
- reports/conversion_audit/baseline_adapter_inventory.{md,json}
- reports/conversion_audit/conversion_stage_log.{md,json}
- reports/conversion_audit/svd_error_report.{md,json}
- reports/conversion_audit/better_conversion_candidates.md
- reports/conversion_audit/generator_summary.md
