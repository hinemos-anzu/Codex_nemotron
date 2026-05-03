# generator_summary

実行環境: local container (no kaggle model assets)

入力アセット: notebook b3-nemotron-svd-26042701.ipynb と既定Kaggleパス

監査結果: 現環境ではadapter未配置のため実体監査は未実行。

段階別結果: conversion stage logのscaffoldを生成。

SVD誤差: 現環境では未計算。

better conversion候補: better_conversion_candidates.md参照。

推奨次実験: Kaggle資産マウント環境で同スクリプト実行。

実装しなかったこと: 学習/post-finetune/residual/tensor swap。

リスク: 資産未配置時は定量監査が空になる。

再現コマンド: `python conversion_audit.py`
