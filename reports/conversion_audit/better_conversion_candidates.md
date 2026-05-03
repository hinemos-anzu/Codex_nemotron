# Better Conversion Candidates

候補名: Error-aware module/layer rank map
期待効果: SVD誤差が大きい層へrankを再配分して性能劣化を抑制。
リスク: サイズ制約(2.5GB)超過。
実装難易度: 中
過去実験との関係: 一律圧縮で悪化した履歴に対する代替。
最小検証方法: 総rank予算固定で上位誤差層のみ+2 rank。
Go/No-Go条件: rank<=32かつzip制限内でLB>=0.86。

候補名: Mamba gate/x merge の orientation・scaling再検証
期待効果: in_proj統合時の情報欠落を低減。
リスク: 実装バグで互換性破壊。
実装難易度: 中
過去実験との関係: tensor swap/residualを使わずconversion-onlyで改善余地。
最小検証方法: 変換前後のΔW Fro誤差とkey整合監査。
Go/No-Go条件: unexpected keyゼロ・誤差低下・LB非劣化。

候補名: deterministic SVD vs randomized/QR-SVD 比較
期待効果: 同rankで再構成誤差低減。
リスク: 計算コスト増。
実装難易度: 低〜中
過去実験との関係: 学習なしで変換品質を上げる方針に一致。
最小検証方法: 同一テンソル群で誤差統計比較。
Go/No-Go条件: 誤差改善が再現し、最終制約を全満たし。
