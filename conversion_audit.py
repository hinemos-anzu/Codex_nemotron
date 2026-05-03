import json, os, re, hashlib, zipfile
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

EXPECTED_MODULES = ["k_proj","o_proj","in_proj","q_proj","up_proj","v_proj","down_proj","out_proj","lm_head"]
BAD_PATTERNS=[".experts.w1.",".experts.w2.",".experts.w3.",".gate_proj.",".x_proj."]


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')

def sha256_file(path: Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()

def load_tensors(path: Path):
    from safetensors import safe_open
    out={}
    with safe_open(str(path), framework='pt', device='cpu') as f:
        for k in f.keys():
            out[k]=tuple(f.get_slice(k).get_shape())
    return out

def module_from_key(key:str):
    for m in EXPECTED_MODULES+['gate_proj','x_proj']:
        if f'.{m}.' in key:
            return m
    return 'other'

def rank_from_pair(shapeA, shapeB):
    # A: [r,in], B:[out,r]
    if len(shapeA)==2: return shapeA[0]
    return None

def audit_adapter(adapter_dir: Path):
    cpath=adapter_dir/'adapter_config.json'; mpath=adapter_dir/'adapter_model.safetensors'
    if not cpath.exists() or not mpath.exists():
        raise FileNotFoundError(f'missing files in {adapter_dir}')
    cfg=json.loads(cpath.read_text())
    tensors=load_tensors(mpath)
    num=len(tensors)
    module_count=Counter(); ranks=[]; zero=[]; unexpected=[]
    has_backbone=False; has_model_model=False
    keyset=set(tensors)
    for k,s in tensors.items():
        has_backbone |= 'base_model.model.backbone' in k
        has_model_model |= 'base_model.model.model' in k
        if 0 in s: zero.append({'key':k,'shape':list(s)})
        if any(p in k for p in BAD_PATTERNS): unexpected.append(k)
        module_count[module_from_key(k)] +=1
        if k.endswith('.lora_A.weight'):
            bkey=k.replace('.lora_A.weight','.lora_B.weight')
            if bkey in keyset:
                r=rank_from_pair(s,tensors[bkey]); ranks.append({'key':k,'rank':r})
    rank_dist=Counter([r['rank'] for r in ranks if r['rank'] is not None])
    max_rank=max(rank_dist) if rank_dist else None
    rank_viol=[r for r in ranks if r['rank'] and r['rank']>32]
    present_mod={k.split('.')[-3] for k in tensors if k.endswith('.lora_A.weight')}
    missing=[m for m in EXPECTED_MODULES if m not in present_mod]
    return {
        'adapter_config':cfg,
        'target_modules':cfg.get('target_modules',[]),
        'lora_alpha':cfg.get('lora_alpha'),
        'r':cfg.get('r'),
        'inference_mode':cfg.get('inference_mode'),
        'tensor_key_count':num,
        'module_tensor_count':dict(module_count),
        'rank_distribution':{str(k):v for k,v in sorted(rank_dist.items())},
        'max_rank':max_rank,
        'rank_violations':rank_viol,
        'zero_shape_tensors':zero,
        'unexpected_keys':unexpected,
        'missing_expected_modules':missing,
        'prefix_check':{'contains_backbone':has_backbone,'contains_model_model':has_model_model},
        'adapter_model_size_bytes':mpath.stat().st_size,
        'adapter_model_sha256':sha256_file(mpath),
    }

def zip_inventory(zpath: Path):
    if not zpath.exists(): return {'exists':False}
    with zipfile.ZipFile(zpath) as zf:
        entries=[i.filename for i in zf.infolist()]
    return {'exists':True,'path':str(zpath),'sha256':sha256_file(zpath),'entries':entries,'size_bytes':zpath.stat().st_size}

def main():
    model_path=Path(os.getenv('MODEL_PATH','/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1'))
    adapter_path=Path(os.getenv('ADAPTER_PATH','/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20'))
    baseline_submission=Path(os.getenv('BASELINE_SUBMISSION_PATH','/kaggle/working/submission.zip'))
    outdir=Path('reports/conversion_audit'); outdir.mkdir(parents=True,exist_ok=True)
    report={'timestamp_utc':now(),'model_path':str(model_path),'adapter_path':str(adapter_path),'baseline_submission_path':str(baseline_submission)}
    if adapter_path.exists():
        report['baseline_inventory']=audit_adapter(adapter_path)
    else:
        report['baseline_inventory']={'error':'adapter path not found in this environment'}
    report['zip_inventory']=zip_inventory(baseline_submission)
    # stage log scaffold
    stage_names=['raw adapter load','prefix rename','expert unfuse','Mamba gate/x merge','SVD compression','target_modules reconciliation','adapter_config rewrite','final safetensors export','submission.zip export']
    report['conversion_stage_log']=[{'stage':s,'status':'not_executed_in_this_environment','notes':'instrumentation scaffold only'} for s in stage_names]
    report['svd_error_report']={'status':'not_computed','reason':'requires conversion run with original and compressed factors'}

    (outdir/'baseline_adapter_inventory.json').write_text(json.dumps({'report':report},indent=2,ensure_ascii=False))
    (outdir/'conversion_stage_log.json').write_text(json.dumps(report['conversion_stage_log'],indent=2,ensure_ascii=False))
    (outdir/'svd_error_report.json').write_text(json.dumps(report['svd_error_report'],indent=2,ensure_ascii=False))

    (outdir/'baseline_adapter_inventory.md').write_text('# Baseline Adapter Inventory\n\n```json\n'+json.dumps(report,indent=2,ensure_ascii=False)+'\n```\n')
    (outdir/'conversion_stage_log.md').write_text('# Conversion Stage Log\n\n'+'\n'.join([f"- {s['stage']}: {s['status']} ({s['notes']})" for s in report['conversion_stage_log']]))
    (outdir/'svd_error_report.md').write_text('# SVD Error Report\n\n- status: not_computed\n- reason: requires conversion run with original and compressed factors\n')

    candidates='''# Better Conversion Candidates\n\n候補名: Error-aware module/layer rank map\n期待効果: SVD誤差が大きい層へrankを再配分して性能劣化を抑制。\nリスク: サイズ制約(2.5GB)超過。\n実装難易度: 中\n過去実験との関係: 一律圧縮で悪化した履歴に対する代替。\n最小検証方法: 総rank予算固定で上位誤差層のみ+2 rank。\nGo/No-Go条件: rank<=32かつzip制限内でLB>=0.86。\n\n候補名: Mamba gate/x merge の orientation・scaling再検証\n期待効果: in_proj統合時の情報欠落を低減。\nリスク: 実装バグで互換性破壊。\n実装難易度: 中\n過去実験との関係: tensor swap/residualを使わずconversion-onlyで改善余地。\n最小検証方法: 変換前後のΔW Fro誤差とkey整合監査。\nGo/No-Go条件: unexpected keyゼロ・誤差低下・LB非劣化。\n\n候補名: deterministic SVD vs randomized/QR-SVD 比較\n期待効果: 同rankで再構成誤差低減。\nリスク: 計算コスト増。\n実装難易度: 低〜中\n過去実験との関係: 学習なしで変換品質を上げる方針に一致。\n最小検証方法: 同一テンソル群で誤差統計比較。\nGo/No-Go条件: 誤差改善が再現し、最終制約を全満たし。\n'''
    (outdir/'better_conversion_candidates.md').write_text(candidates)
    (outdir/'generator_summary.md').write_text('# generator_summary\n\n実行環境: local container (no kaggle model assets)\n\n入力アセット: notebook b3-nemotron-svd-26042701.ipynb と既定Kaggleパス\n\n監査結果: 現環境ではadapter未配置のため実体監査は未実行。\n\n段階別結果: conversion stage logのscaffoldを生成。\n\nSVD誤差: 現環境では未計算。\n\nbetter conversion候補: better_conversion_candidates.md参照。\n\n推奨次実験: Kaggle資産マウント環境で同スクリプト実行。\n\n実装しなかったこと: 学習/post-finetune/residual/tensor swap。\n\nリスク: 資産未配置時は定量監査が空になる。\n\n再現コマンド: `python conversion_audit.py`\n')

if __name__=='__main__':
    main()
