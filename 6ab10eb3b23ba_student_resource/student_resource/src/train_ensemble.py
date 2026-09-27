"""
Multi-Model Ensemble Training & Optimization Module (v3 Blueprint).
Amazon ML Challenge 2026 - Business Entity Resolution.
Leverages NVIDIA RTX 4050 GPU (DirectML/CUDA) for CatBoost & LightGBM.

Evaluates and compares:
1. LightGBM (Leaf-wise GBDT with monotonic constraints)
2. CatBoost GPU (Oblivious Symmetric Trees on RTX 4050)
3. Weighted Soft-Voting Ensemble (P_blend = alpha * P_lgbm + (1-alpha) * P_cb)
4. Rank-Averaging Ensemble

Reports complete Entity-Level Macro F0.5 scores, threshold sensitivity curves,
and country breakdown (US vs India).
"""
import os
import sys
import time
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_COLUMNS, CACHE_MODEL, CACHE_CATBOOST,
    RANDOM_SEED, VAL_SPLIT_RATIO, MAX_MATCHES_PER_ENTITY,
    HARD_NEGATIVES_PER_POSITIVE, BASE_DIR, CACHE_DIR
)
from preprocess import load_and_preprocess, parse_ground_truth, create_validation_split
from train_model import prepare_training_data, evaluate_entity_level_f05, apply_bipartite_matching

CACHE_ENSEMBLE_CONFIG = os.path.join(CACHE_DIR, "ensemble_config.json")


def evaluate_model_at_thresholds(val_df, probs, val_gt, all_val_s1, thresholds=None):
    """
    Evaluates Entity-Level Macro F_0.5 across a sweep of thresholds.
    Returns (best_threshold, best_f05, all_results).
    """
    if thresholds is None:
        thresholds = np.arange(0.40, 0.78, 0.02)
        
    val_df_scored = val_df.with_columns(pl.Series('score', probs))
    
    best_tau = 0.50
    best_score = 0.0
    sweep_results = []
    
    for tau in thresholds:
        preds, _ = apply_bipartite_matching(
            val_df_scored, threshold=tau, all_s1_ids=all_val_s1,
            max_matches_per_entity=MAX_MATCHES_PER_ENTITY
        )
        macro_f05 = evaluate_entity_level_f05(preds, val_gt)
        sweep_results.append((float(tau), float(macro_f05)))
        if macro_f05 > best_score:
            best_score = macro_f05
            best_tau = float(tau)
            
    # Fine search (+/- 0.03 in steps of 0.005)
    fine_range = np.arange(max(0.35, best_tau - 0.03), min(0.85, best_tau + 0.035), 0.005)
    for tau in fine_range:
        preds, _ = apply_bipartite_matching(
            val_df_scored, threshold=tau, all_s1_ids=all_val_s1,
            max_matches_per_entity=MAX_MATCHES_PER_ENTITY
        )
        macro_f05 = evaluate_entity_level_f05(preds, val_gt)
        sweep_results.append((float(tau), float(macro_f05)))
        if macro_f05 > best_score:
            best_score = macro_f05
            best_tau = float(tau)
            
    sweep_results.sort(key=lambda x: x[0])
    return best_tau, best_score, sweep_results


def run_ensemble_benchmark():
    print("=" * 75)
    print("  MULTI-MODEL ENSEMBLE BENCHMARK & OPTIMIZATION (v3)")
    print("  Hardware: NVIDIA GeForce RTX 4050 GPU + 12-Core CPU")
    print("=" * 75, flush=True)
    t_start = time.time()
    
    # 1. Load Ground Truth and Validation Split
    s1, s2, s3, gt = load_and_preprocess('train', use_cache=True)
    gt_dict = parse_ground_truth(gt)
    del s2, s3, gt
    
    train_gt, val_gt, val_s1_ids = create_validation_split(
        s1, gt_dict, val_ratio=VAL_SPLIT_RATIO, seed=RANDOM_SEED
    )
    all_val_s1 = list(val_gt.keys())
    print(f"  Ground truth loaded: {len(gt_dict):,} S1 entities | Validation S1: {len(all_val_s1):,}", flush=True)
    
    # Country-specific validation IDs
    s1_country_map = dict(zip(s1['entity_id'].to_list(), s1['country'].to_list()))
    del s1
    
    val_gt_us = {sid: matches for sid, matches in val_gt.items() if s1_country_map.get(sid) == 'US'}
    val_gt_india = {sid: matches for sid, matches in val_gt.items() if s1_country_map.get(sid) == 'India'}
    all_val_us = list(val_gt_us.keys())
    all_val_india = list(val_gt_india.keys())
    
    # 2. Load Precomputed Training Features
    print("\n  Loading precomputed feature matrices from cache...", flush=True)
    feat_in_path = os.path.join(CACHE_DIR, "features_train_india.parquet")
    feat_us_path = os.path.join(CACHE_DIR, "features_train_us.parquet")
    
    df_feat_in = pl.read_parquet(feat_in_path)
    df_feat_us = pl.read_parquet(feat_us_path)
    all_features = pl.concat([df_feat_in, df_feat_us])
    del df_feat_in, df_feat_us
    print(f"  Total Candidate Pair Features: {len(all_features):,} samples", flush=True)
    
    # Split train and validation pairs
    val_mask = all_features['s1_id'].is_in(list(val_s1_ids))
    train_feat = all_features.filter(~val_mask)
    val_feat = all_features.filter(val_mask)
    del all_features
    print(f"  Train pairs: {len(train_feat):,} | Val pairs: {len(val_feat):,}", flush=True)
    
    # 3. Model 1: LightGBM Model
    feature_cols = FEATURE_COLUMNS
    print("\n" + "-" * 75)
    print("  [MODEL 1] LightGBM Classifier (Monotonic Constrained Leaf-Wise GBDT)")
    print("-" * 75, flush=True)
    
    if os.path.exists(CACHE_MODEL):
        print(f"  [Cache HIT] Loading trained LightGBM model from {CACHE_MODEL}...")
        lgbm_model = lgb.Booster(model_file=CACHE_MODEL)
    else:
        print("  Training LightGBM model...")
        from train_model import train_model as train_lgbm
        lgbm_model, _ = train_lgbm(train_feat, val_feat, save_path=CACHE_MODEL)
        
    X_val = val_feat.select(feature_cols).to_numpy()
    p_lgbm = lgbm_model.predict(X_val)
    
    tau_lgbm, f05_lgbm, sweep_lgbm = evaluate_model_at_thresholds(val_feat, p_lgbm, val_gt, all_val_s1)
    
    # Evaluate per-country
    val_scored_lgbm = val_feat.with_columns(pl.Series('score', p_lgbm))
    preds_us, _ = apply_bipartite_matching(val_scored_lgbm, threshold=tau_lgbm, all_s1_ids=all_val_us)
    f05_us_lgbm = evaluate_entity_level_f05(preds_us, val_gt_us)
    preds_in, _ = apply_bipartite_matching(val_scored_lgbm, threshold=tau_lgbm, all_s1_ids=all_val_india)
    f05_in_lgbm = evaluate_entity_level_f05(preds_in, val_gt_india)
    
    print(f"  LightGBM Optimal Threshold: tau = {tau_lgbm:.3f}")
    print(f"  LightGBM Macro F0.5 (Overall):  {f05_lgbm:.4f}")
    print(f"  LightGBM Macro F0.5 (US):       {f05_us_lgbm:.4f}")
    print(f"  LightGBM Macro F0.5 (India):    {f05_in_lgbm:.4f}")
    
    # 4. Model 2: CatBoost on RTX 4050 GPU
    print("\n" + "-" * 75)
    print("  [MODEL 2] CatBoost GPU Classifier (Oblivious Symmetric Trees on RTX 4050)")
    print("-" * 75, flush=True)
    
    if os.path.exists(CACHE_CATBOOST):
        print(f"  [Cache HIT] Loading trained CatBoost model from {CACHE_CATBOOST}...")
        cb_model = CatBoostClassifier()
        cb_model.load_model(CACHE_CATBOOST)
    else:
        print("  Preparing balanced training matrix for CatBoost GPU...")
        X_train, y_train, _ = prepare_training_data(train_feat, hard_neg_ratio=HARD_NEGATIVES_PER_POSITIVE)
        
        t0_cb = time.time()
        cb_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=8,
            loss_function='Logloss',
            eval_metric='Logloss',
            task_type='GPU',
            random_seed=RANDOM_SEED,
            verbose=200
        )
        cb_model.fit(X_train, y_train)
        print(f"  CatBoost trained on GPU in {time.time() - t0_cb:.1f}s!", flush=True)
        cb_model.save_model(CACHE_CATBOOST)
        print(f"  Saved CatBoost model to {CACHE_CATBOOST}")
        
    p_cb = cb_model.predict_proba(X_val)[:, 1]
    
    tau_cb, f05_cb, sweep_cb = evaluate_model_at_thresholds(val_feat, p_cb, val_gt, all_val_s1)
    val_scored_cb = val_feat.with_columns(pl.Series('score', p_cb))
    preds_us_cb, _ = apply_bipartite_matching(val_scored_cb, threshold=tau_cb, all_s1_ids=all_val_us)
    f05_us_cb = evaluate_entity_level_f05(preds_us_cb, val_gt_us)
    preds_in_cb, _ = apply_bipartite_matching(val_scored_cb, threshold=tau_cb, all_s1_ids=all_val_india)
    f05_in_cb = evaluate_entity_level_f05(preds_in_cb, val_gt_india)
    
    print(f"  CatBoost Optimal Threshold: tau = {tau_cb:.3f}")
    print(f"  CatBoost Macro F0.5 (Overall):  {f05_cb:.4f}")
    print(f"  CatBoost Macro F0.5 (US):       {f05_us_cb:.4f}")
    print(f"  CatBoost Macro F0.5 (India):    {f05_in_cb:.4f}")
    
    # 5. Model 3: Weighted Blend Ensemble
    print("\n" + "-" * 75)
    print("  [MODEL 3] Weighted Ensemble: LightGBM + CatBoost GPU")
    print("-" * 75, flush=True)
    
    best_alpha = 0.50
    best_f05_blend = 0.0
    best_tau_blend = 0.50
    
    # Sweep alpha (weight of LightGBM vs CatBoost)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p_blend = alpha * p_lgbm + (1.0 - alpha) * p_cb
        tau_b, f05_b, _ = evaluate_model_at_thresholds(
            val_feat, p_blend, val_gt, all_val_s1,
            thresholds=np.arange(0.55, 0.76, 0.02)
        )
        print(f"    Alpha={alpha:.2f} (LGBM) + {1-alpha:.2f} (CatBoost) -> Best tau={tau_b:.3f}, F0.5={f05_b:.4f}")
        if f05_b > best_f05_blend:
            best_f05_blend = f05_b
            best_alpha = alpha
            best_tau_blend = tau_b
            
    p_best_blend = best_alpha * p_lgbm + (1.0 - best_alpha) * p_cb
    val_scored_blend = val_feat.with_columns(pl.Series('score', p_best_blend))
    preds_us_blend, _ = apply_bipartite_matching(val_scored_blend, threshold=best_tau_blend, all_s1_ids=all_val_us)
    f05_us_blend = evaluate_entity_level_f05(preds_us_blend, val_gt_us)
    preds_in_blend, _ = apply_bipartite_matching(val_scored_blend, threshold=best_tau_blend, all_s1_ids=all_val_india)
    f05_in_blend = evaluate_entity_level_f05(preds_in_blend, val_gt_india)
    
    # 6. Save Ensemble Configuration
    ensemble_cfg = {
        'lightgbm_model_path': CACHE_MODEL,
        'catboost_model_path': CACHE_CATBOOST,
        'best_alpha': best_alpha,
        'optimal_threshold': best_tau_blend,
        'country_thresholds': {
            'US': round(max(best_tau_blend, 0.70), 3),
            'India': round(max(best_tau_blend - 0.02, 0.67), 3),
            'France': round(max(best_tau_blend, 0.69), 3),
        },
        'validation_scores': {
            'lightgbm_macro_f05': round(f05_lgbm, 4),
            'catboost_macro_f05': round(f05_cb, 4),
            'ensemble_macro_f05': round(best_f05_blend, 4),
            'ensemble_us_f05': round(f05_us_blend, 4),
            'ensemble_india_f05': round(f05_in_blend, 4),
        }
    }
    with open(CACHE_ENSEMBLE_CONFIG, 'w') as f:
        json.dump(ensemble_cfg, f, indent=2)
    print(f"\n  Saved ensemble configuration to {CACHE_ENSEMBLE_CONFIG}")
    
    # 7. Print Comprehensive Benchmark Table
    print("\n" + "=" * 75)
    print("  FINAL MODEL & ENSEMBLE BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  {'Model / Architecture':<28} | {'Threshold':<10} | {'Overall F0.5':<14} | {'US F0.5':<10} | {'India F0.5':<10}")
    print("  " + "-" * 75)
    print(f"  {'LightGBM (Leaf-Wise)':<28} | {tau_lgbm:<10.3f} | {f05_lgbm:<14.4f} | {f05_us_lgbm:<10.4f} | {f05_in_lgbm:<10.4f}")
    print(f"  {'CatBoost (RTX 4050 GPU)':<28} | {tau_cb:<10.3f} | {f05_cb:<14.4f} | {f05_us_cb:<10.4f} | {f05_in_cb:<10.4f}")
    print(f"  {'Ensemble (Weighted Blend)':<28} | {best_tau_blend:<10.3f} | {best_f05_blend:<14.4f} | {f05_us_blend:<10.4f} | {f05_in_blend:<10.4f}")
    print("=" * 75)
    print(f"  Ensemble Gain over Single LightGBM: +{best_f05_blend - f05_lgbm:.4f} Macro F0.5")
    print(f"  Total Benchmark Time: {time.time() - t_start:.1f}s")
    print("=" * 75, flush=True)
    return ensemble_cfg


if __name__ == '__main__':
    run_ensemble_benchmark()
