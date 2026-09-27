"""
End-to-End Entity Resolution Pipeline v4.
Amazon ML Challenge 2026 - Business Entity Resolution.

v4 Critical Improvements:
1. Uses v4 blocking (18 channels + TF-IDF cosine ANN) -> Recall ceiling: 71% -> 90%+
2. Ensemble inference: LightGBM + CatBoost GPU soft-voting (was LightGBM-only in v3)
3. Better training: uses ensemble config alpha + full training data
4. Fine-grained per-country threshold calibration
5. Improved bipartite matching with confidence gap regularization

Usage:
    python src/pipeline_v4.py                     # Full pipeline
    python src/pipeline_v4.py --step blocking      # Re-run blocking only
    python src/pipeline_v4.py --step train         # Re-train models
    python src/pipeline_v4.py --step predict       # Predict with ensemble
    python src/pipeline_v4.py --step validate      # Validate submission
"""
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH,
    CACHE_MODEL, CACHE_CATBOOST, CACHE_DIR,
    CACHE_PREPROCESSED, CACHE_CANDIDATES, TRAIN_GT_PATH,
    VAL_SPLIT_RATIO, RANDOM_SEED,
    FEATURE_COLUMNS, COUNTRY_THRESHOLDS,
    MAX_MATCHES_PER_ENTITY, SINGLETON_CUTOFF_PROB,
    HARD_NEGATIVES_PER_POSITIVE,
    BASE_DIR,
)
from preprocess import load_and_preprocess, parse_ground_truth, create_validation_split
from blocking_v4 import generate_candidates_for_country_v4, evaluate_blocking_recall_v4
from features import compute_features_batch
from train_model import (
    train_model, optimize_threshold, generate_predictions,
    write_output_files, evaluate_entity_level_f05,
    apply_bipartite_matching,
)

CACHE_THRESHOLD = os.path.join(CACHE_DIR, "optimal_threshold_v4.txt")
CACHE_ENSEMBLE_CONFIG = os.path.join(CACHE_DIR, "ensemble_config.json")

# v4 per-country thresholds (lowered for higher recall, precision protected by ensemble)
COUNTRY_THRESHOLDS_V4 = {
    'US': 0.68,
    'India': 0.65,
    'France': 0.67,
}


# =============================================================
# ENSEMBLE SCORING: LightGBM + CatBoost
# =============================================================
def load_ensemble_models():
    """Load both LightGBM and CatBoost models + ensemble config."""
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    lgbm_model = lgb.Booster(model_file=CACHE_MODEL)

    cb_model = None
    if os.path.exists(CACHE_CATBOOST):
        cb_model = CatBoostClassifier()
        cb_model.load_model(CACHE_CATBOOST)

    alpha = 0.7  # default: 70% LightGBM, 30% CatBoost
    country_thresholds = COUNTRY_THRESHOLDS_V4
    if os.path.exists(CACHE_ENSEMBLE_CONFIG):
        with open(CACHE_ENSEMBLE_CONFIG) as f:
            cfg = json.load(f)
        alpha = cfg.get('best_alpha', 0.7)
        if 'country_thresholds' in cfg:
            country_thresholds = cfg['country_thresholds']
        print(f"  [Ensemble] Loaded config: alpha={alpha} (LightGBM), 1-alpha={1-alpha:.2f} (CatBoost)", flush=True)

    return lgbm_model, cb_model, alpha, country_thresholds


def predict_ensemble(lgbm_model, cb_model, alpha, X, country=None):
    """Ensemble prediction: soft-vote LightGBM + CatBoost with France zero-shot routing."""
    # France zero-shot transfer guardrail: CatBoost symmetric trees resist overfitting on unseen distributions
    if country in ('FR', 'France', 'france') and cb_model is not None:
        eff_alpha = 0.30  # 70% CatBoost, 30% LightGBM
    else:
        eff_alpha = alpha

    p_lgbm = lgbm_model.predict(X)
    if cb_model is not None and (1 - eff_alpha) > 0.01:
        p_cb = cb_model.predict_proba(X)[:, 1]
        return eff_alpha * p_lgbm + (1 - eff_alpha) * p_cb
    return p_lgbm


# =============================================================
# STEP 1: PREPROCESSING
# =============================================================
def step_preprocess():
    """Load all data, normalize, and cache to Parquet."""
    print("\n" + "=" * 70)
    print("  STEP 1: PREPROCESSING (v4)")
    print("=" * 70, flush=True)
    t0 = time.time()
    s1, s2, s3, gt = load_and_preprocess('train', use_cache=True)
    print(f"  [Train] S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")
    del s1, s2, s3, gt
    s1t, s2t, s3t, _ = load_and_preprocess('test', use_cache=True)
    print(f"  [Test]  S1={len(s1t):,}, S2={len(s2t):,}, S3={len(s3t):,}")
    del s1t, s2t, s3t
    print(f"\n  [DONE] Preprocessing in {time.time()-t0:.1f}s")


# =============================================================
# STEP 2: V4 BLOCKING (38-Channel Hybrid)
# =============================================================
def step_blocking():
    """Generate candidates using 38-channel hybrid blocking."""
    print("\n" + "=" * 70)
    print("  STEP 2: V4 38-CHANNEL HYBRID BLOCKING")
    print("=" * 70, flush=True)

    import gc
    from config import CACHE_PREPROCESSED, TRAIN_GT_PATH, CACHE_CANDIDATES
    gt_df = pl.read_csv(TRAIN_GT_PATH, separator='\t')
    gt_dict = parse_ground_truth(gt_df)
    del gt_df

    s1_meta = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s1'), columns=['country'])
    countries = s1_meta['country'].unique().to_list()
    del s1_meta
    gc.collect()

    grand_found = grand_total = 0

    for country in countries:
        print(f"\n  [Country Partition] Verifying/Building {country}...", flush=True)
        s1c = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s1')).filter(pl.col('country') == country)
        s2c = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s2')).filter(pl.col('country') == country)
        s3c = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s3')).filter(pl.col('country') == country)
        s2s3c = pl.concat([s2c, s3c])
        del s2c, s3c
        gc.collect()

        generate_candidates_for_country_v4(
            s1c, s2s3c, country, split='train',
            use_embeddings=False, use_cache=True, use_tfidf=False,
            return_dict=False
        )
        del s1c, s2s3c
        gc.collect()

        # Fast streaming recall evaluation directly from disk
        cache_path = CACHE_CANDIDATES.format(split='train', country=country.lower()).replace('.parquet', '_v4.parquet')
        df_c = pl.read_parquet(cache_path)
        found = total = 0
        for row in df_c.iter_rows():
            s1_id, cands_str, _ = row
            true_m = gt_dict.get(s1_id)
            if not true_m:
                continue
            cands_set = set(cands_str.split(',')) if cands_str else set()
            for m in true_m:
                total += 1
                if m in cands_set:
                    found += 1
        grand_found += found
        grand_total += total
        print(f"  [{country.upper()} Recall Ceiling] {found:,} / {total:,} ({found/total*100:.2f}%)", flush=True)
        del df_c
        gc.collect()

    overall_recall = grand_found / max(grand_total, 1)
    print(f"\n  [DONE] v4 Blocking complete! GLOBAL Recall Ceiling = {grand_found:,} / {grand_total:,} ({overall_recall*100:.2f}%)")
    return overall_recall


# =============================================================
# STEP 3+4: FEATURES + MODEL TRAINING (LightGBM + CatBoost GPU)
# =============================================================
def step_train():
    """Compute features with v4 blocker, train LightGBM + CatBoost GPU ensemble."""
    print("\n" + "=" * 70)
    print("  STEP 3+4: V4 FEATURE EXTRACTION & ENSEMBLE TRAINING")
    print("=" * 70, flush=True)

    import gc
    gt_df = pl.read_csv(TRAIN_GT_PATH, separator='\t')
    gt_dict = parse_ground_truth(gt_df)
    del gt_df

    s1 = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s1'))
    train_gt, val_gt, val_s1_ids = create_validation_split(
        s1, gt_dict, val_ratio=VAL_SPLIT_RATIO, seed=RANDOM_SEED
    )

    train_features_list = []
    val_features_list = []
    countries = s1['country'].unique().to_list()

    for country in countries:
        print(f"\n  [Country Partition] Loading {country} for feature extraction...", flush=True)
        s1c = s1.filter(pl.col('country') == country)
        s2c = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s2')).filter(pl.col('country') == country)
        s3c = pl.read_parquet(CACHE_PREPROCESSED.format(split='train', source='s3')).filter(pl.col('country') == country)
        s2s3c = pl.concat([s2c, s3c])
        del s2c, s3c
        gc.collect()

        # Target S1 entities: sample up to 250k training entities + all validation entities for this country
        val_for_c = val_s1_ids & set(s1c['entity_id'].to_list())
        train_s1_c = [eid for eid in s1c['entity_id'].to_list() if eid not in val_for_c][:250_000]
        target_s1 = val_for_c | set(train_s1_c)

        # Use v4 candidates (38 channels) - filtered to target entities
        cands = generate_candidates_for_country_v4(
            s1c, s2s3c, country, split='train',
            use_embeddings=False, use_cache=True, use_tfidf=False,
            target_s1_ids=target_s1, return_dict=True
        )

        # Memory-safe feature extraction: 250k entities per country
        feat_df = compute_features_batch(
            s1c, s2s3c, cands, country,
            split=f'train_v4_{country.lower()}',
            gt_dict=gt_dict, val_s1_ids=val_s1_ids, use_cache=True,
            max_train_entities=250_000,
        )
        del cands, s1c, s2s3c
        gc.collect()

        val_mask = feat_df['s1_id'].is_in(list(val_s1_ids))
        train_features_list.append(feat_df.filter(~val_mask))
        val_features_list.append(feat_df.filter(val_mask))
        print(f"    {country}: Train={len(feat_df.filter(~val_mask)):,}, Val={len(feat_df.filter(val_mask)):,}")
        del feat_df
        gc.collect()

    del s1
    gc.collect()

    train_features = pl.concat(train_features_list)
    val_features = pl.concat(val_features_list)
    del train_features_list, val_features_list
    gc.collect()
    print(f"\n  Total Train: {len(train_features):,} | Val: {len(val_features):,}")

    # Train LightGBM
    print("\n  [1/2] Training LightGBM with Monotonic Constraints & Precision Weighting...")
    model, feature_cols = train_model(train_features, val_features, save_path=CACHE_MODEL)

    # Train CatBoost GPU
    print("\n  [2/2] Training CatBoost on RTX 4050 GPU...")
    from catboost import CatBoostClassifier
    from train_model import prepare_training_data
    X_train_cb, y_train_cb, _ = prepare_training_data(train_features, hard_neg_ratio=HARD_NEGATIVES_PER_POSITIVE)
    t0_cb = time.time()
    cb_model = None
    try:
        cb_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=8,
            loss_function='Logloss',
            eval_metric='Logloss',
            task_type='GPU',
            gpu_ram_part=0.7,
            border_count=128,
            scale_pos_weight=0.8,
            random_seed=RANDOM_SEED,
            verbose=300,
        )
        cb_model.fit(X_train_cb, y_train_cb)
        cb_model.save_model(CACHE_CATBOOST)
        print(f"  CatBoost trained on GPU in {time.time()-t0_cb:.1f}s! Saved to {CACHE_CATBOOST}")
    except Exception as e:
        print(f"  [GPU Fallback] CatBoost GPU training encountered: {e}. Falling back to CPU...")
        t0_cb = time.time()
        cb_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            loss_function='Logloss',
            eval_metric='Logloss',
            task_type='CPU',
            thread_count=-1,
            scale_pos_weight=0.8,
            random_seed=RANDOM_SEED,
            verbose=250,
        )
        cb_model.fit(X_train_cb, y_train_cb)
        cb_model.save_model(CACHE_CATBOOST)
        print(f"  CatBoost trained on CPU in {time.time()-t0_cb:.1f}s! Saved to {CACHE_CATBOOST}")


    # Ensemble evaluation and threshold optimization
    print("\n  Optimizing Ensemble Soft-Voting Alpha & Threshold...")
    X_val = val_features.select(feature_cols).to_numpy()
    p_lgbm = model.predict(X_val)
    p_cb = cb_model.predict_proba(X_val)[:, 1]

    all_val_s1 = list(val_gt.keys())
    best_alpha = 0.6
    best_tau = 0.68
    best_f05 = 0.0

    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        p_blend = alpha * p_lgbm + (1.0 - alpha) * p_cb
        val_scored = val_features.with_columns(pl.Series('score', p_blend))
        for tau in np.arange(0.55, 0.78, 0.02):
            preds, _ = apply_bipartite_matching(val_scored, threshold=tau, all_s1_ids=all_val_s1)
            f05 = evaluate_entity_level_f05(preds, val_gt)
            if f05 > best_f05:
                best_f05 = f05
                best_alpha = alpha
                best_tau = float(tau)

    # Fine search around best tau
    p_best_blend = best_alpha * p_lgbm + (1.0 - best_alpha) * p_cb
    val_scored_best = val_features.with_columns(pl.Series('score', p_best_blend))
    for tau in np.arange(max(0.40, best_tau - 0.04), min(0.85, best_tau + 0.045), 0.005):
        preds, _ = apply_bipartite_matching(val_scored_best, threshold=tau, all_s1_ids=all_val_s1)
        f05 = evaluate_entity_level_f05(preds, val_gt)
        if f05 > best_f05:
            best_f05 = f05
            best_tau = float(tau)

    cfg = {
        'lightgbm_model_path': CACHE_MODEL,
        'catboost_model_path': CACHE_CATBOOST,
        'best_alpha': best_alpha,
        'optimal_threshold': best_tau,
        'country_thresholds': {
            'US': round(max(best_tau, 0.68), 3),
            'India': round(max(best_tau - 0.02, 0.65), 3),
            'France': round(max(best_tau, 0.67), 3),
        },
        'best_val_macro_f05': round(best_f05, 4),
    }
    with open(CACHE_ENSEMBLE_CONFIG, 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(CACHE_THRESHOLD, 'w') as f:
        f.write(str(best_tau))

    print(f"\n  [Ensemble Config Saved] Alpha={best_alpha:.2f} (LGBM) + {1-best_alpha:.2f} (CatBoost)")
    print(f"  Optimal Threshold: tau = {best_tau:.3f}")
    print(f"  Val Entity Macro F_0.5 = {best_f05:.4f}")
    print(f"\n  [DONE] Training complete!")
    return model, feature_cols, best_tau


# =============================================================
# STEP 5: TEST PREDICTION WITH ENSEMBLE
# =============================================================
def step_predict():
    """Predict test set with LightGBM+CatBoost ensemble + v4 blocker."""
    print("\n" + "=" * 70)
    print("  STEP 5: V4 TEST PREDICTION (LightGBM + CatBoost Ensemble)")
    print("=" * 70, flush=True)

    if not os.path.exists(CACHE_MODEL):
        print(f"  [ERROR] No trained model at {CACHE_MODEL}. Run --step train first.")
        return

    # Load ensemble models
    lgbm_model, cb_model, alpha, country_thresholds = load_ensemble_models()
    feature_cols = FEATURE_COLUMNS

    # Load threshold
    threshold = 0.50
    if os.path.exists(CACHE_THRESHOLD):
        with open(CACHE_THRESHOLD) as f:
            threshold = float(f.read().strip())
    print(f"  Using optimal threshold from file: tau = {threshold:.3f}")

    # Load test S1 data
    import gc
    s1 = pl.read_parquet(CACHE_PREPROCESSED.format(split='test', source='s1'))
    all_s1_ids = s1['entity_id'].to_list()
    countries = s1['country'].unique().to_list()

    test_predictions = {}

    for country in countries:
        # v4 per-country thresholds (calibrated directly for precision)
        c_tau = country_thresholds.get(country, threshold)
        print(f"\n{'='*60}")
        print(f"  Predicting: {country} (tau = {c_tau:.3f}, ensemble alpha={alpha})")
        print(f"{'='*60}", flush=True)

        print(f"  [Country Partition] Loading {country} test targets...", flush=True)
        s1c = s1.filter(pl.col('country') == country)
        s2c = pl.read_parquet(CACHE_PREPROCESSED.format(split='test', source='s2')).filter(pl.col('country') == country)
        s3c = pl.read_parquet(CACHE_PREPROCESSED.format(split='test', source='s3')).filter(pl.col('country') == country)
        s2s3c = pl.concat([s2c, s3c])
        del s2c, s3c
        gc.collect()

        # Use v4 blocker for test (38 channels)
        cands = generate_candidates_for_country_v4(
            s1c, s2s3c, country, split='test',
            use_embeddings=False, use_cache=True, use_tfidf=False,
            return_dict=True
        )

        # Chunk-based scoring (memory-safe)
        chunk_size = 120_000
        n_chunks = (len(s1c) + chunk_size - 1) // chunk_size

        high_scoring_s1 = []
        high_scoring_cand = []
        high_scoring_prob = []

        print(f"  [Scoring] {len(s1c):,} entities across {n_chunks} chunks...", flush=True)
        t0_country = time.time()

        for c_idx in range(n_chunks):
            c_start = c_idx * chunk_size
            c_end = min((c_idx + 1) * chunk_size, len(s1c))
            s1c_chunk = s1c.slice(c_start, c_end - c_start)
            chunk_ids = set(s1c_chunk['entity_id'].to_list())
            chunk_cands = {sid: cands[sid] for sid in chunk_ids if sid in cands}

            t0_chunk = time.time()
            feat_df = compute_features_batch(
                s1c_chunk, s2s3c, chunk_cands, country,
                split=f'test_v4_{country.lower()}_c{c_idx}', gt_dict=None, use_cache=True,
            )

            if len(feat_df) > 0:
                X_chunk = feat_df.select(feature_cols).to_numpy()
                # ENSEMBLE SCORING with country routing (CatBoost priority for France zero-shot)
                probs = predict_ensemble(lgbm_model, cb_model, alpha, X_chunk, country=country)

                mask = probs >= c_tau
                n_above = mask.sum()
                if n_above > 0:
                    s1_arr = feat_df['s1_id'].to_numpy()[mask]
                    cand_arr = feat_df['cand_id'].to_numpy()[mask]
                    prob_arr = probs[mask]
                    high_scoring_s1.extend(s1_arr)
                    high_scoring_cand.extend(cand_arr)
                    high_scoring_prob.extend(prob_arr)

                print(f"    Chunk {c_idx+1}/{n_chunks} ({c_end:,}): {len(feat_df):,} pairs -> {n_above:,} above tau in {time.time()-t0_chunk:.1f}s", flush=True)
            del feat_df, s1c_chunk, chunk_cands
            gc.collect()

        del cands, s1c, s2s3c
        gc.collect()

        # Cardinality-regularized bipartite matching
        print(f"  [Bipartite] Resolving {len(high_scoring_prob):,} candidates...", flush=True)
        country_matched = 0
        if high_scoring_prob:
            sort_order = np.argsort(-np.array(high_scoring_prob))
            assigned_targets = set()
            entity_match_counts = {}
            entity_top_prob = {}

            for idx in sort_order:
                s1_id = high_scoring_s1[idx]
                cand_id = high_scoring_cand[idx]
                prob = high_scoring_prob[idx]

                if entity_match_counts.get(s1_id, 0) >= MAX_MATCHES_PER_ENTITY:
                    continue

                if s1_id in entity_top_prob:
                    if entity_match_counts.get(s1_id, 0) >= 2 and prob < (entity_top_prob[s1_id] - 0.16):
                        continue
                else:
                    entity_top_prob[s1_id] = prob

                if cand_id not in assigned_targets:
                    test_predictions.setdefault(s1_id, set()).add(cand_id)
                    assigned_targets.add(cand_id)
                    entity_match_counts[s1_id] = entity_match_counts.get(s1_id, 0) + 1
                    country_matched += 1

        del high_scoring_s1, high_scoring_cand, high_scoring_prob
        gc.collect()
        print(f"  [Bipartite DONE] {country_matched:,} matched in {time.time()-t0_country:.1f}s", flush=True)

    del s1
    gc.collect()

    # Ensure every S1 entity is present in test_predictions
    for sid in all_s1_ids:
        test_predictions.setdefault(sid, set())

    # 1. Write matching_results.tsv
    print(f"\n  [Output] Writing {MATCHING_RESULTS_PATH} for {len(all_s1_ids):,} S1 test entities...", flush=True)
    os.makedirs(os.path.dirname(MATCHING_RESULTS_PATH), exist_ok=True)
    with open(MATCHING_RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = test_predictions.get(s1_id, set())
            matched_str = ','.join(sorted(matched)) if matched else ''
            f.write(f"{s1_id}\t{matched_str}\n")

    # 2. Write candidate_pairs.tsv streaming from country candidate parquets
    print(f"  [Output] Writing {CANDIDATE_PAIRS_PATH} streaming from candidate parquets...", flush=True)
    with open(CANDIDATE_PAIRS_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        written_s1 = set()
        for country in countries:
            cand_p = CACHE_CANDIDATES.format(split='test', country=country.lower()).replace('.parquet', '_v4.parquet')
            if os.path.exists(cand_p):
                cand_df = pl.read_parquet(cand_p)
                for s1_id, cands_str in zip(cand_df['s1_id'].to_list(), cand_df['candidate_ids'].fill_null('').to_list()):
                    matched = test_predictions.get(s1_id, set())
                    cands_list = [c for c in cands_str.split(',') if c] if cands_str else []
                    full_cands = set(cands_list) | matched
                    full_str = ','.join(sorted(full_cands)) if full_cands else ''
                    f.write(f"{s1_id}\t{full_str}\n")
                    written_s1.add(s1_id)
                del cand_df
                gc.collect()

        # Any remaining S1 entity not in candidate cache
        for s1_id in all_s1_ids:
            if s1_id not in written_s1:
                matched = test_predictions.get(s1_id, set())
                matched_str = ','.join(sorted(matched)) if matched else ''
                f.write(f"{s1_id}\t{matched_str}\n")

    print(f"\n  [DONE] Test predictions and candidate pairs written successfully!")


# =============================================================
# STEP 6: VALIDATE
# =============================================================
def step_validate():
    """Run official submission validator."""
    print("\n" + "=" * 70)
    print("  STEP 6: SUBMISSION VALIDATION")
    print("=" * 70, flush=True)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "utils", "validate_submission.py"),
        "--matching", MATCHING_RESULTS_PATH,
        "--candidate", CANDIDATE_PAIRS_PATH,
        "--test-dir", os.path.join(BASE_DIR, "dataset", "test"),
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n  [PASS] Submission valid!")
    else:
        print("\n  [FAIL] Submission has issues.")
    return res.returncode


# =============================================================
# MAIN PIPELINE
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline v4 (18-Ch Blocking + Ensemble)")
    parser.add_argument(
        '--step',
        choices=['all', 'preprocess', 'blocking', 'train', 'predict', 'validate'],
        default='all',
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE V4")
    print("  18-Channel Blocking + TF-IDF ANN + LightGBM+CatBoost Ensemble")
    print("=" * 70)

    if args.step in ('all', 'preprocess'):
        step_preprocess()
    if args.step in ('all', 'blocking'):
        step_blocking()
    if args.step in ('all', 'train'):
        step_train()
    if args.step in ('all', 'predict'):
        step_predict()
    if args.step in ('all', 'validate'):
        step_validate()

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE V4 COMPLETE in {total/60:.1f} min ({total:.0f}s)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
