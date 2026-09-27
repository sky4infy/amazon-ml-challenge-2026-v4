"""
High-Speed 56-Feature Engineering Module (v3 Blueprint).
Amazon ML Challenge 2026 - Business Entity Resolution.
C++ Accelerated via RapidFuzz SIMD.

Computes 56 discriminative features across 7 categories:
1. Name String Similarity (12 features)
2. Transliterated Name Similarity (6 features)
3. Address String Similarity (10 features)
4. Structured Geographic Signals (8 features)
5. Entity Structure & Corpus Frequency (6 features)
6. Cross-Field Interaction Signals (8 features)
7. Meta & Channel Statistics (6 features)
"""
import os
import sys
import time
import math
import numpy as np
from collections import Counter
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.Levenshtein as lev
from rapidfuzz import fuzz
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CACHE_FEATURES, FEATURE_COLUMNS


# ============================================================
# STRING SIMILARITY HELPERS (C++ ACCELERATED)
# ============================================================
def jaccard_token_sim(s1, s2):
    """Token-level Jaccard similarity: |A∩B| / |A∪B|"""
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    union_len = len(t1 | t2)
    return len(t1 & t2) / union_len if union_len > 0 else 0.0


def token_overlap_ratio(s1, s2):
    """Token overlap: |A∩B| / min(|A|, |B|)"""
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    min_len = min(len(t1), len(t2))
    return len(t1 & t2) / min_len if min_len > 0 else 0.0


def char_ngram_jaccard(s1, s2, n=3):
    """Character n-gram set Jaccard similarity."""
    if not s1 or not s2 or len(s1) < n or len(s2) < n:
        return 0.0
    ng1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ng2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    u = len(ng1 | ng2)
    return len(ng1 & ng2) / u if u > 0 else 0.0


def lcs_ratio(s1, s2):
    """Longest common subsequence length divided by max string length."""
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    max_len = max(m, n)
    if max_len == 0:
        return 0.0
    # Quick prefix/suffix check
    if s1 in s2 or s2 in s1:
        return min(m, n) / max_len
    # Fast DP for reasonable lengths
    if max_len > 80:
        return float(lev.normalized_similarity(s1, s2))
    dp = [0] * (n + 1)
    for c1 in s1:
        prev = 0
        for j, c2 in enumerate(s2):
            cur = dp[j+1]
            if c1 == c2:
                dp[j+1] = prev + 1
            elif dp[j] > dp[j+1]:
                dp[j+1] = dp[j]
            prev = cur
    return dp[n] / max_len


# ============================================================
# BATCH FEATURE EXTRACTION
# ============================================================
def compute_features_batch(s1_df, s2s3_df, candidates_dict, country,
                           split='train', gt_dict=None, val_s1_ids=None,
                           max_train_entities=250000, use_cache=True):
    """
    Computes all 56 discriminative features for candidate pairs in a country partition.
    Caches features to Parquet for instantaneous reuse.
    Supports Polars and Pandas input formats seamlessly.
    """
    cache_path = CACHE_FEATURES.format(split=split, country=country.lower())
    
    if use_cache and os.path.exists(cache_path):
        print(f"  [Cache HIT] Loading features for {country} from {cache_path}...", flush=True)
        t0 = time.time()
        df = pl.read_parquet(cache_path)
        print(f"  [Cache HIT] Loaded {len(df):,} candidate pair features in {time.time() - t0:.1f}s", flush=True)
        return df
    
    print(f"\n  [Features v3] Computing 56 features for {country} (S1={len(s1_df):,})...", flush=True)
    t0 = time.time()
    
    def _get_lists(df):
        def _col(name):
            return df[name].fill_null('').to_list() if hasattr(df, 'select') else df[name].fillna('').tolist()
        return (
            _col('entity_id'), _col('name_norm'),
            _col('name_blocking') if 'name_blocking' in df.columns else _col('name_norm'),
            _col('addr_norm'), _col('postal_code'),
            _col('postal_prefix_3') if 'postal_prefix_3' in df.columns else [''] * len(df),
            _col('city'), _col('street_number'),
        )
        
    s1_eids, s1_names, s1_blocks, s1_addrs, s1_postals, s1_p3s, s1_cities, s1_streets = _get_lists(s1_df)
    t_eids, t_names, t_blocks, t_addrs, t_postals, t_p3s, t_cities, t_streets = _get_lists(s2s3_df)
    
    # 1. Corpus name frequencies to detect generic duplicate entities
    name_freqs = Counter(s1_names)
    name_freqs.update(t_names)
    postal_freqs = Counter([p for p in s1_postals if p])
    postal_freqs.update([p for p in t_postals if p])
    city_freqs = Counter([c for c in s1_cities if c])
    city_freqs.update([c for c in t_cities if c])
    
    # 2. Build fast lookup dictionaries
    s1_lookup = {}
    for eid, name, block, addr, p, p3, city, st in zip(
        s1_eids, s1_names, s1_blocks, s1_addrs, s1_postals, s1_p3s, s1_cities, s1_streets
    ):
        s1_lookup[eid] = (name, block, addr, p, p3, city, st)
        
    s2s3_lookup = {}
    for eid, name, block, addr, p, p3, city, st in zip(
        t_eids, t_names, t_blocks, t_addrs, t_postals, t_p3s, t_cities, t_streets
    ):
        s2s3_lookup[eid] = (name, block, addr, p, p3, city, st)
        
    del s1_eids, s1_names, s1_blocks, s1_addrs, s1_postals, s1_p3s, s1_cities, s1_streets
    del t_eids, t_names, t_blocks, t_addrs, t_postals, t_p3s, t_cities, t_streets
    
    # Lists for features
    s1_id_list = []
    cand_id_list = []
    label_list = []
    
    # Feature columns storage
    feats = {col: [] for col in FEATURE_COLUMNS}
    
    total_s1 = len(candidates_dict)
    processed = 0
    train_entities_count = 0
    
    for s1_id, cand_info in candidates_dict.items():
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        n1, b1, a1, p1, p3_1, c1, sn1 = s1_rec
        
        cand_ids = cand_info.get('candidates', set())
        if not cand_ids:
            continue
            
        true_matches = gt_dict.get(s1_id, set()) if gt_dict is not None else None
        tfidf_scores = cand_info.get('tfidf_scores', {})
        ch_counts = cand_info.get('channel_counts', {})
        
        # In training mode, balance positives & hard negatives
        if true_matches is not None and val_s1_ids is not None:
            is_val = s1_id in val_s1_ids
            if not is_val:
                if train_entities_count >= max_train_entities:
                    continue
                train_entities_count += 1
                
                pos_cands = [c for c in cand_ids if c in true_matches]
                neg_cands = [c for c in cand_ids if c not in true_matches]
                if len(neg_cands) > 5:
                    # Sort negatives by highest TF-IDF or channel counts (hardest negatives)
                    neg_cands = sorted(neg_cands, key=lambda x: (ch_counts.get(x, 1), tfidf_scores.get(x, 0.0)), reverse=True)[:5]
                selected_cands = pos_cands + neg_cands
            else:
                if len(cand_ids) > 30:
                    selected_cands = sorted(cand_ids, key=lambda x: ch_counts.get(x, 1), reverse=True)[:30]
                else:
                    selected_cands = list(cand_ids)
        else:
            if len(cand_ids) > 30:
                selected_cands = sorted(cand_ids, key=lambda x: ch_counts.get(x, 1), reverse=True)[:30]
            else:
                selected_cands = list(cand_ids)
            
        # S1 precomputations
        sorted_ranks = {cid: r for r, (cid, _) in enumerate(sorted(tfidf_scores.items(), key=lambda x: -x[1]))}
        log_freq_s1 = math.log1p(name_freqs.get(n1, 1))
        t1_tokens = n1.split() if n1 else []
        t1_first = t1_tokens[0] if t1_tokens else ''
        t1_last = t1_tokens[-1] if t1_tokens else ''
        num_cands_val = len(cand_ids)
        
        for cand_id in selected_cands:
            c_rec = s2s3_lookup.get(cand_id)
            if c_rec is None:
                continue
            n2, b2, a2, p2, p3_2, c2, sn2 = c_rec
            
            s1_id_list.append(s1_id)
            cand_id_list.append(cand_id)
            
            if true_matches is not None:
                label_list.append(1 if cand_id in true_matches else 0)
                
            # 1. Name String Similarity
            sim_jw = float(jw.similarity(n1, n2)) if (n1 and n2) else 0.0
            sim_lev = float(lev.normalized_similarity(n1, n2)) if (n1 and n2) else 0.0
            sim_jaccard = jaccard_token_sim(n1, n2)
            sim_overlap = token_overlap_ratio(n1, n2)
            sim_sort = fuzz.token_sort_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
            sim_set = fuzz.token_set_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
            sim_lcs = lcs_ratio(n1, n2)
            
            t2_tokens = n2.split() if n2 else []
            t2_first = t2_tokens[0] if t2_tokens else ''
            t2_last = t2_tokens[-1] if t2_tokens else ''
            
            first_tok_m = 1.0 if (t1_first and t2_first and t1_first == t2_first) else 0.0
            last_tok_m = 1.0 if (t1_last and t2_last and t1_last == t2_last) else 0.0
            contains_m = 1.0 if (n1 and n2 and (n1 in n2 or n2 in n1)) else 0.0
            
            # Common prefix
            prefix_len = 0
            for c_a, c_b in zip(n1, n2):
                if c_a == c_b:
                    prefix_len += 1
                else:
                    break
            norm_prefix = prefix_len / max(len(n1), len(n2), 1)
            char_3g = char_ngram_jaccard(n1, n2, 3)
            
            # 2. Transliterated / Blocking Name Similarity
            t_jw = float(jw.similarity(b1, b2)) if (b1 and b2) else 0.0
            t_lev = float(lev.normalized_similarity(b1, b2)) if (b1 and b2) else 0.0
            t_jaccard = jaccard_token_sim(b1, b2)
            t_overlap = token_overlap_ratio(b1, b2)
            b1_tok = b1.split()[0] if b1 else ''
            b2_tok = b2.split()[0] if b2 else ''
            t_first = 1.0 if (b1_tok and b2_tok and b1_tok == b2_tok) else 0.0
            t_char_3g = char_ngram_jaccard(b1, b2, 3)
            
            # 3. Address String Similarity
            has_both_a = 1.0 if (a1 and a2) else 0.0
            missing_cand_a = 1.0 if not a2 else 0.0
            
            if has_both_a:
                a_jw = float(jw.similarity(a1, a2))
                a_lev = float(lev.normalized_similarity(a1, a2))
                a_jaccard = jaccard_token_sim(a1, a2)
                a_overlap = token_overlap_ratio(a1, a2)
                a_sort = fuzz.token_sort_ratio(a1, a2) / 100.0
                a_set = fuzz.token_set_ratio(a1, a2) / 100.0
                a_len_r = min(len(a1), len(a2)) / max(len(a1), len(a2), 1)
                a_4g = char_ngram_jaccard(a1, a2, 4)
            else:
                a_jw = 0.0
                a_lev = 0.0
                a_jaccard = 0.0
                a_overlap = 0.0
                a_sort = 0.0
                a_set = 0.0
                a_len_r = 0.0
                a_4g = 0.0
                
            # 4. Structured Geographic Signals
            if p1 and p2:
                pmatch = 1.0 if p1 == p2 else 0.0
                has_p_both = 1.0
            else:
                pmatch = -1.0
                has_p_both = 0.0
                
            if p3_1 and p3_2:
                p3match = 1.0 if p3_1 == p3_2 else 0.0
            else:
                p3match = -1.0
                
            if sn1 and sn2:
                snmatch = 1.0 if sn1 == sn2 else 0.0
            else:
                snmatch = -1.0
                
            sim_city = float(jw.similarity(c1, c2)) if (c1 and c2) else 0.0
            city_exact = 1.0 if (c1 and c2 and c1 == c2) else 0.0
            
            postal_l_freq = math.log1p(postal_freqs.get(p1, 0))
            city_l_freq = math.log1p(city_freqs.get(c1, 0))
            
            # 5. Entity Structure & Frequency
            name_len_r = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
            name_c_diff = abs(len(n1) - len(n2))
            name_t_diff = abs(len(t1_tokens) - len(t2_tokens))
            log_freq_cand = math.log1p(name_freqs.get(n2, 1))
            is_generic = 1.0 if name_freqs.get(n1, 1) > 50 else 0.0
            
            # 6. Cross-Field Interactions
            name_addr_p = sim_jw * a_jaccard
            name_addr_s = (sim_jw + a_jw) / 2.0
            name_post_p = sim_jw if pmatch == 1.0 else 0.0
            name_x_miss = sim_jw * missing_cand_a
            
            all_name_sims = [sim_jw, sim_sort, t_jw]
            max_n_sim = max(all_name_sims)
            min_n_sim = min(sim_jw, sim_jaccard)
            n_sim_range = max_n_sim - min_n_sim
            a_harmonic = (2.0 * a_jw * a_jaccard) / (a_jw + a_jaccard + 1e-6)
            
            # 7. Meta & Channel Statistics
            is_s2 = 1.0 if cand_id.startswith('S2') or cand_id.startswith('s2') else 0.0
            is_s3 = 1.0 if cand_id.startswith('S3') or cand_id.startswith('s3') else 0.0
            tf_score = tfidf_scores.get(cand_id, 0.0)
            tf_rank = sorted_ranks.get(cand_id, 99)
            ch_count = ch_counts.get(cand_id, 1)
            
            # Store in feature dictionary
            feats['name_jw'].append(sim_jw)
            feats['name_lev'].append(sim_lev)
            feats['name_jaccard'].append(sim_jaccard)
            feats['name_overlap'].append(sim_overlap)
            feats['name_token_sort_ratio'].append(sim_sort)
            feats['name_token_set_ratio'].append(sim_set)
            feats['name_lcs_ratio'].append(sim_lcs)
            feats['name_first_tok'].append(first_tok_m)
            feats['name_last_tok'].append(last_tok_m)
            feats['name_contains'].append(contains_m)
            feats['name_prefix_len'].append(norm_prefix)
            feats['name_char_3gram_jaccard'].append(char_3g)
            
            feats['translit_jw'].append(t_jw)
            feats['translit_lev'].append(t_lev)
            feats['translit_jaccard'].append(t_jaccard)
            feats['translit_overlap'].append(t_overlap)
            feats['translit_first_tok'].append(t_first)
            feats['translit_char_3gram_jaccard'].append(t_char_3g)
            
            feats['addr_jw'].append(a_jw)
            feats['addr_lev'].append(a_lev)
            feats['addr_jaccard'].append(a_jaccard)
            feats['addr_overlap'].append(a_overlap)
            feats['addr_token_sort_ratio'].append(a_sort)
            feats['addr_token_set_ratio'].append(a_set)
            feats['addr_len_ratio'].append(a_len_r)
            feats['addr_has_both'].append(has_both_a)
            feats['addr_missing_cand'].append(missing_cand_a)
            feats['addr_char_4gram_jaccard'].append(a_4g)
            
            feats['postal_match'].append(pmatch)
            feats['postal_prefix_match'].append(p3match)
            feats['street_num_match'].append(snmatch)
            feats['city_sim'].append(sim_city)
            feats['city_exact_match'].append(city_exact)
            feats['postal_log_freq'].append(postal_l_freq)
            feats['city_log_freq'].append(city_l_freq)
            feats['has_postal_both'].append(has_p_both)
            
            feats['name_len_ratio'].append(name_len_r)
            feats['name_char_diff'].append(name_c_diff)
            feats['name_token_diff'].append(name_t_diff)
            feats['name_log_freq_s1'].append(log_freq_s1)
            feats['name_log_freq_cand'].append(log_freq_cand)
            feats['name_is_generic'].append(is_generic)
            
            feats['name_addr_prod'].append(name_addr_p)
            feats['name_addr_sum'].append(name_addr_s)
            feats['name_postal_prod'].append(name_post_p)
            feats['name_jw_x_addr_missing'].append(name_x_miss)
            feats['max_name_sim'].append(max_n_sim)
            feats['min_name_sim'].append(min_n_sim)
            feats['name_sim_range'].append(n_sim_range)
            feats['addr_sim_harmonic'].append(a_harmonic)
            
            feats['source_is_s2'].append(is_s2)
            feats['source_is_s3'].append(is_s3)
            feats['num_candidates'].append(num_cands_val)
            feats['tfidf_score'].append(tf_score)
            feats['tfidf_rank'].append(tf_rank)
            feats['blocking_channel_count'].append(ch_count)
            
        processed += 1
        if processed % 100000 == 0:
            print(f"    Computed features for {processed:,} / {total_s1:,} entities ({time.time() - t0:.1f}s)...", flush=True)
            
    # Build Polars DataFrame
    col_dict = {
        's1_id': s1_id_list,
        'cand_id': cand_id_list,
    }
    if label_list:
        col_dict['label'] = label_list
        
    for k, v in feats.items():
        col_dict[k] = v
        
    out_df = pl.DataFrame(col_dict)
    
    print(f"  [Cache SAVE] Saving feature matrix ({len(out_df):,} pairs) to {cache_path}...", flush=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out_df.write_parquet(cache_path)
    print(f"  [DONE] Feature extraction for {country} complete in {time.time() - t0:.1f}s", flush=True)
    
    return out_df
