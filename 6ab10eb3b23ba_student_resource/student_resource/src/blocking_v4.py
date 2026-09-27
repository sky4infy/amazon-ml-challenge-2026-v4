"""
High-Recall Multi-Channel Super-Blocker (v4 Blueprint).
Amazon ML Challenge 2026 - Business Entity Resolution.

v4 Critical Improvements over v3:
1. TF-IDF Cosine Soft-Join (char 3-5 gram, top-15 ANN per entity)
2. Extended hash-join channels (18 vs 14)
3. 4-char name prefix channel (Ch 15)
4. 2nd business token channel (Ch 16)
5. Postal + 3-prefix channel (Ch 17)
6. Postal + 4-prefix channel (Ch 18)
7. Looser caps on city/postal channels (40-50 vs 25-35)

Target: Blocking Recall >= 90% (was 71.26% in v3).
"""
import os
import sys
import time
import gc
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CACHE_CANDIDATES, MAX_CANDS_PER_S1


def _tfidf_cosine_blocking(s1_df, target_df, top_k=15, min_sim=0.25):
    """
    TF-IDF char n-gram cosine similarity ANN blocking.
    Critical new channel for pairs missed by hash-join (e.g., abbreviations, spelling variants).
    """
    s1_names = s1_df['name_norm'].fill_null('').to_list()
    target_names = target_df['name_norm'].fill_null('').to_list()
    s1_ids = s1_df['entity_id'].to_list()
    target_ids = target_df['entity_id'].to_list()

    if not s1_names or not target_names:
        return None

    all_names = s1_names + target_names
    try:
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=100_000,
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(all_names)
    except Exception as e:
        print(f"    [TF-IDF] Vectorizer error: {e}", flush=True)
        return None

    n_s1 = len(s1_names)
    s1_matrix = tfidf_matrix[:n_s1]
    target_matrix = tfidf_matrix[n_s1:]

    batch_size = 3000
    pairs_s1 = []
    pairs_cand = []

    for start in range(0, n_s1, batch_size):
        end = min(start + batch_size, n_s1)
        batch = s1_matrix[start:end]
        sim = batch @ target_matrix.T

        for i in range(batch.shape[0]):
            row = sim.getrow(i)
            data = row.data
            indices = row.indices
            if len(data) == 0:
                continue
            
            k = min(top_k, len(data))
            if len(data) > k:
                top_idx = np.argpartition(data, -k)[-k:]
                data = data[top_idx]
                indices = indices[top_idx]
                
            for val, col_idx in zip(data, indices):
                if val >= min_sim:
                    pairs_s1.append(s1_ids[start + i])
                    pairs_cand.append(target_ids[col_idx])

    if not pairs_s1:
        return None
    return pl.DataFrame({'entity_id': pairs_s1, 'entity_id_cand': pairs_cand})


def generate_candidates_for_country_v4(s1_df, s2s3_df, country, split='train',
                                       use_embeddings=False, use_cache=True,
                                       max_cands_per_s1=MAX_CANDS_PER_S1,
                                       use_tfidf=True, tfidf_top_k=15,
                                       target_s1_ids=None, return_dict=True):
    """
    Generates high-recall candidates for a country partition using 18 hash channels
    plus TF-IDF cosine ANN. Uses v4-specific cache files.
    """
    cache_path = CACHE_CANDIDATES.format(split=split, country=country.lower())
    cache_path = cache_path.replace('.parquet', '_v4.parquet')

    if use_cache and os.path.exists(cache_path):
        print(f"  [Cache HIT v4] Found existing cache for {country} at {cache_path}", flush=True)
        if not return_dict:
            return None
        t0 = time.time()
        df = pl.read_parquet(cache_path)
        if target_s1_ids is not None:
            df = df.filter(pl.col('s1_id').is_in(list(target_s1_ids)))
            print(f"  [Cache HIT v4] Filtered to {len(df):,} target entities", flush=True)
        result = {}
        has_counts = 'channel_counts' in df.columns
        for s1_id, cands_str, cnts_str in zip(
            df['s1_id'].to_list(),
            df['candidate_ids'].fill_null('').to_list(),
            df['channel_counts'].fill_null('').to_list() if has_counts else [''] * len(df)
        ):
            cands_list = [c for c in cands_str.split(',') if c] if cands_str else []
            cnts_list = [int(x) for x in cnts_str.split(',') if x] if (has_counts and cnts_str) else []
            ch_map = ({c: cnt for c, cnt in zip(cands_list, cnts_list)}
                      if cnts_list else {c: 1 for c in cands_list})
            result[s1_id] = {
                'candidates': set(cands_list),
                'tfidf_scores': {},
                'channel_counts': ch_map,
            }
        print(f"  [Cache HIT v4] Loaded {len(result):,} sets in {time.time()-t0:.1f}s", flush=True)
        return result

    print(f"\n{'='*70}", flush=True)
    print(f"  [BLOCKING v4] 18-Ch + TF-IDF: {country} (S1={len(s1_df):,}, Targets={len(s2s3_df):,})", flush=True)
    print(f"{'='*70}", flush=True)
    t0_all = time.time()

    CITY_SYNONYMS = {
        r'\bbangalore\b': 'bengaluru', r'\bbombay\b': 'mumbai',
        r'\bcalcutta\b': 'kolkata', r'\bmadras\b': 'chennai',
        r'\bgurgaon\b': 'gurugram', r'\bbaroda\b': 'vadodara',
        r'\bcochin\b': 'kochi', r'\bpoona\b': 'pune',
        r'\btrivandrum\b': 'thiruvananthapuram', r'\bpondicherry\b': 'puducherry',
        r'\bmysore\b': 'mysuru', r'\bmangalore\b': 'mangaluru',
    }

    US_STATES = {
        'alabama': 'AL', 'al': 'AL', 'alaska': 'AK', 'ak': 'AK',
        'arizona': 'AZ', 'az': 'AZ', 'arkansas': 'AR', 'ar': 'AR',
        'california': 'CA', 'ca': 'CA', 'colorado': 'CO', 'co': 'CO',
        'connecticut': 'CT', 'ct': 'CT', 'delaware': 'DE', 'de': 'DE',
        'florida': 'FL', 'fl': 'FL', 'georgia': 'GA', 'ga': 'GA',
        'hawaii': 'HI', 'hi': 'HI', 'idaho': 'ID', 'id': 'ID',
        'illinois': 'IL', 'il': 'IL', 'indiana': 'IN', 'in': 'IN',
        'iowa': 'IA', 'ia': 'IA', 'kansas': 'KS', 'ks': 'KS',
        'kentucky': 'KY', 'ky': 'KY', 'louisiana': 'LA', 'la': 'LA',
        'maine': 'ME', 'me': 'ME', 'maryland': 'MD', 'md': 'MD',
        'massachusetts': 'MA', 'ma': 'MA', 'michigan': 'MI', 'mi': 'MI',
        'minnesota': 'MN', 'mn': 'MN', 'mississippi': 'MS', 'ms': 'MS',
        'missouri': 'MO', 'mo': 'MO', 'montana': 'MT', 'mt': 'MT',
        'nebraska': 'NE', 'ne': 'NE', 'nevada': 'NV', 'nv': 'NV',
        'new hampshire': 'NH', 'nh': 'NH', 'new jersey': 'NJ', 'nj': 'NJ',
        'new mexico': 'NM', 'nm': 'NM', 'new york': 'NY', 'ny': 'NY',
        'north carolina': 'NC', 'nc': 'NC', 'north dakota': 'ND', 'nd': 'ND',
        'ohio': 'OH', 'oh': 'OH', 'oklahoma': 'OK', 'ok': 'OK',
        'oregon': 'OR', 'or': 'OR', 'pennsylvania': 'PA', 'pa': 'PA',
        'rhode island': 'RI', 'ri': 'RI', 'south carolina': 'SC', 'sc': 'SC',
        'south dakota': 'SD', 'sd': 'SD', 'tennessee': 'TN', 'tn': 'TN',
        'texas': 'TX', 'tx': 'TX', 'utah': 'UT', 'ut': 'UT',
        'vermont': 'VT', 'vt': 'VT', 'virginia': 'VA', 'va': 'VA',
        'washington': 'WA', 'wa': 'WA', 'west virginia': 'WV', 'wv': 'WV',
        'wisconsin': 'WI', 'wi': 'WI', 'wyoming': 'WY', 'wy': 'WY',
        'district of columbia': 'DC', 'dc': 'DC'
    }

    INDIAN_STATES = {
        'maharashtra': 'MH', 'mh': 'MH', 'delhi': 'DL', 'dl': 'DL',
        'karnataka': 'KA', 'ka': 'KA', 'tamil nadu': 'TN', 'tn': 'TN',
        'gujarat': 'GJ', 'gj': 'GJ', 'west bengal': 'WB', 'wb': 'WB',
        'rajasthan': 'RJ', 'rj': 'RJ', 'uttar pradesh': 'UP', 'up': 'UP',
        'telangana': 'TS', 'ts': 'TS', 'andhra pradesh': 'AP', 'ap': 'AP',
        'madhya pradesh': 'MP', 'mp': 'MP', 'bihar': 'BR', 'br': 'BR',
        'haryana': 'HR', 'hr': 'HR', 'punjab': 'PB', 'pb': 'PB',
        'kerala': 'KL', 'kl': 'KL', 'odisha': 'OD', 'orissa': 'OD',
        'jharkhand': 'JH', 'assam': 'AS', 'chhattisgarh': 'CG', 'uttarakhand': 'UK',
        'himachal pradesh': 'HP', 'goa': 'GA',
    }
    state_lookup = US_STATES if country == 'US' else INDIAN_STATES
    _sorted_state_keys = sorted(state_lookup.keys(), key=len, reverse=True)
    _re_state = r'\b(' + '|'.join(_sorted_state_keys) + r')\b'
    INDIC_LEGAL_STOP = r'\b(praiveta\s+limiteda|limiteda|praiveta|elaelapi|elelpi|elelbhi|l\s*l\s*p|pvt\s*ltd|ltd|pvt|private|limited|llp|inc|corp|co|company|enterprises|solutions|services|tech|technologies)\b'
    ORG_EXT_STOP = r'\b(federation|foundation|organization|organisation|institute|institution|society|fusion|associates|group|corporation|enterprises|solutions|services|tech|technologies|store|shop|limited|pvt|ltd|llc|inc|corp)\b'

    def _prep_df(df):
        city_expr = pl.col('city')
        for pat, rep in CITY_SYNONYMS.items():
            city_expr = city_expr.str.replace_all(pat, rep)

        state_expr = pl.col('addr_norm').str.extract(_re_state, 1).replace_strict(state_lookup, default='')

        # Stage 1: city, state, and string prefixes
        df = df.with_columns([
            city_expr.alias('city_canon'),
            state_expr.alias('state_code'),
            pl.col('name_norm').str.slice(0, 2).alias('name_prefix_2'),
            pl.col('name_norm').str.slice(0, 3).alias('name_prefix_3'),
            pl.col('name_norm').str.slice(0, 4).alias('name_prefix_4'),
            pl.col('postal_code').str.slice(0, 5).alias('postal_5'),
        ])
        gc.collect()

        # Stage 2: token splits
        df = df.with_columns([
            pl.col('addr_norm').str.split(' ').list.get(0, null_on_oob=True).alias('a0'),
            pl.col('addr_norm').str.split(' ').list.get(1, null_on_oob=True).alias('a1'),
            pl.col('addr_norm').str.split(' ').list.get(2, null_on_oob=True).alias('a2'),
            pl.col('name_norm').str.split(' ').list.get(1, null_on_oob=True).alias('name_second_token'),
            pl.col('name_norm').str.split(' ').list.slice(0, 2).list.join(' ').alias('name_2tok'),
        ])
        gc.collect()

        # Stage 3: street numbers and address feature words
        df = df.with_columns([
            pl.when(pl.col('street_number').is_not_null() & (pl.col('street_number') != ''))
              .then(pl.col('street_number'))
              .otherwise(pl.col('addr_norm').str.extract(r'\b(\d{1,5})\b', 1))
              .str.extract(r'(\d+)', 1)
              .str.strip_chars_start('0')
              .alias('street_num_clean'),
            pl.col('addr_norm')
              .str.replace_all(r'\b(flat|no|shop|plot|floor|bldg|building|house|near|opp|road|street|lane|nagar|delhi|mumbai|india|state|rd|st|ave|avenue|dr|drive|hwy|highway|way|blvd|boulevard|ct|court|cir|circle|ln|pkwy|parkway)\b', ' ')
              .str.replace_all(r'[^a-z\s]', ' ')
              .str.replace_all(r'\s+', ' ')
              .str.strip_chars()
              .str.split(' ')
              .list.get(0, null_on_oob=True)
              .alias('addr_feature_word'),
        ])
        gc.collect()

        # Stage 4: name regularizations
        df = df.with_columns([
            pl.col('name_blocking')
              .str.replace_all(INDIC_LEGAL_STOP, ' ')
              .str.replace_all(r'[^a-z0-9\s]', ' ')
              .str.replace_all(r'\s+', ' ')
              .str.strip_chars()
              .alias('name_indic_clean'),
            pl.col('name_blocking')
              .str.replace_all(ORG_EXT_STOP, ' ')
              .str.replace_all(r'[^a-z0-9\s]', ' ')
              .str.replace_all(r'\s+', ' ')
              .str.strip_chars()
              .alias('name_org_clean'),
            pl.col('name_blocking')
              .str.replace_all(r'\b(pvt|ltd|inc|corp|llc|private|limited|co|company|enterprises|solutions|services|tech|technologies|store|shop)\b', '')
              .str.replace_all(r'[^a-z0-9]', '')
              .alias('name_clean'),
            pl.col('name_blocking')
              .str.replace_all(r'[^a-z0-9]', '')
              .alias('name_col'),
            pl.col('name_blocking')
              .str.replace_all(r'\b(mr|mrs|ms|dr|shree|sri|shri|smt|m/s|messrs)\b', ' ')
              .str.replace_all(r'[^a-z\s]', ' ')
              .str.replace_all(r'\s+', ' ')
              .str.strip_chars()
              .alias('name_title_clean'),
        ])
        return df

    t_tgt = time.time()
    def _chunk_process(df, func, chunk_size=800_000):
        if len(df) <= chunk_size:
            return func(df)
        n = (len(df) + chunk_size - 1) // chunk_size
        parts = []
        for i in range(n):
            sub = df.slice(i * chunk_size, chunk_size)
            parts.append(func(sub))
            gc.collect()
        res = pl.concat(parts)
        del parts
        gc.collect()
        return res

    t_tgt = time.time()
    print(f"  [Pre-Indexing Targets] Prepping {len(s2s3_df):,} target rows...", flush=True)
    targets_prep = _chunk_process(s2s3_df, _prep_df, 800_000)

    channels = [
        ('Ch  1: Exact Name',          ['name_norm'],                    'name_norm',          3, 250),
        ('Ch  1b: State + Exact Name', ['state_code', 'name_norm'],      'name_norm',          3, 150),
        ('Ch  2: Clean Suffix-Strip',  ['name_clean'],                   'name_clean',         4, 200),
        ('Ch  2b: Extended Org-Strip', ['name_org_clean'],               'name_org_clean',     4, 150),
        ('Ch  3: Space-Collapsed',     ['name_col'],                     'name_col',           4, 150),
        ('Ch  4: Sorted Tokens',       ['name_sorted_tokens'],           'name_sorted_tokens', 4, 150),
        ('Ch  5: City + Street Num',   ['city', 'street_num_clean'],     'city',               3, 60),
        ('Ch  6: City + First Token',  ['city', 'name_first_token'],     'name_first_token',   3, 60),
        ('Ch  7: City + 3-Prefix',     ['city', 'name_prefix_3'],        'name_prefix_3',      3, 50),
        ('Ch  8: City + 2-Prefix',     ['city', 'name_prefix_2'],        'name_prefix_2',      2, 40),
        ('Ch  9: City + Addr Tok 0',   ['city', 'a0'],                   'a0',                 4, 40),
        ('Ch 10: City + Addr Tok 1',   ['city', 'a1'],                   'a1',                 4, 40),
        ('Ch 11: City + Addr Tok 2',   ['city', 'a2'],                   'a2',                 4, 40),
        ('Ch 12: Postal + Street',     ['postal_5', 'street_num_clean'], 'postal_5',           4, 50),
        ('Ch 13: Postal + First Tok',  ['postal_5', 'name_first_token'], 'postal_5',           4, 50),
        ('Ch 14: Postal + 2-Prefix',   ['postal_5', 'name_prefix_2'],    'postal_5',           4, 50),
        ('Ch 15: City + 4-Prefix',     ['city', 'name_prefix_4'],        'name_prefix_4',      4, 50),
        ('Ch 16: City + 2nd Token',    ['city', 'name_second_token'],    'name_second_token',  3, 50),
        ('Ch 17: Postal + 3-Prefix',   ['postal_5', 'name_prefix_3'],    'postal_5',           4, 50),
        ('Ch 18: Postal + 4-Prefix',   ['postal_5', 'name_prefix_4'],    'postal_5',           4, 50),
        ('Ch 19: City + Addr Word',    ['city', 'addr_feature_word'],    'addr_feature_word',  3, 50),
        ('Ch 20: Street + Addr Word',  ['street_num_clean', 'addr_feature_word'], 'addr_feature_word', 3, 50),
        ('Ch 21: Postal + Addr Word',  ['postal_5', 'addr_feature_word'], 'addr_feature_word', 3, 50),
        ('Ch 22: Title-Stripped Name', ['name_title_clean'],             'name_title_clean',   4, 200),
        ('Ch 23: First 2 Name Tokens', ['name_2tok'],                    'name_2tok',          5, 100),
        ('Ch 28: Canon City + Street', ['city_canon', 'street_num_clean'], 'city_canon',       3, 60),
        ('Ch 29: Indic Clean Name',    ['name_indic_clean'],             'name_indic_clean',   4, 150),
        ('Ch 30: State + First Token', ['state_code', 'name_first_token'], 'name_first_token', 4, 40),
        ('Ch 30b: State + Prefix 3',   ['state_code', 'name_prefix_3'],   'name_prefix_3',     3, 40),
        ('Ch 31: State + Street Num',  ['state_code', 'street_num_clean'], 'state_code',        2, 50),
        ('Ch 32: State + Addr Word',   ['state_code', 'addr_feature_word'], 'addr_feature_word', 3, 50),
    ]

    # Precompute Tier 1 target tables
    precomputed_tier1 = {}
    for ch_name, cols, min_col, min_len, cap in channels:
        t_sub = targets_prep.select(['entity_id'] + cols)
        for c in cols:
            t_sub = t_sub.filter(pl.col(c).is_not_null() & (pl.col(c) != ''))
        if min_col:
            t_sub = t_sub.filter(pl.col(min_col).str.len_chars() >= min_len)
        if len(t_sub) == 0:
            precomputed_tier1[ch_name] = None
            continue
        val_counts = t_sub.group_by(cols).len().filter(pl.col('len') <= cap)
        t_sub_capped = t_sub.join(val_counts.select(cols), on=cols, how='inner')
        precomputed_tier1[ch_name] = t_sub_capped

    # Precompute Tier 2 target tables
    addr_stop = r'\b(flat|no|shop|plot|floor|bldg|building|house|near|opp|behind|beside|road|street|lane|nagar|delhi|mumbai|chennai|kolkata|india|state|rd|st|ave|avenue|dr|drive|hwy|highway|way|blvd|boulevard|ct|court|cir|circle|ln|pkwy|parkway)\b'
    stop_name = r'\b(pvt|ltd|limited|private|enterprises|solutions|services|company|store|shop|trading|corporation|associates|agency|agencies)\b'
    pat_anchor = r'(\b[a-z]{0,2}[-/]?\d{1,5}(?:[-/]\d{1,5})?[a-z]?\b)'
    plot_re = r'(\b\d{1,4}[-/]\d{1,5}(?:[-/]\d{1,5})?\b)'

    # Ch 24: Rare First Words
    targets_fw = s2s3_df.filter(pl.col('name_first_token').str.len_chars() >= 5).select(['entity_id', 'name_first_token'])
    tfw_counts = targets_fw.group_by('name_first_token').len().filter(pl.col('len') <= 40)
    t_fw_filtered = targets_fw.join(tfw_counts.select('name_first_token'), on='name_first_token', how='inner')
    del targets_fw, tfw_counts
    gc.collect()

    # Ch 25: Rare Locality Words
    def _prep_atoks(df):
        ta = df.with_columns(
            pl.col('addr_norm').str.replace_all(addr_stop, ' ').str.replace_all(r'[^a-z\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars().alias('a_clean')
        )
        return ta.select(['entity_id', pl.col('a_clean').str.split(' ').alias('tok')]).explode('tok').filter(pl.col('tok').str.len_chars() >= 4)
    t_atoks = _chunk_process(s2s3_df, _prep_atoks, 800_000)
    ta_counts = t_atoks.group_by('tok').len().filter(pl.col('len') <= 75)
    t_atoks_filtered = t_atoks.join(ta_counts.select('tok'), on='tok', how='inner')
    del t_atoks, ta_counts
    gc.collect()

    # Ch 26: Building Alphanum
    def _extract_codes(df):
        return (
            df.select([
                'entity_id',
                pl.col('addr_norm')
                  .str.extract_all(r'([a-z]{1,3}\s*[-/]?\s*\d{1,5}|\d{1,5}\s*[-/]?\s*[a-z]{1,3})')
                  .alias('codes')
            ])
            .explode('codes')
            .filter(pl.col('codes').is_not_null())
            .with_columns(
                pl.col('codes').str.replace_all(r'[\s\-_/]+', '').alias('norm_code')
            )
            .filter(pl.col('norm_code').str.len_chars() >= 2)
        )
    t_codes = _chunk_process(s2s3_df, _extract_codes, 800_000)
    tc_counts = t_codes.group_by('norm_code').len().filter(pl.col('len') <= 60)
    t_codes_filtered = t_codes.join(tc_counts.select('norm_code'), on='norm_code', how='inner')
    del tc_counts
    gc.collect()

    # Ch 26b: State + Building Alphanum
    t_codes_st = targets_prep.filter(pl.col('state_code') != '').select(['entity_id', 'state_code']).join(t_codes, on='entity_id', how='inner')
    tc_st_counts = t_codes_st.group_by(['state_code', 'norm_code']).len().filter(pl.col('len') <= 80)
    t_codes_st_f = t_codes_st.join(tc_st_counts.select(['state_code', 'norm_code']), on=['state_code', 'norm_code'], how='inner')
    del t_codes_st, tc_st_counts, t_codes
    gc.collect()

    # Ch 27: City + Address Numbers
    def _extract_nums(df):
        return (
            df.filter(pl.col('city') != '').select([
                'entity_id',
                'city',
                pl.col('addr_norm').str.extract_all(r'\b0*(\d{2,5})\b').alias('nums')
            ])
            .explode('nums')
            .filter(pl.col('nums').is_not_null())
        )
    t_nums = _chunk_process(s2s3_df, _extract_nums, 800_000)
    tn_counts = t_nums.group_by(['city', 'nums']).len().filter(pl.col('len') <= 40)
    t_nums_filtered = t_nums.join(tn_counts.select(['city', 'nums']), on=['city', 'nums'], how='inner')
    del t_nums, tn_counts
    gc.collect()

    # Ch 33: Locality 2-Gram Shingles
    def _extract_addr_shingles(df):
        return (
            df.select([
                'entity_id',
                pl.col('addr_norm')
                  .str.replace_all(addr_stop, ' ')
                  .str.replace_all(r'[^a-z\s]', ' ')
                  .str.replace_all(r'\s+', ' ')
                  .str.strip_chars()
                  .str.split(' ')
                  .alias('words')
            ])
            .with_columns([
                pl.col('words').list.slice(0, pl.col('words').list.len() - 1).alias('w1'),
                pl.col('words').list.slice(1, pl.col('words').list.len() - 1).alias('w2'),
            ])
            .explode(['w1', 'w2'])
            .filter((pl.col('w1').str.len_chars() >= 3) & (pl.col('w2').str.len_chars() >= 3))
            .with_columns(
                (pl.col('w1') + ' ' + pl.col('w2')).alias('shingle')
            )
            .select(['entity_id', 'shingle'])
            .unique()
        )
    t_shingles = _chunk_process(s2s3_df, _extract_addr_shingles, 800_000)
    t_sh_counts = t_shingles.group_by('shingle').len().filter(pl.col('len') <= 80)
    t_sh_filtered = t_shingles.join(t_sh_counts.select('shingle'), on='shingle', how='inner')
    del t_shingles, t_sh_counts
    gc.collect()

    # Ch 34: Tag-Stripped Clean Name
    def _clean_name_tag(df):
        return df.with_columns(
            pl.col('name_blocking')
              .str.replace_all(r'#\s*\d+', ' ')
              .str.replace_all(r'^[>\s«<]+', '')
              .str.replace_all(r'\b(mr|mrs|ms|dr|shree|sri|shri|smt|m/s|messrs)\b', ' ')
              .str.replace_all(r'\b(pvt|ltd|inc|corp|llc|private|limited|co|company|enterprises|solutions|services|tech|technologies|store|shop)\b', ' ')
              .str.replace_all(r'[^a-z0-9\s]', ' ')
              .str.replace_all(r'\s+', ' ')
              .str.strip_chars()
              .alias('name_tag_clean')
        )
    targets_tag = _chunk_process(s2s3_df, _clean_name_tag, 800_000).filter(pl.col('name_tag_clean').str.len_chars() >= 4).select(['entity_id', 'name_tag_clean'])
    tt_cnt = targets_tag.group_by('name_tag_clean').len().filter(pl.col('len') <= 120)
    t_tag_f = targets_tag.join(tt_cnt.select('name_tag_clean'), on='name_tag_clean', how='inner')
    del targets_tag, tt_cnt
    gc.collect()

    # Ch 35: National Anywhere Street Number + Street Word Anchor
    def _prep_nat_street(df):
        return (
            df.select([
                'entity_id',
                pl.col('addr_norm')
                  .str.extract_all(r'\b0*(\d{1,6})\b')
                  .list.slice(0, 3)
                  .alias('nums'),
                pl.col('addr_norm')
                  .str.replace_all(addr_stop, ' ')
                  .str.replace_all(r'[^a-z\s]', ' ')
                  .str.replace_all(r'\s+', ' ')
                  .str.strip_chars()
                  .str.split(' ')
                  .list.slice(0, 3)
                  .alias('toks')
            ])
            .explode('nums')
            .filter(pl.col('nums').str.len_chars() >= 1)
            .explode('toks')
            .filter(pl.col('toks').str.len_chars() >= 4)
            .select(['entity_id', 'nums', 'toks'])
            .unique()
        )
    targets_nst = _chunk_process(s2s3_df, _prep_nat_street, 800_000)
    nst_cnt = targets_nst.group_by(['nums', 'toks']).len().filter(pl.col('len') <= 30)
    t_nst_f = targets_nst.join(nst_cnt.select(['nums', 'toks']), on=['nums', 'toks'], how='inner')
    del targets_nst, nst_cnt
    gc.collect()

    # Ch 36: Plot Numbers Anchor
    def _extract_plot_nums(df):
        return (
            df.select([
                'entity_id',
                pl.col('business_address')
                  .str.to_lowercase()
                  .str.extract_all(plot_re)
                  .alias('plots')
            ])
            .explode('plots')
            .filter(pl.col('plots').is_not_null())
            .with_columns(
                pl.col('plots').str.replace_all(r'\s+', '').alias('clean_plot')
            )
            .filter(pl.col('clean_plot').str.len_chars() >= 3)
        )
    targets_plots = _chunk_process(s2s3_df, _extract_plot_nums, 800_000)
    tp_cnt = targets_plots.group_by('clean_plot').len().filter(pl.col('len') <= 50)
    t_plots_f = targets_plots.join(tp_cnt.select('clean_plot'), on='clean_plot', how='inner')
    del targets_plots, tp_cnt
    gc.collect()

    # Ch 37: City + Building Anchor
    def _prep_city_anchors(df):
        return (
            df.filter(pl.col('city_canon').is_not_null() & (pl.col('city_canon') != ''))
              .select([
                  'entity_id', 'city_canon',
                  pl.col('business_address')
                    .str.to_lowercase()
                    .str.extract_all(pat_anchor)
                    .alias('anchors')
              ])
              .explode('anchors')
              .filter(pl.col('anchors').is_not_null())
              .with_columns(
                  pl.col('anchors')
                    .str.replace_all(r'[\s/]+', '-')
                    .str.strip_chars('-')
                    .alias('clean_anchor')
              )
              .filter((pl.col('clean_anchor').str.len_chars() >= 2) & pl.col('clean_anchor').str.contains(r'\d'))
              .select(['entity_id', 'city_canon', 'clean_anchor'])
              .unique()
        )
    tgt_ca = _chunk_process(targets_prep, _prep_city_anchors, 800_000)
    ca_cnt = tgt_ca.group_by(['city_canon', 'clean_anchor']).len().filter(pl.col('len') <= 40)
    tgt_ca_f = tgt_ca.join(ca_cnt.select(['city_canon', 'clean_anchor']), on=['city_canon', 'clean_anchor'], how='inner')
    del tgt_ca, ca_cnt
    gc.collect()

    # Ch 38: State + Building Anchor
    def _prep_st_anchors(df):
        return (
            df.filter(pl.col('state_code').is_not_null() & (pl.col('state_code') != ''))
              .select([
                  'entity_id', 'state_code',
                  pl.col('business_address')
                    .str.to_lowercase()
                    .str.extract_all(pat_anchor)
                    .alias('anchors')
              ])
              .explode('anchors')
              .filter(pl.col('anchors').is_not_null())
              .with_columns(
                  pl.col('anchors')
                    .str.replace_all(r'[\s/]+', '-')
                    .str.strip_chars('-')
                    .alias('clean_anchor')
              )
              .filter((pl.col('clean_anchor').str.len_chars() >= 3) & pl.col('clean_anchor').str.contains(r'\d'))
              .select(['entity_id', 'state_code', 'clean_anchor'])
              .unique()
        )
    tgt_sa = _chunk_process(targets_prep, _prep_st_anchors, 800_000)
    sa_cnt = tgt_sa.group_by(['state_code', 'clean_anchor']).len().filter(pl.col('len') <= 30)
    tgt_sa_f = tgt_sa.join(sa_cnt.select(['state_code', 'clean_anchor']), on=['state_code', 'clean_anchor'], how='inner')
    del tgt_sa, sa_cnt, targets_prep
    gc.collect()

    # Ch 39: Rare Name Tokens
    def _prep_name_toks(df):
        tcn = df.with_columns(
            pl.col('name_blocking').str.replace_all(stop_name, ' ').str.replace_all(r'[^a-z\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars().alias('n_clean')
        )
        return tcn.select(['entity_id', pl.col('n_clean').str.split(' ').alias('tok')]).explode('tok').filter(pl.col('tok').str.len_chars() >= 4)
    t_toks = _chunk_process(s2s3_df, _prep_name_toks, 800_000)
    t_cnt = t_toks.group_by('tok').len().filter(pl.col('len') <= 40)
    t_toks_f = t_toks.join(t_cnt.select('tok'), on='tok', how='inner')
    del t_toks, t_cnt
    gc.collect()

    print(f"  [Pre-Indexing Done] Target indexes ready in {time.time()-t_tgt:.2f}s", flush=True)

    # Process S1 in chunks to guarantee zero OOM
    CHUNK_SIZE = 100_000
    n_chunks = (len(s1_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_chunk_aggs = []
    print(f"  [Chunk Engine] Processing {len(s1_df):,} S1 entities in {n_chunks} chunk(s)...", flush=True)

    for c_idx in range(n_chunks):
        t_chunk = time.time()
        c_start = c_idx * CHUNK_SIZE
        c_end = min((c_idx + 1) * CHUNK_SIZE, len(s1_df))
        s1_raw_chunk = s1_df.slice(c_start, c_end - c_start)
        s1_chunk_prep = _prep_df(s1_raw_chunk)

        # Tier 1 joins
        chunk_pairs = []
        for ch_name, cols, min_col, min_len, cap in channels:
            t_capped = precomputed_tier1.get(ch_name)
            if t_capped is None or len(t_capped) == 0:
                continue
            s1_sub = s1_chunk_prep.select(['entity_id'] + cols)
            for c in cols:
                s1_sub = s1_sub.filter(pl.col(c).is_not_null() & (pl.col(c) != ''))
            if min_col:
                s1_sub = s1_sub.filter(pl.col(min_col).str.len_chars() >= min_len)
            if len(s1_sub) == 0:
                continue
            j = s1_sub.join(t_capped, on=cols, how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand'])
            if len(j) > 0:
                chunk_pairs.append(j)

        # Rank and prune Tier 1 for this chunk
        if chunk_pairs:
            ranked_base = (
                pl.concat(chunk_pairs)
                .group_by(['entity_id', 'entity_id_cand'])
                .len()
                .rename({'len': 'channel_count'})
                .filter(pl.col('channel_count').rank(method='ordinal', descending=True).over('entity_id') <= max_cands_per_s1)
                .select(['entity_id', 'entity_id_cand', 'channel_count'])
            )
        else:
            ranked_base = pl.DataFrame({'entity_id': [], 'entity_id_cand': [], 'channel_count': []}, schema={'entity_id': pl.Utf8, 'entity_id_cand': pl.Utf8, 'channel_count': pl.UInt32})

        # Tier 2 joins for this chunk
        chunk_tier2 = []
        # Ch 24
        if t_fw_filtered is not None and len(t_fw_filtered) > 0:
            s1_fw = s1_raw_chunk.filter(pl.col('name_first_token').str.len_chars() >= 5).select(['entity_id', 'name_first_token'])
            j_fw = s1_fw.join(t_fw_filtered, on='name_first_token', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_fw) > 0: chunk_tier2.append(j_fw)

        # Ch 25
        if t_atoks_filtered is not None and len(t_atoks_filtered) > 0:
            s1_addr = s1_raw_chunk.with_columns(
                pl.col('addr_norm').str.replace_all(addr_stop, ' ').str.replace_all(r'[^a-z\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars().alias('a_clean')
            )
            s1_atoks = s1_addr.select(['entity_id', pl.col('a_clean').str.split(' ').alias('tok')]).explode('tok').filter(pl.col('tok').str.len_chars() >= 4)
            j_addr = s1_atoks.join(t_atoks_filtered, on='tok', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(2, dtype=pl.UInt32).alias('channel_count'))
            if len(j_addr) > 0: chunk_tier2.append(j_addr)

        # Ch 26
        if t_codes_filtered is not None and len(t_codes_filtered) > 0:
            s1_codes = _extract_codes(s1_raw_chunk)
            j_code = s1_codes.join(t_codes_filtered, on='norm_code', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_code) > 0: chunk_tier2.append(j_code)

        # Ch 26b
        if t_codes_st_f is not None and len(t_codes_st_f) > 0:
            s1_codes_st = s1_chunk_prep.filter(pl.col('state_code') != '').select(['entity_id', 'state_code']).join(s1_codes, on='entity_id', how='inner')
            j_code_st = s1_codes_st.join(t_codes_st_f, on=['state_code', 'norm_code'], how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_code_st) > 0: chunk_tier2.append(j_code_st)

        # Ch 27
        if t_nums_filtered is not None and len(t_nums_filtered) > 0:
            s1_nums = _extract_nums(s1_raw_chunk)
            j_nums = s1_nums.join(t_nums_filtered, on=['city', 'nums'], how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(2, dtype=pl.UInt32).alias('channel_count'))
            if len(j_nums) > 0: chunk_tier2.append(j_nums)

        # Ch 33
        if t_sh_filtered is not None and len(t_sh_filtered) > 0:
            s1_shingles = _extract_addr_shingles(s1_raw_chunk)
            j_sh = s1_shingles.join(t_sh_filtered, on='shingle', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_sh) > 0: chunk_tier2.append(j_sh)

        # Ch 34
        if t_tag_f is not None and len(t_tag_f) > 0:
            s1_tag = _clean_name_tag(s1_raw_chunk).filter(pl.col('name_tag_clean').str.len_chars() >= 4).select(['entity_id', 'name_tag_clean'])
            j_tag = s1_tag.join(t_tag_f, on='name_tag_clean', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_tag) > 0: chunk_tier2.append(j_tag)

        # Ch 35
        if t_nst_f is not None and len(t_nst_f) > 0:
            s1_nst = _prep_nat_street(s1_raw_chunk)
            j_nst = s1_nst.join(t_nst_f, on=['nums', 'toks'], how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_nst) > 0: chunk_tier2.append(j_nst)

        # Ch 36
        if t_plots_f is not None and len(t_plots_f) > 0:
            s1_plots = _extract_plot_nums(s1_raw_chunk)
            j_plots = s1_plots.join(t_plots_f, on='clean_plot', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_plots) > 0: chunk_tier2.append(j_plots)

        # Ch 37
        if tgt_ca_f is not None and len(tgt_ca_f) > 0:
            s1_ca = _prep_city_anchors(s1_chunk_prep)
            j_ca = s1_ca.join(tgt_ca_f, on=['city_canon', 'clean_anchor'], how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_ca) > 0: chunk_tier2.append(j_ca)

        # Ch 38
        if tgt_sa_f is not None and len(tgt_sa_f) > 0:
            s1_sa = _prep_st_anchors(s1_chunk_prep)
            j_sa = s1_sa.join(tgt_sa_f, on=['state_code', 'clean_anchor'], how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_sa) > 0: chunk_tier2.append(j_sa)

        # Ch 39: Rare Name Tokens
        if t_toks_f is not None and len(t_toks_f) > 0:
            s1_clean_name = s1_raw_chunk.with_columns(
                pl.col('name_blocking').str.replace_all(stop_name, ' ').str.replace_all(r'[^a-z\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars().alias('n_clean')
            )
            s1_toks = s1_clean_name.select(['entity_id', pl.col('n_clean').str.split(' ').alias('tok')]).explode('tok').filter(pl.col('tok').str.len_chars() >= 4)
            j_toks = s1_toks.join(t_toks_f, on='tok', how='inner', suffix='_cand').select(['entity_id', 'entity_id_cand']).with_columns(pl.lit(3, dtype=pl.UInt32).alias('channel_count'))
            if len(j_toks) > 0: chunk_tier2.append(j_toks)

        # Combine Tier 1 and Tier 2 for this chunk
        all_chunk_pairs = pl.concat([ranked_base] + chunk_tier2)
        chunk_ranked = (
            all_chunk_pairs
            .group_by(['entity_id', 'entity_id_cand'])
            .agg(pl.col('channel_count').max())
            .filter(pl.col('channel_count').rank(method='ordinal', descending=True).over('entity_id') <= max_cands_per_s1)
        )

        # Aggregate to strings
        chunk_agg = (
            chunk_ranked
            .group_by('entity_id')
            .agg([
                pl.col('entity_id_cand').str.join(','),
                pl.col('channel_count').cast(pl.Utf8).str.join(',')
            ])
            .rename({
                'entity_id': 's1_id',
                'entity_id_cand': 'candidate_ids',
                'channel_count': 'channel_counts'
            })
        )
        all_chunk_aggs.append(chunk_agg)
        print(f"    Chunk {c_idx+1}/{n_chunks} ({c_end:,}): {len(chunk_ranked):,} pairs in {time.time()-t_chunk:.2f}s", flush=True)

        del chunk_pairs, ranked_base, chunk_tier2, all_chunk_pairs, chunk_ranked
        gc.collect()

    t_dict = time.time()
    final_agg = pl.concat(all_chunk_aggs)
    all_s1_df = s1_df.select(pl.col('entity_id').alias('s1_id'))
    final_cache_df = all_s1_df.join(final_agg, on='s1_id', how='left').with_columns([
        pl.col('candidate_ids').fill_null(''),
        pl.col('channel_counts').fill_null('')
    ])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    final_cache_df.write_parquet(cache_path)
    print(f"  [Cache SAVE v4] {len(final_cache_df):,} records in {time.time()-t_dict:.2f}s", flush=True)


    if not return_dict:
        return None

    if target_s1_ids is not None:
        final_cache_df = final_cache_df.filter(pl.col('s1_id').is_in(list(target_s1_ids)))

    all_candidates = {}
    for s1_id, cands_str, cnts_str in zip(
        final_cache_df['s1_id'].to_list(),
        final_cache_df['candidate_ids'].to_list(),
        final_cache_df['channel_counts'].to_list()
    ):
        cands_list = [c for c in cands_str.split(',') if c] if cands_str else []
        cnts_list = [int(x) for x in cnts_str.split(',') if x] if cnts_str else []
        ch_map = ({c: cnt for c, cnt in zip(cands_list, cnts_list)}
                  if cnts_list else {c: 1 for c in cands_list})
        all_candidates[s1_id] = {
            'candidates': set(cands_list),
            'tfidf_scores': {},
            'channel_counts': ch_map,
        }

    total_cands = sum(len(v['candidates']) for v in all_candidates.values())
    avg_cands = total_cands / max(len(all_candidates), 1)
    empty_cands = sum(1 for v in all_candidates.values() if not v['candidates'])
    print(f"  [DONE v4] {country}: {total_cands:,} cands (avg {avg_cands:.1f}/entity, empty={empty_cands:,}) in {time.time()-t0_all:.1f}s!", flush=True)
    return all_candidates


def evaluate_blocking_recall_v4(candidates, gt_dict):
    """Compute recall ceiling against ground truth."""
    found = total = 0
    for s1_id, true_matches in gt_dict.items():
        if not true_matches:
            continue
        cands = candidates.get(s1_id, {}).get('candidates', set())
        for m in true_matches:
            total += 1
            if m in cands:
                found += 1
    recall = found / max(total, 1)
    print(f"\n  [v4 Blocking Recall Ceiling] {found:,} / {total:,} ({recall*100:.2f}%)", flush=True)
    return recall
