"""
Configuration constants for the Entity Resolution Pipeline (v3 Blueprint).
Amazon ML Challenge 2026 - Business Entity Resolution.
Optimized for RTX 4050 GPU, Polars, DuckDB, RapidFuzz, and Macro F0.5.
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Training files
TRAIN_S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Test files
TEST_S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

# Output files
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Cache files (intermediate results)
CACHE_PREPROCESSED = os.path.join(CACHE_DIR, "preprocessed_{split}_{source}.parquet")
CACHE_CANDIDATES = os.path.join(CACHE_DIR, "candidates_{split}_{country}.parquet")
CACHE_FEATURES = os.path.join(CACHE_DIR, "features_{split}_{country}.parquet")
CACHE_EMBEDDINGS = os.path.join(CACHE_DIR, "embeddings_{split}_{source}.npy")
CACHE_MODEL = os.path.join(CACHE_DIR, "lgbm_model.txt")
CACHE_CATBOOST = os.path.join(CACHE_DIR, "catboost_model.cbm")
CACHE_SINGLETON_MODEL = os.path.join(CACHE_DIR, "singleton_classifier.joblib")

# ============================================================
# BLOCKING PARAMETERS (14-CHANNEL SUPER-BLOCKING)
# ============================================================
# Primary Character N-gram TF-IDF on entities
TFIDF_ANALYZER = 'char_wb'
TFIDF_NGRAM_RANGE = (3, 5)          # Character 3-5 gram range
TFIDF_MAX_FEATURES = 150_000        # Vocabulary size
TFIDF_MIN_DF = 2                    # Prune 1-off noise n-grams
TFIDF_MAX_DF = 0.15                 # Prune ubiquitous corporate suffixes (pvt, ltd, inc, corp, llc)
TFIDF_TOP_K = 20                    # Top-K candidates per S1 entity
TFIDF_MIN_SIMILARITY = 0.28         # Minimum cosine similarity (rejects suffix-only noise)
TFIDF_SUBLINEAR_TF = True

MAX_CANDS_PER_S1 = 180              # Max candidate pool size per S1 (optimized for 94%+ recall)

# ============================================================
# FEATURE ENGINEERING (56 FEATURES)
# ============================================================
FEATURE_COLUMNS = [
    # 1. Name String Similarity (12 features)
    'name_jw', 'name_lev', 'name_jaccard', 'name_overlap',
    'name_token_sort_ratio', 'name_token_set_ratio', 'name_lcs_ratio',
    'name_first_tok', 'name_last_tok', 'name_contains',
    'name_prefix_len', 'name_char_3gram_jaccard',
    
    # 2. Transliterated Name Similarity (6 features)
    'translit_jw', 'translit_lev', 'translit_jaccard',
    'translit_overlap', 'translit_first_tok', 'translit_char_3gram_jaccard',
    
    # 3. Address String Similarity (10 features)
    'addr_jw', 'addr_lev', 'addr_jaccard', 'addr_overlap',
    'addr_token_sort_ratio', 'addr_token_set_ratio', 'addr_len_ratio',
    'addr_has_both', 'addr_missing_cand', 'addr_char_4gram_jaccard',
    
    # 4. Structured Geographic Signals (8 features)
    'postal_match', 'postal_prefix_match', 'street_num_match',
    'city_sim', 'city_exact_match', 'postal_log_freq', 'city_log_freq',
    'has_postal_both',
    
    # 5. Entity Structure & Frequency (6 features)
    'name_len_ratio', 'name_char_diff', 'name_token_diff',
    'name_log_freq_s1', 'name_log_freq_cand', 'name_is_generic',
    
    # 6. Cross-Field Interactions (8 features)
    'name_addr_prod', 'name_addr_sum', 'name_postal_prod',
    'name_jw_x_addr_missing', 'max_name_sim', 'min_name_sim',
    'name_sim_range', 'addr_sim_harmonic',
    
    # 7. Meta & Channel Statistics (6 features)
    'source_is_s2', 'source_is_s3', 'num_candidates',
    'tfidf_score', 'tfidf_rank', 'blocking_channel_count',
]

# Monotonic constraints: features that can only increase match confidence
MONOTONE_CONSTRAINTS = {
    'name_jw': 1, 'name_lev': 1, 'name_jaccard': 1,
    'addr_jw': 1, 'addr_jaccard': 1, 'postal_match': 1,
}

# ============================================================
# MODEL PARAMETERS
# ============================================================
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': 10,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 40,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'scale_pos_weight': 0.8,        # Guardrail: precision priority for F0.5 (penalizes false positives)
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

LGBM_NUM_ROUNDS = 2500
LGBM_EARLY_STOPPING = 60
HARD_NEGATIVES_PER_POSITIVE = 5     # Top hard negatives from blocking per positive

# ============================================================
# POST-PROCESSING & CARDINALITY REGULARIZATION
# ============================================================
# Match distribution priors from empirical forensic EDA
MAX_MATCHES_PER_ENTITY = 8          # Mode is 3, 98.9% <= 7, max ever observed was 11
MIN_PROB_THRESHOLD = 0.40           # Baseline classification cutoff
SINGLETON_CUTOFF_PROB = 0.40        # Entities where max(p) < cutoff are predicted empty (protects 123k singletons)
SINGLETON_GAP_THRESHOLD = 0.12      # Disambiguation gap between top-1 and top-2

# Per-country thresholds calibrated around optimal threshold (tau = 0.690)
COUNTRY_THRESHOLDS = {
    'US': 0.70,
    'India': 0.67,
    'France': 0.69,
}

# ============================================================
# GENERAL DATASET & LANGUAGE CONFIGURATION
# ============================================================
COUNTRIES_TRAIN = ['US', 'India']
COUNTRIES_TEST = ['US', 'India', 'France']

# Stopwords (English + French)
STOPWORDS = {
    # English
    'the', 'of', 'and', 'a', 'in', 'for', 'to', 'at', 'on', 'by', 'an', 'or',
    'is', 'it', 'with', 'from', 'llc', 'inc', 'corp', 'co', 'center', 'group',
    'services', 'ltd', 'pvt', 'limited', 'enterprises', 'company',
    # French
    'de', 'du', 'la', 'le', 'les', 'des', 'et', 'en', 'au', 'aux', 'sur',
    'sous', 'pres', 'dans', 'par', 'pour', 'avec', 'sa', 'sarl', 'sas',
    'eurl', 'sci', 'cie', 'societe',
}

# French Street / Address designators
FRENCH_STREET_WORDS = {
    'rue', 'boulevard', 'bd', 'avenue', 'av', 'allee', 'impasse',
    'chemin', 'place', 'passage', 'square', 'route', 'rte', 'quai',
    'cours', 'pont', 'rond', 'point', 'chausee', 'parc',
}

# Corporate legal suffixes across US, India, France (sorted desc for greedy match)
LEGAL_SUFFIXES = [
    # Multi-word legal suffixes
    'private limited', 'pvt limited', 'pvt. ltd.', 'pvt ltd',
    'incorporated', 'corporation', 'limited liability company',
    'limited', 'company', 'enterprises', 'associates',
    'societe anonyme', 'auto entrepreneur',
    # Short / abbreviated
    'pvt.', 'pvt', 'private', 'ltd.', 'ltd', 'l.l.c.', 'l.l.p.',
    'llc', 'llp', 'inc.', 'inc', 'corp.', 'corp', 'co.', 'co',
    # French
    's.a.r.l.', 'sarl', 's.a.s.', 'sasu', 'sas',
    'eurl', 's.a.', 'sci', 'sa', 'cie', 'snc', 'gie', 'ei',
]

# Indian state abbreviations
INDIAN_STATE_ABBREVS = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam',
    'br': 'bihar', 'cg': 'chhattisgarh', 'dl': 'delhi',
    'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana',
    'hp': 'himachal pradesh', 'jh': 'jharkhand', 'ka': 'karnataka',
    'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra',
    'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram',
    'nl': 'nagaland', 'od': 'odisha', 'pb': 'punjab',
    'rj': 'rajasthan', 'sk': 'sikkim', 'tn': 'tamil nadu',
    'tg': 'telangana', 'tr': 'tripura', 'up': 'uttar pradesh',
    'uk': 'uttarakhand', 'wb': 'west bengal',
}

# US state abbreviations
US_STATE_ABBREVS = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut',
    'de': 'delaware', 'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii',
    'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine',
    'md': 'maryland', 'ma': 'massachusetts', 'mi': 'michigan',
    'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri',
    'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico',
    'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota',
    'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon',
    'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas',
    'ut': 'utah', 'vt': 'vermont', 'va': 'virginia',
    'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin',
    'wy': 'wyoming', 'dc': 'district of columbia',
}

# Address abbreviation mappings
ADDRESS_ABBREVS = {
    r'\bst\b': 'street', r'\brd\b': 'road', r'\bave\b': 'avenue',
    r'\bav\b': 'avenue', r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive',
    r'\bln\b': 'lane', r'\bct\b': 'court', r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway', r'\bhwy\b': 'highway', r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor', r'\bste\b': 'suite',
    # French
    r'\br\.\b': 'rue', r'\bbd\b': 'boulevard', r'\ball\b': 'allee',
    r'\bimp\b': 'impasse', r'\bav\.\b': 'avenue',
}

# Validation settings
VAL_SPLIT_RATIO = 0.15              # 15% holdout
RANDOM_SEED = 42
