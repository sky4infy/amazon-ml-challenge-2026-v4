# Business Entity Resolution Pipeline V4 (Championship Edition)
**Amazon ML Challenge 2026 — Team GradientX**  
**Official Peak Public Leaderboard Benchmark: 0.8380 Macro $F_{0.5}$**

---

## 1. System Overview
This package contains the complete, self-contained, end-to-end entity resolution pipeline developed for the **Amazon ML Challenge 2026**.
The system resolves and links business records across heterogeneous sources (`S1`, `S2`, `S3`) at massive scale (~11.7 million records total) optimizing for **Entity-Level Macro $F_{0.5}$**.

### Pipeline Architecture (V4)
1. **Multilingual Preprocessing & Normalization:**
   - Multi-script detection (Latin, Devanagari, Gurmukhi, Tamil, etc.).
   - Transliteration, punctuation stripping, case folding, legal suffix normalization (`pvt ltd` $\to$ `ltd`, `corp` $\to$ `inc`).
   - Structured subfield extraction: postal/PIN codes, street numbers, cities.
2. **38-Channel High-Recall Super-Blocking Engine:**
   - Tier 1: Exact normalized name, cleaned suffix-strip name, space-collapsed name, sorted tokens.
   - Tier 2: City + street number, city + first token, postal 5-digit + first token, city + prefix 2/3/4.
   - Tier 3: State code + exact name, state code + clean suffix, state code + building/plot codes.
   - Tier 4: City + address shingles (a0, a1, a2), street anchors, natural street names, anywhere street numbers.
   - Verified candidate recall: **97.99% US, 94.62% India, 96.64% Global** across all ground-truth links.
3. **High-Efficiency C++ Accelerated Feature Extraction (56 Features):**
   - RapidFuzz AVX2 SIMD vectorization: Jaro-Winkler, Levenshtein, Token Sort, Jaccard, Simpson overlap.
   - Address token overlap, postal code exact match, street number exact match, cross-field interaction terms (`name_jw * addr_jw`).
4. **Dual-Model GBDT Ensemble:**
   - **LightGBM**: Leaf-wise tree growth trained with monotonic constraints and precision-priority positive weighting.
   - **CatBoost GPU**: Oblivious symmetric trees trained for robust out-of-distribution transfer on unseen countries (France).
   - Soft-voting probability blend: $P = 0.80 \cdot P_{\text{LightGBM}} + 0.20 \cdot P_{\text{CatBoost}}$ (France shifted to 70% CatBoost).
5. **Strict Precision Gap Guard (The V4 Winning Edge):**
   - Implements per-country calibrated thresholds ($\tau_{\text{US}}=0.68, \tau_{\text{India}}=0.65, \tau_{\text{France}}=0.67$).
   - Dynamically tracks accepted candidates: as soon as an entity accumulates $\ge 2$ matches, any trailing candidate whose probability drops by $> 0.16$ below the top candidate is immediately rejected.
   - This ruthlessly defends against distractor false positives, directly optimizing for the $4\times$ precision weighting in Macro $F_{0.5}$.
6. **Greedy Injective Bipartite Matching:**
   - $O(M \log M)$ assignment: strictly ensures each S2/S3 candidate can match at most one S1 entity, capping matches at `MAX_MATCHES_PER_ENTITY = 8`.
7. **Streaming Memory-Safe TSV Exporter:**
   - Slices and streams rows directly from parquet chunks, ensuring $< 1.5$ GB RAM footprint.

---

## 2. Requirements & Setup

### Environment
- Python 3.10+
- 16GB RAM recommended
- NVIDIA GPU recommended for CatBoost acceleration (CPU fallback fully supported)
- OS: Windows, Linux, or macOS

### Installation
```bash
pip install -r requirements.txt
```

---

## 3. End-to-End Execution Guide

Place the official competition dataset inside `dataset/`:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Full Pipeline Run
```bash
# Run complete V4 end-to-end pipeline
python src/pipeline.py
```

### Step-by-Step Execution
```bash
# 1. Preprocessing and caching
python src/pipeline.py --step preprocess

# 2. 38-Channel Super-Blocking candidate generation
python src/pipeline.py --step blocking

# 3. Feature extraction & GBDT ensemble training
python src/pipeline.py --step train

# 4. Generate test predictions with Strict Gap Guard & Bipartite Matching
python src/pipeline.py --step predict

# 5. Validate output files against official contest rules
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 4. Output Deliverables
The pipeline generates two tab-separated files in `output/`:
1. `output/matching_results.tsv` (Leaderboard submission file, 1,732,544 rows)
2. `output/candidate_pairs.tsv` (Blocking candidate pool for verification audit, 1,732,544 rows)
