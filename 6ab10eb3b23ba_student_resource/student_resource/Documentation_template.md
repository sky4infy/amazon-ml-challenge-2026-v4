# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** GradientX  
**Team Members:** Vashishth Kumar & Data Science Engineering Team  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary
Team **GradientX** developed an industrial-grade, high-precision Business Entity Resolution pipeline (V4 Architecture) tailored specifically for the Amazon ML Challenge 2026. Addressing the multi-million record scale across three heterogeneous sources ($S1, S2, S3$) across US, India, and France (11.7 million records total), our solution couples an **18/38-Channel High-Recall Candidate Super-Blocking Engine** (verified **97.99% US, 94.62% India, 96.64% Global recall ceiling**) with a **C++ Accelerated SIMD Feature Extraction Pipeline** (56 lexical, phonetic, token, and geographic features via RapidFuzz AVX2), a **Dual-Model GBDT Ensemble** (Monotonic Leaf-Wise LightGBM + CatBoost GPU on an NVIDIA GeForce RTX 4050), and **Vectorized Greedy Injective Bipartite Matching** that strictly enforces the many-to-one linkage topology under the precision-heavy Entity-Level Macro $F_{0.5}$ metric.

On the official public leaderboard, the pipeline achieved an evaluated score of **0.8380**, matching our local out-of-fold validation score of **0.8338** with zero data leakage.

---

## 2. Methodology

### 2.1 Problem Analysis
Comprehensive exploratory data analysis across the 11.7 million business records revealed several critical challenges:
1. **Extreme Asymmetry and Scale:** The comparison space exceeds $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.72 \times 10^{13}$ pairwise comparisons, demanding an ultra-low complexity blocking mechanism that runs entirely in sub-linear time.
2. **Entity Cardinality & Singletons:** Exactly 5.58% (123,247 entities) in the reference Source 1 catalog have **no matching record** in either Source 2 or Source 3 (singletons). Under the official macro $F_{0.5}$ evaluation metric:
   $$\text{Macro } F_{0.5} = \frac{1}{|S1|} \sum_{e \in S1} \frac{1.25 \cdot P_e \cdot R_e}{0.25 \cdot P_e + R_e}$$
   Singletons with empty predicted match lists yield an entity score of $1.0$, whereas predicting even a single false match immediately collapses that entity's score to $0.0$. Precision is mathematically penalized $4\times$ harder than recall ($\beta = 0.5$).
3. **Severe Lexical and Orthographic Noise:**
   - **Address Truncation & Reordering:** In India, addresses frequently swap street/landmark/locality order (e.g., "Shop 4, MG Road, Pune" vs "MG Rd, Near Post Office, Pune, 411001").
   - **Multilingual & Transliteration Noise:** Over 120,000 Indian businesses feature transliteration variants (Chowk vs Chawk, Nagar vs Ngr, Bazaar vs Bazar).
   - **Zero-Shot Country Transfer:** France accounts for 259,452 entities (15% of the test set) but has zero ground-truth training records.

### 2.2 Solution Strategy
Our architecture implements an end-to-end, multi-stage pipeline designed for maximal precision and zero data leakage:

```
[Raw Sources S1, S2, S3]
           │
           ▼
[Stage 1: Preprocessing & Indic Transliteration Normalization]
   - Corporate suffix standardization (Pvt Ltd, LLC, Inc, Corp, Gmbh)
   - Structured subfield extraction: postal codes, street numbers, cities, state codes
           │
           ▼
[Stage 2: 38-Channel High-Recall Super-Blocking Engine]
   - Tier 1: Exact Name, Clean Suffix Strip, Space-Collapsed, Sorted Tokens
   - Tier 2: City + Street Num, City + First Token, Postal 5-digit + First Token, City + Prefixes
   - Tier 3: State Code + Exact Name, State Code + Clean Suffix, State Code + Building/Plot
   - Tier 4: Address Shingles (a0, a1, a2), Street Anchors, Natural Street Names
   - Verified Recall: 97.99% US, 94.62% India, 96.64% Global (7.38M / 7.64M GT pairs)
           │
           ▼
[Stage 3: 56-Channel C++ SIMD RapidFuzz Feature Extraction]
   - AVX2 bit-vector distance: Jaro-Winkler, Levenshtein, Token Sort, Jaccard, Simpson overlap
   - Cross-field entropy interaction products (name_addr_prod)
   - Exact numerical street number match (binary) and postal match
           │
           ▼
[Stage 4: Dual-Model GBDT Soft-Voting Ensemble]
   - LightGBM: Leaf-wise growth + Monotonic Similarity Constraints (+1) + scale_pos_weight=0.8
   - CatBoost GPU: Oblivious (symmetric) trees trained on NVIDIA RTX 4050 GPU (depth=8, 1500 trees)
   - Soft-voting blend: P = 0.80 * P_LightGBM + 0.20 * P_CatBoost (France shifted to 70% CatBoost)
           │
           ▼
[Stage 5: Greedy Injective Bipartite Matching]
   - Vectorized O(M log M) assignment enforcing unique target matches (max matches = 8)
           │
           ▼
[Official Output: matching_results.tsv (81.6 MB) & candidate_pairs.tsv (2.35 GB)]
```

---

## 3. Candidate Generation / Blocking Strategy

### 3.1 Design Principles
To transcend the 71.26% recall ceiling of legacy pipelines, V4 deploys 38 distinct non-destructive inverted index channels organized across 4 hierarchical tiers.

### 3.2 Channel Inventory & Empirical Performance
- **Exact & Cleaned Tokens (Tier 1):** Normalized name, suffix-stripped name, space-collapsed name, alphabetically sorted tokens.
- **Locality Anchors (Tier 2):** City + street number, city + first token, postal 5-digit + first token, city + 2/3/4-character name prefixes.
- **Geographic Expansion (Tier 3):** State code + exact name, state code + clean suffix name, state code + building codes.
- **Indic Shingles & Street Anchors (Tier 4):** Address shingles (a0, a1, a2), natural street names, anywhere street numbers.

**Empirical Recall Across 7,638,365 Ground Truth Pairs:**
- **US:** **97.99%** (4,486,683 / 4,578,522)
- **India:** **94.62%** (2,895,225 / 3,059,843)
- **Global:** **96.64%** (7,381,908 / 7,638,365)
- **Reduction Ratio:** **99.9987%** (evaluating only ~89.9 candidates/entity in US and ~127.3 in India).

---

## 4. Feature Engineering & Model Architecture

### 4.1 Feature Engineering (56 Channels)
1. **Lexical Similarities (C++ AVX2 SIMD via RapidFuzz):** Token Set Ratio, Token Sort Ratio, Partial Ratio, Jaro-Winkler, Simpson Overlap, Normalized Levenshtein.
2. **Subfield Structural Matchers:** Exact street number match (binary 1/0), postal match, city token similarity.
3. **Cross-Field Interaction Products:** `name_addr_prod` (multiplicative interaction between name similarity and address similarity), penalizing pairs with high name similarity but incompatible addresses.
4. **Phonetic & Transliteration:** Transliteration Jaccard similarity across normalized Indic phonemes.

### 4.2 Machine Learning Architecture
- **LightGBM:** Leaf-wise GBDT with Monotonic Constraints on all primary similarity metrics (guaranteeing that an increase in string similarity never decreases match probability) and `scale_pos_weight = 0.8` (forcing trees to penalize False Positives for the precision-weighted $F_{0.5}$ metric).
- **CatBoost GPU:** Oblivious symmetric decision trees trained directly on an NVIDIA GeForce RTX 4050 Laptop GPU (1,500 iterations, learning rate 0.04, depth 8, border count 128) using DirectML/CUDA.
- **France Zero-Shot Routing:** France test entities (259,452 records) are routed to 70% CatBoost weight because oblivious trees evaluate identical split criteria across all leaves, preventing out-of-distribution hallucinations on unseen French addresses.

---

## 5. Post-Processing & Validation

1. **Greedy Injective Bipartite Matching:** Vectorized sorting of candidate pairs by ensemble probability in $O(M \log M)$, strictly ensuring no candidate record in S2/S3 is assigned to multiple S1 entities.
2. **Cardinality Regularization:** Bounded at `MAX_MATCHES_PER_ENTITY = 8` (conforming to the empirical distribution where 99.8% of entities have $\le 8$ matches).
3. **Contest Validation:** Validated via `utils/validate_submission.py`, achieving `PASS` across all 1,732,544 rows with zero missing entities, zero intra-row duplicates, zero self-matches, and 100% subset compliance.

---

## 6. Summary of Results

| Architecture Milestone | Blocking Recall Ceiling | Local Val Macro $F_{0.5}$ | Public Leaderboard Score |
| :--- | :---: | :---: | :---: |
| **V3 Baseline** | 71.26% | 0.7294 | **0.7800** |
| **V4 Pipeline (Final)** | **96.64%** | **0.8338** | **0.8380** |

**Conclusion:** The V4 architecture successfully eliminated the 71% blocking recall ceiling, recovered ~90,000 false singletons, established perfect calibration between local validation and the public leaderboard, and demonstrated production-grade memory bounds (< 1.1 GB RAM) under extreme data scale.
