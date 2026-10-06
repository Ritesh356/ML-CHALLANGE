# ML-CHALLANGE
Entity resolution for business records across source datasets using ML-driven matching, robust feature design, and validation-focused model tuning. Achieves 98.77% accuracy.
# Business Entity Resolution

Business Entity Resolution pipeline for matching business records across multiple sources using normalization, blocking, feature engineering, and machine learning-based matching.

## Overview

This project identifies matching records for each business in Source 1 against records in Source 2 and Source 3. The pipeline is built around a robust record-linkage workflow with learned normalization, candidate generation, feature extraction, model scoring, and final re-ranking for uncertain matches.

The project achieves:
- 98.77% accuracy
- strong macro F0.5 performance
- reproducible validation workflow
- final-stage cross-encoder reranking for difficult cases

## Challenge Goal

For every Source 1 (S1) business, find its matching record in Source 2 / Source 3 (S2/S3), with scoring based on macro F0.5 per S1 and inclusion of singletons.

## Pipeline

The workflow includes:

1. Name and address normalization
2. Candidate generation with blocking
3. Pair feature extraction
4. LightGBM-based matching
5. Uncertain record reranking with a cross-encoder
6. Final decision logic and submission output

## Project Structure

```bash
code/
  business_entity_resolution/
    README.md
    requirements.txt
    kaggle/
      bet2_rerank.ipynb
      bet2_test.ipynb
    models/
      cross_encoder/
      v4/
    src/
      baseline.py
      blocking.py
      decision.py
      embed_blocking.py
      ensemble.py
      evaluate.py
      features.py
      filler_words.json
      learn_translit.py
      normalize.py
      predict.py
      rerank.py
      run_normalize.py
      segment_thresholds.py
      splice_country.py
      state_aliases.json
      train.py
      train_all.py
      train_catboost.py
      train_xgb.py
      validation.py
      translit_map.json
output/
  candidate_pairs.tsv
  matching_results.tsv
