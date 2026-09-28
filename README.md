# Sentence-BERT (SBERT) Paper Reproduction

*A complete reproduction and experimental analysis of the EMNLP 2019 paper.*

**Sentence-BERT: Sentence Embeddings using Siamese BERT Networks**

> **Authors:** Nils Reimers & Iryna Gurevych
> **Conference:** EMNLP-IJCNLP 2019

---

## Overview

Sentence-BERT (SBERT) modifies the original BERT architecture into a **Siamese and Triplet Network** that produces fixed-length sentence embeddings. Unlike BERT's cross-encoder, which requires comparing every sentence pair jointly, SBERT encodes each sentence independently and computes similarity using cosine similarity. This makes semantic search, clustering, duplicate detection, and information retrieval practical at large scale.

The original paper demonstrated that searching among **10,000 sentences** drops from roughly **65 hours** with BERT to about **5 seconds** with SBERT while maintaining state-of-the-art semantic similarity performance.

This repository reproduces the major experiments reported in the original paper using modern implementations while preserving the original training objectives, evaluation protocol, and reported metrics.

---

## Objectives

This project reproduces and validates the principal experiments from the SBERT paper.

### Experiments Reproduced

- Unsupervised Semantic Textual Similarity (Table 1)
- Supervised STS Benchmark (Table 2)
- Argument Facet Similarity (Table 3)
- Wikipedia Section Triplets (Table 4)
- SentEval Transfer Tasks (Table 5)
- Multi-seed reproducibility analysis

Each experiment compares:

- Original paper results
- Reproduced implementation
- Numerical differences
- Visual comparisons

---

## Repository Structure

```text
├── charts/
│   ├── chart.py
│   ├── fig1_table1_unsupervised_sts.png
│   ├── fig2_table2_stsb_supervised.png
│   ├── fig3_table3_afs.png
│   ├── fig4_table4_wikisec.png
│   ├── fig5_table5_senteval.png
│   ├── fig6_summary_differences.png
│   └── fig7_archi.png
│
├── notebooks/
│   ├── TABLE-1/
│   │   ├── SBERT_NLI_base/
│   │   └── SRoBERTa_NLI_base/
│   │
│   ├── TABLE-2/
│   │   ├── SBERT_STSb_base/
│   │   ├── SBERT_NLI_STSb_base/
│   │   ├── SRoBERTa_STSb_base/
│   │   └── SRoBERTa-NLI-STSb-base/
│   │
│   ├── TABLE-3/
│   │   └── SBERT_AFS_base/
│   │
│   ├── TABLE-4/
│   │   └── SBERT_WikiSec_base/
│   │
│   └── TABLE-5/
│       └── SBERT_NLI_base/
│
├── .gitignore
├── Data_Verification.ipynb
├── D19-1410.pdf
└── README.md
```

---

# SBERT Architecture

Unlike BERT's cross-encoder, SBERT processes both sentences through two identical BERT encoders with shared weights.

<p align="center">
  <img src="charts/fig7_archi.png" width="700">
</p>

The resulting sentence embeddings are compared using cosine similarity.

### Training Objectives

SBERT is trained using three different objectives depending on the downstream task.

| Objective | Purpose |
|-----------|---------|
| Classification | SNLI / MultiNLI |
| Regression | STS Benchmark |
| Triplet Loss | Wikipedia Sections |

### Classification Objective

The classification objective concatenates the two sentence embeddings with their absolute element-wise difference before applying a softmax classifier.

$$
o=\text{softmax}\left(W_t(u,v,|u-v|)\right)
$$

where:

- $u$ = sentence embedding of sentence A
- $v$ = sentence embedding of sentence B
- $W_t$ = trainable weight matrix
- $o$ = predicted class probabilities

### Triplet Loss

For triplet training, SBERT minimizes the distance between an anchor and a positive sentence while maximizing the distance to a negative sentence.

$$
\max\left(\|s_a-s_p\|-\|s_a-s_n\|+\epsilon,0\right)
$$

where:

- $s_a$ = anchor embedding
- $s_p$ = positive embedding
- $s_n$ = negative embedding
- $\epsilon$ = margin (set to 1 in the original paper)

---

# Experimental Setup

| Component | Configuration |
|------------|----------------|
| Base Model | BERT-base / RoBERTa-base |
| Optimizer | Adam |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Pooling | Mean |
| Evaluation Metric | Spearman Correlation |

The implementation follows the original SBERT training configuration as closely as possible.

---

# Results

Each reproduced experiment includes the original paper values, the reproduced implementation, and the corresponding visualization.

---

# Table 1 — Unsupervised Semantic Textual Similarity

This experiment evaluates sentence embeddings on seven Semantic Textual Similarity datasets without using STS-specific training labels.

## Paper vs Replication

| Dataset | SBERT Paper | SBERT Replicated | Δ | SRoBERTa Paper | SRoBERTa Replicated | Δ |
|---------|------------:|-----------------:|--:|---------------:|--------------------:|--:|
| STS12 | 70.97 | 70.66 | -0.31 | 71.54 | 72.61 | +1.07 |
| STS13 | 76.53 | 72.78 | -3.75 | 72.49 | 73.98 | +1.49 |
| STS14 | 73.19 | 70.66 | -2.53 | 70.80 | 72.29 | +1.49 |
| STS15 | 79.09 | 78.87 | -0.22 | 78.74 | 79.58 | +0.84 |
| STS16 | 74.30 | 73.36 | -0.94 | 73.69 | 74.59 | +0.90 |
| STSb | 77.03 | 75.43 | -1.60 | 77.77 | 78.14 | +0.37 |
| SICK-R | 72.91 | 75.91 | +3.00 | 74.46 | 74.24 | -0.22 |

### Average Performance

| Model | Paper | Replicated | Difference |
|--------|------:|-----------:|-----------:|
| **SBERT-NLI-base** | **74.89** | **73.88** | **-1.01** |
| **SRoBERTa-NLI-base** | **74.21** | **75.06** | **+0.85** |

### Visualization

<p align="center">
  <img src="charts/fig1_table1_unsupervised_sts.png" width="850">
</p>

### Key Observations

- SBERT remains within approximately **1 point** of the original average.
- SRoBERTa slightly exceeds the reported paper average.
- The ranking across datasets is faithfully reproduced.

---

# Table 2 — STS Benchmark (Supervised)

This experiment fine-tunes SBERT on the STS Benchmark using 10 random seeds.

## Paper vs Replication

| Model | Training Setup | Paper (Mean ± SD) | Replicated (Mean ± SD) | Difference |
|-------|----------------|------------------:|------------------------:|-----------:|
| SBERT-STSb-base | STSb only | 84.67 ± 0.19 | 84.07 ± 0.49 | -0.60 |
| SRoBERTa-STSb-base | STSb only | 84.92 ± 0.34 | 85.14 ± 0.19 | +0.22 |
| SBERT-NLI-STSb-base | NLI → STSb | 85.35 ± 0.17 | 84.51 ± 0.47 | -0.84 |
| SRoBERTa-NLI-STSb-base | NLI → STSb | 84.79 ± 0.38 | 85.14 ± 0.18 | +0.35 |

### Visualization

<p align="center">
  <img src="charts/fig2_table2_stsb_supervised.png" width="850">
</p>

### Key Observations

- Multi-seed averaging closely matches the reported performance.
- SRoBERTa slightly exceeds the paper's reported value.
- Variance remains consistently low.

---

# Table 3 — Argument Facet Similarity (SBERT-AFS-base)

This experiment evaluates **SBERT-AFS-base** on the Argument Facet Similarity benchmark using both 10-fold cross-validation and cross-topic evaluation.

## Paper vs Replication

| Evaluation | Paper Pearson | Replicated Pearson | Δ | Paper Spearman | Replicated Spearman | Δ |
|------------|--------------:|-------------------:|--:|---------------:|--------------------:|--:|
| 10-fold Cross-Validation | 76.57 | 75.87 | -0.70 | 74.13 | 73.21 | -0.92 |
| Cross-topic | 52.34 | 51.21 | -1.13 | 50.65 | 49.20 | -1.45 |

### Visualization

<p align="center">
  <img src="charts/fig3_table3_afs.png" width="850">
</p>

### Key Observations

- Cross-topic evaluation remains the most challenging setting.
- The degradation trend matches the original paper.
- Results remain within roughly one point.

---

# Table 4 — Wikipedia Section Triplets (SBERT-WikiSec-base)

This experiment trains **SBERT-WikiSec-base** using triplet loss on approximately **1.8 million** Wikipedia triplets.

## Paper vs Replication

| Model | Paper | Replicated | Difference |
|-------|------:|-----------:|-----------:|
| **SBERT-WikiSec-base** | **80.42%** | **79.00%** | **-1.42%** |

### Visualization

<p align="center">
  <img src="charts/fig4_table4_wikisec.png" width="750">
</p>

### Key Observations

- Triplet-loss training reproduces the expected behavior.
- The reproduced accuracy remains very close to the original.

---

# Table 5 — SentEval Transfer Learning (SBERT-NLI-base)

This experiment evaluates **SBERT-NLI-base** sentence embeddings on downstream transfer tasks using the SentEval evaluation toolkit.

## Paper vs Replication

| Task | Paper | Replicated | Difference |
|------|------:|-----------:|-----------:|
| MR | 83.64 | 82.14 | -1.50 |
| CR | 89.43 | 88.77 | -0.66 |
| SUBJ | 94.39 | 93.09 | -1.30 |
| MPQA | 89.86 | 89.91 | +0.05 |
| SST | 88.96 | 88.41 | -0.55 |
| TREC | 89.60 | 88.00 | -1.60 |
| MRPC | 76.00 | 76.23 | +0.23 |

### Average Performance

| Model | Paper | Replicated | Difference |
|--------|------:|-----------:|-----------:|
| **SBERT-NLI-base** | **87.41** | **86.65** | **-0.76** |

### Visualization

<p align="center">
  <img src="charts/fig5_table5_senteval.png" width="900">
</p>

### Key Observations

- Most downstream tasks remain within one percentage point.
- MPQA and MRPC slightly exceed the reported values.
- Overall transfer-learning behavior matches the original paper.

---

# Overall Performance Summary

| Experiment | Model | Difference |
|------------|-------|-----------:|
| Table 1 | SBERT-NLI-base | -1.01 |
| Table 1 | SRoBERTa-NLI-base | +0.85 |
| Table 2 | SBERT-STSb-base | -0.60 |
| Table 3 | SBERT-AFS-base | ≈ -1.00 |
| Table 4 | SBERT-WikiSec-base | -1.42 |
| Table 5 | SBERT-NLI-base | -0.76 |

### Overall Comparison

<p align="center">
  <img src="charts/fig6_summary_differences.png" width="900">
</p>

The reproduced implementation consistently follows the original paper's trends while remaining within approximately **1–1.5 points** across nearly every benchmark.

---

# Why Small Differences Exist

Minor performance differences are expected due to:

- Random weight initialization
- Different hardware
- Updated Transformers implementations
- Tokenizer revisions
- Seed variance
- Modern library behavior

Despite these factors, the reproduced implementation preserves the original paper's conclusions across every benchmark.

---

# How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Recommended execution order:

1. Data Verification
2. STS Evaluation
3. STSb Fine-tuning
4. Argument Facet Similarity
5. Wikipedia Triplets
6. SentEval

---

# Future Improvements

- Full hyperparameter matching with the original environment
- Distributed training
- Multilingual SBERT variants
- FAISS semantic search benchmarking
- Large-scale retrieval experiments

---

# Author

**Aditya Mantri**

B.Tech Artificial Intelligence & Data Science

---

# Reference

Reimers, N., & Gurevych, I. (2019).

> **Sentence-BERT: Sentence Embeddings using Siamese BERT Networks**

EMNLP-IJCNLP 2019.