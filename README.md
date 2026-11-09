# CRISPR-Cas9 On-Target Efficiency Predictor 🧬

A machine learning system for predicting CRISPR-SpCas9 guide RNA (sgRNA) on-target cleavage activity. Built using LightGBM, biophysical feature engineering, and an interactive Streamlit web application.

---

## Architecture & Biological Principles

The SpCas9 endonuclease requires a 20-nucleotide guide RNA (sgRNA) complementary to the genomic target, positioned immediately 5' of a canonical `NGG` Protospacer Adjacent Motif (PAM).

### Key Biological Modeling:
1. **GC Content Dynamics**: Moderate-to-high GC content (40%–70%) promotes stable R-loop formation and hybridization thermodynamics, driving higher cleavage efficiency.
2. **PAM-Proximal Seed Region (Positions 11–20)**: The 10 nucleotides adjacent to the PAM govern initial Cas9 interrogation, target unwinding, and nucleolytic activation.
3. **Poly-T Tract Premature Termination Penalty**: In mammalian expression using standard type III RNA Polymerase promoters (U6 / H1), a contiguous tract of 4 thymines (`TTTT`) acts as a termination signal. Guides harboring poly-T tracts suffer severe expression deficits and are penalized by -0.30 in benchmark scoring.
4. **Positional Mononucleotide Preferences**: Captures position-specific nucleotide affinities across all 20 positions of the protospacer (e.g. preference for G and avoidance of T at position 20).

---

## Project Structure

```
crispr_on_target_predictor/
├── get_data.py             # Synthetic benchmark dataset generator (2,000 samples)
├── features.py             # Biophysical & positional feature extraction pipeline
├── train.py                # LightGBM Regressor training & Spearman correlation evaluation
├── app.py                  # Interactive Streamlit Web Application
├── doench_benchmark.csv    # Benchmark dataset (sequence, activity)
├── crispr_lgbm_model.pkl   # Serialized trained LightGBM model artifact
├── requirements.txt        # Python dependency manifest
└── README.md               # Architecture and documentation
```

---

## Feature Engineering Breakdown (83 Features)

| Feature | Type | Biological Significance |
| :--- | :--- | :--- |
| `overall_gc` | Float [0.0–1.0] | Overall GC content of the 20-nt guide RNA. |
| `seed_gc` | Float [0.0–1.0] | GC content across PAM-proximal seed region (positions 11–20). |
| `poly_t` | Binary [0 or 1] | Indicator of `TTTT` tract (U6/H1 Pol III termination). |
| `pos_{1..20}_{A,C,G,T}` | Binary [0 or 1] | One-hot encoded mononucleotide state across all 20 guide positions (80 features). |

---

## Model Evaluation (Test Set)

Trained with `n_estimators=150` and `learning_rate=0.05` on an 80/20 train/test split:

- **Spearman Rank Correlation ($\rho$)**: **0.8626** ($p = 7.89 \times 10^{-120}$)
- **Pearson Correlation ($r$)**: **0.9288** ($p = 9.86 \times 10^{-174}$)
- **Coefficient of Determination ($R^2$)**: **0.8606**
- **Root Mean Squared Error (RMSE)**: **0.0634**
- **Mean Absolute Error (MAE)**: **0.0497**

---

## Quickstart & Usage

### 1. Generate Synthetic Benchmark Dataset
```bash
python get_data.py --n_samples 2000 --output doench_benchmark.csv
```

### 2. Extract Features & Train Model
```bash
python train.py --data doench_benchmark.csv --output crispr_lgbm_model.pkl
```

### 3. Launch Streamlit Web UI
```bash
streamlit run app.py
```
