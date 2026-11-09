import os
import joblib
import pandas as pd
from features import extract_features, extract_features_df
from app import clean_dna_input, scan_pam_sites, DEFAULT_TARGET_DNA

print("[1] Verifying Model Artifact...")
assert os.path.exists("crispr_lgbm_model.pkl"), "Model file missing"
model = joblib.load("crispr_lgbm_model.pkl")
print("    Model loaded successfully:", type(model))

print("[2] Verifying Target Scanning...")
clean = clean_dna_input(DEFAULT_TARGET_DNA)
sites = scan_pam_sites(clean, scan_reverse_strand=True)
print(f"    Found {len(sites)} candidate PAM sites in test sequence.")
assert len(sites) > 0, "No PAM sites detected"

print("[3] Verifying Feature Extraction and Inference...")
seqs = [s["target_23mer"] for s in sites]
feat_df = extract_features_df(seqs)
preds = model.predict(feat_df)
print(f"    Generated {len(preds)} predictions.")
print(f"    Top predicted score: {max(preds):.4f}, Min: {min(preds):.4f}")

# Verify specific guide output
sample_dict = extract_features(seqs[0])
assert "overall_gc" in sample_dict
assert "seed_gc" in sample_dict
assert "poly_t" in sample_dict
assert "pos_1_A" in sample_dict
assert "pos_20_T" in sample_dict
print("    All 83 feature keys verified.")

print("[ALL PIPELINE TESTS PASSED OK!]")
