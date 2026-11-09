"""
features.py - CRISPR-Cas9 Guide RNA Feature Extraction Pipeline

Extracts biophysical and sequence-derived numerical features from 23-mer DNA strings
(20-nt protospacer guide + 3-nt 'NGG' PAM) for ML model training and inference.

Extracted Features:
1. Overall GC Content: GC percentage across the 20-nt guide.
2. Seed Region GC Content: GC percentage across the PAM-proximal seed region
   (positions 11-20, where position 1 is 5' end and 20 is PAM-adjacent 3' end).
3. Poly-T Tract Flag: Binary indicator (1/0) of 'TTTT' presence in the guide.
4. One-Hot Encoded Mononucleotides: 80 binary features representing A, C, G, T
   at each of the 20 positions of the guide (pos_1_A ... pos_20_T).
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd

NUCLEOTIDES = ["A", "C", "G", "T"]
GUIDE_LENGTH = 20
PAM_LENGTH = 3
TOTAL_LENGTH = GUIDE_LENGTH + PAM_LENGTH  # 23-mer


def get_feature_names() -> List[str]:
    """
    Return the standard, deterministic list of feature names.
    Order:
    1. overall_gc
    2. seed_gc
    3. poly_t
    4. pos_{i}_{nt} for i in 1..20, nt in ['A', 'C', 'G', 'T']
    """
    names = ["overall_gc", "seed_gc", "poly_t"]
    for pos in range(1, GUIDE_LENGTH + 1):
        for nt in NUCLEOTIDES:
            names.append(f"pos_{pos}_{nt}")
    return names


def calculate_gc(sequence: str) -> float:
    """Calculate GC ratio (0.0 to 1.0) of a nucleotide sequence."""
    if not sequence:
        return 0.0
    seq = sequence.upper()
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)


def validate_sequence(sequence: str) -> str:
    """Validate and clean a 23-mer target DNA sequence."""
    clean_seq = sequence.strip().upper()
    if len(clean_seq) != TOTAL_LENGTH:
        raise ValueError(
            f"Expected sequence length of {TOTAL_LENGTH} (20-nt guide + 3-nt PAM), "
            f"got {len(clean_seq)}: '{sequence}'"
        )
    valid_chars = set("ACGTN")
    if not set(clean_seq).issubset(valid_chars):
        raise ValueError(f"Sequence contains invalid non-DNA characters: '{clean_seq}'")
    return clean_seq


def extract_features(sequence: str) -> Dict[str, Union[float, int]]:
    """
    Extract numerical feature dictionary from a single 23-mer target sequence.

    Parameters:
        sequence: 23-nt DNA string (20-nt guide + 3-nt PAM, e.g. 'GTCG...TGG')

    Returns:
        Dict[str, Union[float, int]]: Feature dictionary with keys:
            - 'overall_gc': float (0.0 - 1.0)
            - 'seed_gc': float (0.0 - 1.0)
            - 'poly_t': int (0 or 1)
            - 'pos_{i}_{nt}': int (0 or 1) for i in 1..20 and nt in A, C, G, T
    """
    seq = validate_sequence(sequence)
    guide = seq[:GUIDE_LENGTH]  # 20-nt guide RNA
    # pam = seq[GUIDE_LENGTH:]  # 3-nt PAM (e.g. NGG)

    # 1. Overall GC content of the guide
    overall_gc = calculate_gc(guide)

    # 2. Seed region GC content: positions 11-20 (PAM-proximal 10 nucleotides)
    # In 1-based indexing: pos 11-20 corresponds to Python slice [10:20]
    seed_region = guide[10:20]
    seed_gc = calculate_gc(seed_region)

    # 3. Poly-T tract flag (Pol III premature termination motif)
    poly_t = 1 if "TTTT" in guide else 0

    # Assemble dictionary
    features: Dict[str, Union[float, int]] = {
        "overall_gc": round(overall_gc, 4),
        "seed_gc": round(seed_gc, 4),
        "poly_t": poly_t,
    }

    # 4. One-hot encode each nucleotide for positions 1 through 20
    for idx, char in enumerate(guide):
        pos_num = idx + 1  # 1-indexed (pos 1 to 20)
        for nt in NUCLEOTIDES:
            features[f"pos_{pos_num}_{nt}"] = 1 if char == nt else 0

    return features


def extract_features_df(sequences: Union[List[str], pd.Series]) -> pd.DataFrame:
    """
    Vectorized / batch extraction returning a formatted pandas DataFrame
    with deterministic feature columns.
    """
    feature_dicts = [extract_features(seq) for seq in sequences]
    df = pd.DataFrame(feature_dicts)
    # Ensure strict column ordering
    cols = get_feature_names()
    return df[cols]


if __name__ == "__main__":
    test_seq = "GAGTCCGAGCAGAAGAAGAAGGG"
    print(f"[+] Testing feature extraction on: {test_seq}")
    feats = extract_features(test_seq)
    print(f"    - Overall GC: {feats['overall_gc']}")
    print(f"    - Seed GC: {feats['seed_gc']}")
    print(f"    - Poly-T present: {feats['poly_t']}")
    print(f"    - pos_1_G: {feats['pos_1_G']}, pos_1_A: {feats['pos_1_A']}")
    print(f"    - pos_20_A: {feats['pos_20_A']}")
    print(f"    - Total feature count: {len(feats)} (Expected: 83)")
    assert len(feats) == 83, "Feature count mismatch!"
    print("[OK] Feature extraction test passed successfully!")
