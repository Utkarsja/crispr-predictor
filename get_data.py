"""
get_data.py - CRISPR-Cas9 Synthetic Benchmark Dataset Generator

Generates a synthetic benchmark dataset (doench_benchmark.csv) simulating
on-target guide RNA activity based on biological CRISPR-Cas9 principles
(e.g., Doench et al. Rule Set 2, Hsu et al.).

Key Biological Dynamics:
1. Canonical SpCas9 PAM: 20-nt guide + 'NGG' PAM (23-mer total).
2. GC Content Correlation: Moderate-to-high GC content correlates positively
   with target binding efficiency and melting temperature.
3. Poly-T Tract Penalty: A stretch of 4 or more thymines ('TTTT') triggers
   premature termination by RNA Polymerase III (U6/H1 promoters), drastically
   reducing functional sgRNA expression. Penalized by -0.3.
4. Bounded Output: Activity scores are strictly normalized within [0.0, 1.0].
"""

import argparse
import os
import random
import numpy as np
import pandas as pd


def generate_random_guide(length: int = 20, poly_t_prob: float = 0.15) -> str:
    """
    Generate a 20-nt guide RNA protospacer sequence.
    Occasionally injects a poly-T tract to represent premature termination sites.
    """
    bases = ["A", "C", "G", "T"]
    
    # Intentionally enrich a fraction of sequences with 'TTTT' to benchmark penalty
    if random.random() < poly_t_prob:
        insert_pos = random.randint(0, length - 4)
        prefix = "".join(random.choices(bases, k=insert_pos))
        suffix = "".join(random.choices(bases, k=length - insert_pos - 4))
        return prefix + "TTTT" + suffix
    
    return "".join(random.choices(bases, k=length))


def generate_pam() -> str:
    """Generate a canonical SpCas9 PAM ('NGG')."""
    n_base = random.choice(["A", "C", "G", "T"])
    return f"{n_base}GG"


def calculate_gc(sequence: str) -> float:
    """Compute GC content ratio for a nucleotide sequence."""
    seq = sequence.upper()
    g_count = seq.count("G")
    c_count = seq.count("C")
    return (g_count + c_count) / len(seq) if seq else 0.0


def compute_synthetic_activity(guide: str, pam: str) -> float:
    """
    Compute synthetic continuous activity score (0.0 to 1.0) simulating biological rules:
    - Base score influenced by overall GC content (optimal around 40-70%).
    - Seed region (positions 11-20, PAM-proximal) GC contribution.
    - Favorable nucleotide preferences (e.g., G at position 20, avoid T at position 20).
    - Poly-T penalty: deduction of 0.3 if 'TTTT' occurs in the guide.
    - Biological noise / stochastic experimental variance.
    """
    # 1. Base activity distribution centered around 0.50
    base_activity = 0.42

    # 2. GC content correlation (+0.35 weight scaled around optimal GC ~0.50)
    overall_gc = calculate_gc(guide)
    seed_region = guide[10:20]  # PAM-proximal positions 11-20 (0-indexed 10:20)
    seed_gc = calculate_gc(seed_region)
    
    gc_effect = 0.28 * (overall_gc - 0.45) + 0.12 * (seed_gc - 0.45)

    # 3. Position-specific biological nuances (Doench Rule Set heuristics)
    pos_effect = 0.0
    # SpCas9 preference: G preferred at pos 20 (PAM-adjacent), T strongly disfavored
    if guide[-1] == "G":
        pos_effect += 0.06
    elif guide[-1] == "T":
        pos_effect -= 0.08
        
    # Preference: Purines (A/G) favored in position 1
    if guide[0] in ("A", "G"):
        pos_effect += 0.03

    # 4. Poly-T tract penalty (-0.3 penalty for Pol III termination signal)
    has_poly_t = "TTTT" in guide
    poly_t_penalty = -0.30 if has_poly_t else 0.0

    # 5. Biological experimental noise
    noise = np.random.normal(loc=0.0, scale=0.06)

    # Aggregate total raw score
    raw_score = base_activity + gc_effect + pos_effect + poly_t_penalty + noise

    # Strict clamping between 0.0 and 1.0
    activity_score = float(np.clip(raw_score, 0.0, 1.0))
    return round(activity_score, 4)


def generate_benchmark_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic benchmark dataset containing 20-nt guide + 'NGG' PAM (23-mer)
    and continuous activity scores.
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    seen_sequences = set()

    while len(records) < n_samples:
        guide = generate_random_guide(length=20, poly_t_prob=0.18)
        pam = generate_pam()
        full_seq = guide + pam  # 23-mer

        if full_seq in seen_sequences:
            continue
        seen_sequences.add(full_seq)

        activity = compute_synthetic_activity(guide, pam)
        records.append({
            "sequence": full_seq,
            "activity": activity
        })

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CRISPR-Cas9 benchmark dataset.")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of benchmark samples (default: 2000)")
    parser.add_argument("--output", type=str, default="doench_benchmark.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"[+] Generating {args.n_samples} synthetic CRISPR benchmark samples (seed={args.seed})...")
    df = generate_benchmark_dataset(n_samples=args.n_samples, seed=args.seed)

    # Compute diagnostic metrics
    guides = df["sequence"].apply(lambda s: s[:20])
    df["gc_content"] = guides.apply(calculate_gc)
    df["has_poly_t"] = guides.apply(lambda g: "TTTT" in g)

    corr = df["gc_content"].corr(df["activity"])
    poly_t_count = df["has_poly_t"].sum()
    mean_poly_t_act = df[df["has_poly_t"]]["activity"].mean() if poly_t_count > 0 else 0.0
    mean_non_poly_t_act = df[~df["has_poly_t"]]["activity"].mean()

    print(f"[+] Generation complete.")
    print(f"    - Total records: {len(df)}")
    print(f"    - GC Content vs Activity Pearson Correlation: {corr:.4f}")
    print(f"    - Guides with Poly-T ('TTTT'): {poly_t_count} ({poly_t_count / len(df) * 100:.1f}%)")
    print(f"    - Mean Activity (Non-Poly-T): {mean_non_poly_t_act:.4f}")
    print(f"    - Mean Activity (Poly-T Present): {mean_poly_t_act:.4f} (Delta: {mean_poly_t_act - mean_non_poly_t_act:.4f})")
    print(f"    - Score Range: [{df['activity'].min():.4f}, {df['activity'].max():.4f}]")

    # Save clean benchmark dataset with columns: ['sequence', 'activity']
    export_df = df[["sequence", "activity"]]
    output_path = os.path.abspath(args.output)
    export_df.to_csv(output_path, index=False)
    print(f"[OK] Saved benchmark dataset to: {output_path}")


if __name__ == "__main__":
    main()
