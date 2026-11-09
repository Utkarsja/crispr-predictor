"""
app.py - CRISPR-Cas9 On-Target Efficiency Predictor Streamlit Application

A Lead Bio-AI Engineer interface for discovering, scoring, and ranking SpCas9
guide RNAs (sgRNAs) from user-provided target DNA sequences.

Powered by LightGBM trained on biophysical and sequence-specific features:
- GC content correlation & seed-region dynamics (positions 11-20)
- Poly-T transcription termination penalties ('TTTT')
- Positional nucleotide preferences across all 20 guide positions
"""

import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure current directory is accessible for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from features import extract_features_df, calculate_gc, get_feature_names

MODEL_PATH = os.path.join(CURRENT_DIR, "crispr_lgbm_model.pkl")

# Standard sample target sequence (Human EMX1 gene exon 3 locus, canonical CRISPR benchmark)
DEFAULT_TARGET_DNA = (
    "GGGCCAGGGTTCTCCTCCCTCAGTGACCCAGGAGAGTGAGGAGGCTCTTCCTGGAGGGGG"
    "ACTGGCAGGCAGGCCCCAGGAGGGTCTTAGTCCGAGCAGAAGAAGAAGGGCTCCCATCAC"
    "ATCAACCGGTGGCGCATTGCCACGAAGCAGGCCAATGGGGAGGACATCGATGTCACCTCC"
)


@st.cache_resource
def load_crispr_model():
    """Load serialized LightGBM model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def clean_dna_input(raw_text: str) -> str:
    """Strip FASTA headers, numbers, whitespace, and convert to uppercase."""
    lines = raw_text.strip().splitlines()
    non_header_lines = [line.strip() for line in lines if not line.startswith(">")]
    merged = "".join(non_header_lines)
    clean = re.sub(r"[^A-Za-z]", "", merged).upper()
    return clean


def reverse_complement(dna_seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return dna_seq.translate(complement_map)[::-1]


def scan_pam_sites(dna_seq: str, scan_reverse_strand: bool = True):
    """
    Scan DNA sequence for all canonical SpCas9 'NGG' PAM sites (20-nt guide + 3-nt PAM).
    Uses regex lookahead (?=([ACGT]{20}[ACGT]GG)) to capture overlapping targets.
    """
    candidates = []
    seq_len = len(dna_seq)

    # 1. Sense Strand (+)
    pattern = re.compile(r"(?=([ACGT]{20}[ACGT]GG))")
    for match in pattern.finditer(dna_seq):
        start_idx = match.start()
        full_23 = match.group(1)
        guide = full_23[:20]
        pam = full_23[20:23]
        end_idx = start_idx + 23  # 0-indexed exclusive

        candidates.append({
            "target_23mer": full_23,
            "guide_seq": guide,
            "pam": pam,
            "strand": "+ (Sense)",
            "start": start_idx + 1,  # 1-indexed for biological readability
            "end": end_idx,
            "seed_seq": guide[10:20],
        })

    # 2. Antisense Strand (-)
    if scan_reverse_strand and seq_len >= 23:
        rev_comp = reverse_complement(dna_seq)
        for match in pattern.finditer(rev_comp):
            rev_start = match.start()
            full_23 = match.group(1)
            guide = full_23[:20]
            pam = full_23[20:23]
            # Map coordinates back to the original (+) genomic coordinates
            genomic_end = seq_len - rev_start
            genomic_start = genomic_end - 23 + 1

            candidates.append({
                "target_23mer": full_23,
                "guide_seq": guide,
                "pam": pam,
                "strand": "- (Antisense)",
                "start": genomic_start,
                "end": genomic_end,
                "seed_seq": guide[10:20],
            })

    return candidates


def assign_tier(score: float) -> str:
    """Categorize predicted score into biological efficacy tiers."""
    if score >= 0.60:
        return "🟢 High"
    elif score >= 0.35:
        return "🟡 Moderate"
    else:
        return "🔴 Low / Inefficient"


def main():
    # Configure page layout and style
    st.set_page_config(
        page_title="CRISPR-Cas9 On-Target Efficiency Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 CRISPR-Cas9 On-Target Efficiency Predictor")
    st.markdown(
        "**Bio-AI Engineering Suite** | Automated Target DNA Scanning, Biophysical Feature Extraction & "
        "LightGBM Activity Scoring."
    )

    # Sidebar: Biological Controls and System Metadata
    with st.sidebar:
        st.header("⚙️ Configuration & Filters")
        scan_both_strands = st.checkbox(
            "Scan Both Strands (+ / -)",
            value=True,
            help="Scan both sense and reverse-complement strands for NGG PAMs."
        )
        min_score_filter = st.slider(
            "Min Activity Score Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
        top_k_select = st.number_input(
            "Display Top K Guides",
            min_value=5,
            max_value=500,
            value=50,
            step=5
        )

        st.markdown("---")
        st.header("🔬 Model & Benchmark Specs")
        st.markdown(
            """
            - **Target Cas Enzyme**: *Streptococcus pyogenes* Cas9 (SpCas9)
            - **PAM Specificity**: `5'-NGG-3'`
            - **Architecture**: LightGBM Gradient Boosted Regressor
            - **Features (83)**:
              - Overall GC Content (20-mer)
              - Seed Region GC (positions 11-20)
              - Poly-T Tract Penalty (`TTTT`)
              - Positional Mononucleotides (pos 1-20)
            - **Validation Metric**: Spearman Rank Correlation ($\rho \\approx 0.86$)
            """
        )
        st.markdown("---")
        st.caption("Developed by Lead Bio-AI Engineer")

    # Main Content: Target Sequence Input
    st.subheader("1. Target DNA Sequence Input")
    dna_input = st.text_area(
        "Paste target genomic sequence (raw DNA or FASTA format):",
        value=DEFAULT_TARGET_DNA,
        height=140,
        help="Supports A, C, G, T nucleotides. Non-nucleotide characters and FASTA headers are automatically stripped.",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        scan_button = st.button("🚀 Scan & Predict Guides", type="primary", use_container_width=True)

    with col_info:
        clean_dna = clean_dna_input(dna_input)
        st.info(f"Sequence Length: **{len(clean_dna)} bp** | Target Strand GC: **{calculate_gc(clean_dna) * 100:.1f}%**")

    model = load_crispr_model()

    if not model:
        st.error(
            "❌ Trained LightGBM model (`crispr_lgbm_model.pkl`) not found! "
            "Please execute `python train.py` to generate the model artifact."
        )
        st.stop()

    # Execution on click or initial load
    if scan_button or dna_input:
        if len(clean_dna) < 23:
            st.warning("⚠️ Target sequence must be at least 23 base pairs long to contain a full 20-nt guide + 3-nt PAM.")
            st.stop()

        with st.spinner("Scanning for PAM sites and computing biophysical feature vectors..."):
            candidate_sites = scan_pam_sites(clean_dna, scan_reverse_strand=scan_both_strands)

            if not candidate_sites:
                st.error("No valid 'NGG' PAM sites (with 20-nt upstream flanking region) detected in the provided sequence.")
                st.stop()

            # Batch feature extraction
            sequences_23 = [c["target_23mer"] for c in candidate_sites]
            features_df = extract_features_df(sequences_23)

            # LightGBM inference
            raw_predictions = model.predict(features_df)
            predicted_scores = np.clip(raw_predictions, 0.0, 1.0)

            # Build comprehensive results DataFrame
            records = []
            for i, c in enumerate(candidate_sites):
                score = float(predicted_scores[i])
                poly_t_flag = "TTTT" in c["guide_seq"]
                guide_gc = calculate_gc(c["guide_seq"])
                seed_gc = calculate_gc(c["seed_seq"])

                records.append({
                    "Guide Sequence (5'->3')": c["guide_seq"],
                    "PAM": c["pam"],
                    "Predicted Activity": round(score, 4),
                    "Efficiency Tier": assign_tier(score),
                    "Strand": c["strand"],
                    "Start": c["start"],
                    "End": c["end"],
                    "Overall GC %": round(guide_gc * 100, 1),
                    "Seed GC %": round(seed_gc * 100, 1),
                    "Poly-T Warning": "⚠️ Yes (-0.30)" if poly_t_flag else "✅ None",
                    "Full 23-mer": c["target_23mer"],
                })

            results_df = pd.DataFrame(records)
            # Sort descending by Predicted Activity
            results_df = results_df.sort_values(by="Predicted Activity", ascending=False).reset_index(drop=True)
            results_df.index = results_df.index + 1
            results_df.index.name = "Rank"

            # Apply filtering
            filtered_df = results_df[results_df["Predicted Activity"] >= min_score_filter].head(int(top_k_select))

        st.markdown("---")
        st.subheader("2. Candidate Guide RNA Scoring & Ranking")

        # Metric summary row
        top_score = results_df["Predicted Activity"].max()
        avg_score = results_df["Predicted Activity"].mean()
        poly_t_ratio = (results_df["Poly-T Warning"] != "✅ None").mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top Guide Score", f"{top_score:.4f}", delta="Optimal Candidate")
        m2.metric("Total PAM Sites Found", f"{len(results_df)}")
        m3.metric("Average On-Target Score", f"{avg_score:.4f}")
        m4.metric("Guides with Poly-T Penalty", f"{poly_t_ratio:.1f}%")

        # Interactive Table Display with native Streamlit column configuration
        display_cols = [
            "Predicted Activity",
            "Efficiency Tier",
            "Guide Sequence (5'->3')",
            "PAM",
            "Strand",
            "Start",
            "End",
            "Overall GC %",
            "Seed GC %",
            "Poly-T Warning",
        ]

        st.dataframe(
            filtered_df[display_cols],
            column_config={
                "Predicted Activity": st.column_config.ProgressColumn(
                    "Predicted Activity",
                    help="On-target cleavage efficiency score (0.0 to 1.0)",
                    format="%.4f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Overall GC %": st.column_config.NumberColumn(
                    "Overall GC %",
                    format="%.1f%%",
                ),
                "Seed GC %": st.column_config.NumberColumn(
                    "Seed GC %",
                    format="%.1f%%",
                ),
            },
            use_container_width=True,
            height=min(450, 40 + len(filtered_df) * 35),
        )

        # Export Button
        csv_bytes = filtered_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Export Candidate Guide RNAs (CSV)",
            data=csv_bytes,
            file_name="crispr_candidate_guides.csv",
            mime="text/csv",
        )

        # 3. Bio-AI Visual Analytics
        st.markdown("---")
        st.subheader("3. Biophysical Analytics & Distribution")
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.markdown("**Positional Guide Distribution along Target Sequence**")
            pos_chart_data = results_df.copy()
            st.scatter_chart(
                pos_chart_data,
                x="Start",
                y="Predicted Activity",
                color="Strand",
                height=300,
            )

        with vcol2:
            st.markdown("**Predicted On-Target Activity Score Distribution**")
            counts, bin_edges = np.histogram(results_df["Predicted Activity"], bins=15, range=(0.0, 1.0))
            bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))]
            hist_df = pd.DataFrame({"Score Range": bin_labels, "Guide Count": counts}).set_index("Score Range")
            st.bar_chart(hist_df, height=300)

        # 4. Deep Guide Inspector
        st.markdown("---")
        st.subheader("4. Detailed Candidate Guide Inspector")
        top_candidates = filtered_df.head(10)

        def format_guide_choice(r):
            seq = top_candidates.loc[r, "Guide Sequence (5'->3')"]
            act = top_candidates.loc[r, "Predicted Activity"]
            return f"Rank {r}: {seq} (Score: {act:.4f})"

        selected_rank = st.selectbox(
            "Select candidate guide by rank to view biophysical breakdown:",
            options=top_candidates.index.tolist(),
            format_func=format_guide_choice,
        )

        if selected_rank in filtered_df.index:
            guide_row = filtered_df.loc[selected_rank]
            guide_seq_val = guide_row["Guide Sequence (5'->3')"]
            pam_val = guide_row["PAM"]
            strand_val = guide_row["Strand"]
            start_val = guide_row["Start"]
            end_val = guide_row["End"]
            seed_region_val = guide_row["Full 23-mer"][10:20]
            gc_val = guide_row["Overall GC %"]
            seed_gc_val = guide_row["Seed GC %"]
            poly_t_val = guide_row["Poly-T Warning"]
            tier_val = guide_row["Efficiency Tier"]
            score_val = guide_row["Predicted Activity"]

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"**Guide Sequence (20-nt):** `{guide_seq_val}`")
                st.markdown(f"**PAM:** `{pam_val}` ({strand_val})")
                st.markdown(f"**Coordinates:** {start_val}..{end_val}")
            with d2:
                st.markdown(f"**PAM-Proximal Seed Region (Pos 11-20):** `{seed_region_val}`")
                st.markdown(f"**Overall GC Content:** `{gc_val}%`")
                st.markdown(f"**Seed Region GC Content:** `{seed_gc_val}%`")
            with d3:
                st.markdown(f"**Poly-T Tract Status:** {poly_t_val}")
                st.markdown(f"**Predicted Efficiency Tier:** {tier_val}")
                st.markdown(f"**Final Predicted Activity Score:** `{score_val:.4f}`")


if __name__ == "__main__":
    main()
