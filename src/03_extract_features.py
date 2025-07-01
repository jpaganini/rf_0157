import os
import sys
import argparse
import pandas as pd
from pathlib import Path

"""
03_extract_features.py

Extracts a subset of user-selected features from a full feature matrix and merges with labels and group IDs.

Inputs (CLI args):
  --muvr_file     Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv).
                   Must include:
                     * sample ID as index
                     * one column with the chosen label (specified via --label)
                     * remaining columns are selected feature names.
  --chisq_file    Path to full feature matrix TSV (e.g. full_chisq_matrix.tsv).
                   Must include:
                     * sample ID as first column (will be used as index)
                     * all feature columns, from which selected ones will be extracted.
  --metadata      Path to metadata TSV containing sample ID (index) and a grouping column.
  --label         Name of the label column in the MUVR file to include in output.
  --group_column  Name of the grouping column in the metadata file.
  --output_dir    Directory where the extracted feature matrix will be written.
                   Parent directories will be created if needed.
  --name          Base name (without extension) for the output file; “.tsv” will be appended.

Outputs:
  A TSV file at <output_dir>/<name>.tsv that contains, for each sample:
    * extracted features (columns matching those in the MUVR file minus the label)
    * a 'label' column with the outcome values
    * a 'group' column with group IDs from metadata

Usage Example:
  python 03_extract_features.py \
    --muvr_file results/muvr_RFC_min.tsv \
    --chisq_file data/full_chisq_matrix.tsv \
    --metadata data/sample_metadata.tsv \
    --label SYMP \
    --group_column cohort \
    --output_dir results \
    --name features_min_merged
"""

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract MUVR-selected features from a full feature matrix for all samples (train/test)."
    )
    parser.add_argument('--muvr_file', type=str, required=True,
                        help='Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv)')
    parser.add_argument('--chisq_file', type=str, required=True,
                        help='Path to full feature matrix (e.g. full_chisq_matrix.tsv)')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata TSV containing index and group column')
    parser.add_argument('--label', type=str, required=True,
                        help='Name of the label column to include in output')
    parser.add_argument('--group_column', type=str, required=True,
                        help = 'Name of the grouping column in metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help = 'Directory where the extracted feature matrix will be written')
    parser.add_argument('--name', type=str, required=True,
                        help = 'Base name (without extension) for the output file')

    return parser.parse_args()


def load_selected(muvr_path, label_col):
    """
        Load the MUVR file, extract selected feature names and labels.
        Returns:
            features: list of feature column names
            labels: pd.Series indexed by sample ID
        """
    df = pd.read_csv(muvr_path, sep='\t', index_col=0)
    if label_col not in df.columns:
        raise SystemExit(f"Error: label '{label_col}' not found in MUVR file")
    # all other columns are selected features
    features = [c for c in df.columns if c != label_col]
    return features

def load_metadata(meta_path, label_col, group_col):
    """
    Load metadata, ensuring group_col exists.
    Returns:
        pd.Series of group IDs indexed by sample ID
    """
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
    missing = [col for col in (label_col, group_col) if col not in meta.columns]
    if missing:
        raise SystemExit(f"Error: columns {missing} not found in metadata file")
    labels = meta[label_col]
    groups = meta[group_col]
    return labels, groups


def extract_features(chisq_file, selected_features):
    """Extract selected features from full chisq matrix."""
    #usecols = ['Unnamed: 0'] + selected_features  # 'Unnamed: 0' ensures the index column is loaded
    #df = pd.read_csv(chisq_file, sep='\t', usecols=usecols, index_col=0)
    df_full = pd.read_csv(chisq_file, sep='\t', index_col=0)
    df = df_full[selected_features]
    return df

def main():
    args = parse_arguments()

    # 1. Load selected feature list and labels
    print(f"Loading MUVR-selected features from {args.muvr_file}")
    features = load_selected(args.muvr_file,args.label)

    # 2. Load metadata groups
    print(f"Loading metadata (group IDs) from {args.metadata}")
    labels, groups = load_metadata(args.metadata, args.label, args.group_column)

    # 3. Extract features from full matrix
    print(f"Extracting {len(features)} features …")
    chisq_features = extract_features(args.chisq_file, features)

    # 4. Merge into one table
    print("Merging features, labels, and group IDs")
    df = pd.concat([chisq_features, labels, groups], axis=1, join='inner')

    # 5. Write output
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ensure .tsv extension
    fname = args.name
    if not fname.endswith('.tsv'):
        fname += '.tsv'
    final_path = outdir / fname

    print(f"Saving assembled dataset to {final_path}")
    df.to_csv(final_path, sep='\t')
    print("Done.")


if __name__ == "__main__":
    main()
