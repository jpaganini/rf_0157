import os
import argparse
import pandas as pd


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
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write the extracted feature matrix (e.g. results/features_min.tsv)')
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
        sys.exit(f"Error: label '{label_col}' not found in MUVR file")
    labels = df[label_col].rename('label')
    # all other columns are selected features
    features = [c for c in df.columns if c != label_col]
    return features, labels

def load_metadata(meta_path, group_col):
    """
    Load metadata, ensuring group_col exists.
    Returns:
        pd.Series of group IDs indexed by sample ID
    """
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
    if group_col not in meta.columns:
        sys.exit(f"Error: group column '{group_col}' not found in metadata")
    return meta[group_col].rename('group')


def extract_features(chisq_file, selected_features):
    """Extract selected features from full chisq matrix."""
    usecols = ['Unnamed: 0'] + selected_features  # 'Unnamed: 0' ensures the index column is loaded
    df = pd.read_csv(chisq_file, sep='\t', usecols=usecols, index_col=0)
    return df

def main():
    args = parse_arguments()

    # 1. Load selected feature list and labels
    print(f"Loading MUVR-selected features from {args.muvr_file}")
    features,labels = load_selected(args.muvr_file, args.label)

    # 2. Load metadata groups
    print(f"Loading metadata (group IDs) from {args.metadata}")
    groups = load_metadata(args.metadata, args.group_column)

    # 3. Extract features from full matrix
    print(f"Extracting {len(selected_features)} features from chisq matrix")
    chisq_features = extract_features(args.chisq_file, features)

    # 4. Merge into one table
    print("Merging features, labels, and group IDs")
    df = pd.concat([chisq_features, labels, groups], axis=1, join='inner')

    # 5. Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving assembled dataset to {args.output}")
    df.to_csv(args.output, sep='\t')
    print("Done.")


if __name__ == "__main__":
    main()
