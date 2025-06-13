import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
import argparse


########## FUNCTIONS ###############

def get_opts():
    parser = argparse.ArgumentParser(description="Chi2 feature selection from genomic data.")
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV with columns SRA, SYMP)')
    parser.add_argument('--features1', required=True, help='Path to first feature matrix (TSV)')
    parser.add_argument('--features2', required=True, help='Path to second feature matrix (TSV)')
    parser.add_argument('--output_dir', required=True, help='Directory to write output files')
    parser.add_argument('--name', required=True, help='Base name for output files (no extension)')
    parser.add_argument('--length_threshold', type=int, default=80, help='Minimum column name length to keep')
    return parser.parse_args()


def read_meta(meta_file):
    all_pd = pd.read_csv(meta_file, sep='\t', header=0)
    all_pd = all_pd[['SRA', 'SYMP']]
    return all_pd.set_index('SRA')

def load_and_filter_features(feature_file, length_threshold):
    print(f"Loading {feature_file}")
    df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Binarize
    df[df > 0] = 1
    df = df.T

    # Filter by k-mer length
    long_cols = [col for col in df.columns if len(col) >= length_threshold]
    df = df[long_cols]

    # Reduce memory usage
    df = df.astype('int8')
    return df.T

def merge_feature_sets(df1, df2):
    # Identify missing rows in each
    to_add1 = set(df1.index) - set(df2.index)
    to_add2 = set(df2.index) - set(df1.index)

    # Fill missing rows with 0s
    df1 = pd.concat([df1, pd.DataFrame(0, index=to_add2, columns=df1.columns).astype("int8")])
    df2 = pd.concat([df2, pd.DataFrame(0, index=to_add1, columns=df2.columns).astype("int8")])

    # Inner join on index
    return df1.join(df2, how='inner')

def perform_chi2_analysis(merged_df, output_dir, name):
    label_encoder = LabelEncoder()
    merged_df['SYMP'] = label_encoder.fit_transform(merged_df['SYMP'])

    X = merged_df.drop('SYMP', axis=1)
    y = merged_df['SYMP']

    chi_scores = chi2(X, y)

    # Select top 100k features
    selector = SelectKBest(chi2, k=100_000)
    X_kbest = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    print(f'Original feature count: {X.shape[1]}')
    print(f'Reduced feature count (top 100k): {X_kbest.shape[1]}')

    # Save top 100k features
    os.makedirs(output_dir, exist_ok=True)
    out_100k = os.path.join(output_dir, f'{name}_100k.tsv')
    merged_df[selected_features].to_csv(out_100k, sep='\t')

    # Save p-value filtered features (p ≤ 0.05)
    p_values = pd.Series(chi_scores[1], index=X.columns)
    good_cols = p_values[p_values <= 0.05].index
    out_pval = os.path.join(output_dir, f'{name}_pvalue.tsv')
    merged_df[good_cols].to_csv(out_pval, sep='\t')

    print(f'Saved top 100k features to: {out_100k}')
    print(f'Saved p ≤ 0.05 features to: {out_pval}')

########### MAIN #############

if __name__ == "__main__":
    args = get_opts()

    meta_df = read_meta(args.meta)
    features1 = load_and_filter_features(args.features1, args.length_threshold)
    features2 = load_and_filter_features(args.features2, args.length_threshold)

    combined = merge_feature_sets(features1, features2)
    combined = combined.T

    print("Merging with metadata")
    merged_with_meta = pd.merge(combined, meta_df, left_index=True, right_index=True)

    perform_chi2_analysis(merged_with_meta, args.output_dir, args.name)



