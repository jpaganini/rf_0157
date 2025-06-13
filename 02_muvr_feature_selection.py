import numpy as np
import sys
import os
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from py_muvr.feature_selector import FeatureSelector


TARGET_COLUMNS = {
    'binary': 'SYMP H/L',
    'multilabel': 'SYMP'
}

def get_opts_muvr():
    parser = argparse.ArgumentParser(description="Run MUVR-based feature selection on input data.")
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data TSV')
    parser.add_argument('--chisq_file', type=str, required=True, help='Path to chisq features TSV')
    parser.add_argument('--model', type=str, choices=['RFC', 'XGBC'], required=True, help='Model to use: RFC or XGBC')
    parser.add_argument('--class_type', type=str, choices=['binary', 'multilabel'], default='binary',
                        help='Type of classification')
    parser.add_argument('--output', type=str, required=True, help='Output directory for storing results')
    parser.add_argument('--filtered_train_dir', type=str, required=True, help='Directory to save the filtered training data (after deduplication)')
    parser.add_argument('--name', type=str, required=True, help='Base filename for outputs')

    args = parser.parse_args()
    return args.train_data, args.chisq_file, args.model, args.class_type, args.output, args.filtered_train_dir, args.name

def prepare_data_muvr(train_data, filtered_dir):

    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)

    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=['t5', 'SYMP'],
                                                       keep='last')  # remove samples that contain a +

    # Ensure parent directory exists
    os.makedirs(filtered_dir, exist_ok=True)
    filtered_output_path = os.path.join(filtered_dir, f"{name}.tsv")

    train_data_muvr.to_csv(filtered_output_path, sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model,class_type, output_dir,name):

    target_col = TARGET_COLUMNS[class_type]
    train_data_muvr = train_data_muvr[[target_col]]

    train_data_muvr = train_data_muvr.drop(columns=columns_to_drop)

    # Create an iterator for reading chisq_features line by line
    reader_chisq = pd.read_csv(chisq_file, sep='\t', header=0, iterator=True, chunksize=1)

    # Create a dataframe to hold the results
    model_input = pd.DataFrame()

    print("Loading chisq feateres")
    # Get the first line of chisq_features
    try:
        chunk_chisq = next(reader_chisq)
    except StopIteration:
        chunk_chisq = pd.DataFrame()

    while not chunk_chisq.empty:
        # Set the index as the first column
        chunk_chisq.set_index(chunk_chisq.columns[0], inplace=True)
        chunk_chisq = chunk_chisq.astype("int8")

        # Merge the current line with isolate_metadata based on your desired criteria
        merged_line = pd.merge(train_data_muvr, chunk_chisq, left_index=True, right_index=True, how='inner')
        #print(merged_line)
        model_input = pd.concat([model_input, merged_line], ignore_index=False)

        #Get the following lines of the dataframe
        try:
            chunk_chisq = next(reader_chisq)
        except StopIteration:
            chunk_chisq = pd.DataFrame()

    if class_type == "multilabel":
        to_predict = ['SYMP']
        X_muvr = model_input.drop('SYMP', axis = 1).to_numpy()
        y_muvr = model_input['SYMP'].values.ravel()
        feature_names = model_input.drop(columns=["SYMP"]).columns

    else:
        print("Binary")
        to_predict = ['SYMP H/L']
        X_muvr = model_input.drop('SYMP H/L', axis = 1).to_numpy()
        y_muvr = model_input['SYMP H/L'].values.ravel()
        feature_names = model_input.drop(columns=["SYMP H/L"]).columns

    if model=='XGBC':
        encoder = OneHotEncoder(sparse=False)

    # Reshape y to a 2D array as fit_transform expects a 2D array
        y_encoded = encoder.fit_transform(np.array(y_muvr).reshape(-1, 1))
        y_variable = y_encoded

    elif model=='RFC':
        y_variable = y_muvr

    else:
        sys.exit("Select a valid model: RFC or XGBC")

    feature_selector = FeatureSelector(
        n_repetitions=10,
        n_outer=5,
        n_inner=4,
        estimator=model,
        metric="MISS",
        features_dropout_rate=0.9
    )

    print("Running MUVR")
    feature_selector.fit(X_muvr, y_variable)
    selected_features = feature_selector.get_selected_features(feature_names=feature_names)

    # Obtain a dataframe containing MUVR selected features
    df_muvr_min = model_input[to_predict+list(selected_features.min)]
    df_muvr_mid = model_input[to_predict+list(selected_features.mid)]
    df_muvr_max = model_input[to_predict+list(selected_features.max)]

    #Write features to a new file.
    output_path = os.path.join(output_dir, class_type)
    os.makedirs(output_path, exist_ok=True)
    min_features_file_name = os.path.join(output_path, f'{name}_muvr_{model}_min.tsv')
    mid_features_file_name = os.path.join(output_path, f'{name}_muvr_{model}_mid.tsv')
    max_features_file_name = os.path.join(output_path, f'{name}_muvr_{model}_max.tsv')

    df_muvr_min.to_csv(min_features_file_name, sep='\t')
    df_muvr_mid.to_csv(mid_features_file_name, sep='\t')
    df_muvr_max.to_csv(max_features_file_name, sep='\t')

    return df_muvr_min,df_muvr_mid,df_muvr_max

def feature_extraction(muvr_features_file_draft, chisq_file, model, class_type, feature_size):
    #features_columns = pd.read_csv(muvr_features_file_draft, sep='\t', header=0, index_col=0).columns[1:].tolist()
    features_columns=muvr_features_file_draft.columns[1:].tolist() 
    #Get column names
    features_columns = ['Unnamed: 0'] + features_columns

    #Get data from chisq file
    chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=features_columns)

    #Write the data
    complete_features_file_name = f'data/03_muvr_features/{class_type}/2023_jp_muvr_complete_{model}_{feature_size}.tsv'  # Using f-string formatting
    chisq_data.to_csv(complete_features_file_name, sep='\t')


#######################################################
#
#                  MAIN                               #
#
#######################################################
if __name__ == "__main__":
    train_data, chisq_file, model, class_type, output_dir,filtered_train_dir, name = get_opts_muvr()

    print("Filtering data")
    train_data_muvr = prepare_data_muvr(train_data, filtered_train_dir, name)

    print("MUVR feature reduction")
    min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file = feature_reduction(
        train_data_muvr, chisq_file, model, class_type, output_dir, name)


#4. FEATURE EXTRACTION STEP
    #Extract relevant features from all samples
#4.1 MULTILABEL
#feature_extraction(min_muvr_filtered_file,chisq_file,"RFC","multilabel","min")
#Mid features
#feature_extraction(mid_muvr_filtered_file,chisq_file,"RFC","multilabel","mid")
#Max Features
#feature_df = feature_extraction(max_muvr_filtered_file,chisq_file,"RFC","multilabel","max")
#4.2 BINARY LABELS
#Min features
#feature_extraction(min_muvr_filtered_file,chisq_file,"RFC","binary","min")
#Mid features
#feature_extraction(mid_muvr_filtered_file,chisq_file,"RFC","binary","mid")
#Max Features
#feature_extraction(max_muvr_filtered_file,chisq_file,"RFC","binary","max")

