import numpy as np
import random
from collections import Counter
import itertools
from scipy import interp
from itertools import cycle
import sys
import math

# pandas
import pandas as pd

from sklearn.preprocessing import OneHotEncoder



# model evaluation
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import roc_curve, auc,  RocCurveDisplay
from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#saving models
import joblib


###### ------- MUVR ------- ######
from py_muvr.feature_selector import FeatureSelector
from concurrent.futures import ProcessPoolExecutor
def get_opts_muvr():
    if len(sys.argv) != 4:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3]

def prepare_data_muvr(train_data):

    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)

    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=['t5', 'SYMP'],
                                                       keep='last')  # remove samples that contain a +

    train_data_muvr.to_csv(r'data/02_test_train_split/2023_train_data_filtered_test.tsv', sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model,class_type):

    #train_data_muvr = pd.read_csv(train_data_muvr, sep='\t', header=0)
    train_data_muvr= train_data_muvr.set_index('SRA')
    if class_type == 'multilabel':
        columns_to_drop = ['MOLIS', 'LINEAGE','STX','SNP ADDRESS','t5','SYMP H/L']  # Replace with the actual column names
    else:
        columns_to_drop = ['MOLIS', 'LINEAGE', 'STX', 'SNP ADDRESS', 't5', 'SYMP']
    train_data_muvr = train_data_muvr.drop(columns=columns_to_drop)

    # Create an iterator for reading chisq_features line by line
    reader_chisq = pd.read_csv(chisq_file, sep='\t', header=0, iterator=True, chunksize=1)

    # Create a dataframe to hold the results
    model_input = pd.DataFrame()

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
        to_predict = ['SYMP']
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
        print ("Select a valid model: RFC or XBGC")
        SystemExit


    feature_selector = FeatureSelector(
        n_repetitions=10,
        n_outer=5,
        n_inner=4,
        estimator=model,
        metric="MISS",
        features_dropout_rate=0.9
    )

    feature_selector.fit(X_muvr, y_variable)
    selected_features = feature_selector.get_selected_features(feature_names=feature_names)

    # Obtain a dataframe containing MUVR selected features
    df_muvr_min = model_input[to_predict+list(selected_features.min)]
    df_muvr_mid = model_input[to_predict+list(selected_features.mid)]
    df_muvr_max = model_input[to_predict+list(selected_features.max)]

    #Write features to a new file.
    min_features_file_name = f'data/03_muvr_features/{class_type}/2023_jp_muvr_{model}_min.tsv'  # Using f-string formatting
    df_muvr_min.to_csv(min_features_file_name, sep='\t')
    mid_features_file_name = f'data/03_muvr_features/{class_type}/2023_jp_muvr_{model}_mid.tsv'  # Using f-string formatting
    df_muvr_mid.to_csv(mid_features_file_name, sep='\t')
    max_features_file_name = f'data/03_muvr_features/{class_type}/2023_jp_muvr_{model}_max.tsv'  # Using f-string formatting
    df_muvr_max.to_csv(max_features_file_name, sep='\t')

    return df_muvr_min,df_muvr_mid,df_muvr_max

def feature_extraction(muvr_features_file_draft, chisq_file, model, class_type, feature_size):
    features_columns = pd.read_csv(muvr_features_file_draft, sep='\t', header=0, index_col=0).columns[1:].tolist()

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
#1. Load the data needed to run the model
train_data, chisq_file,model= get_opts_muvr()

#1. Create a sub-set of the train set, which will be used for doing muvr.
#In this subset, only one isolate within the same t5 cluster are retained
print("Filtering data")
train_data_muvr=prepare_data_muvr(train_data)

#2. MUVR step
 #This will have to be run on an HPC (size of chisq_file=4GBs)
 #model can refer to: "RFC" or "XGBC"
print ("MUVR feature reduction")
#class_type can be: binary, multilabel
min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, feature_df = feature_reduction(train_data_muvr, chisq_file, "RFC", "multilabel")

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
feature_extraction(min_muvr_filtered_file,chisq_file,"RFC","binary","min")
#Mid features
feature_extraction(mid_muvr_filtered_file,chisq_file,"RFC","binary","mid")
#Max Features
feature_df = feature_extraction(max_muvr_filtered_file,chisq_file,"RFC","binary","max")