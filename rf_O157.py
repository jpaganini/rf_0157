# IMPORT PACKAGE 

# Stats
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

# visualisation
import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython.display import Image
#from subprocess import call
#from sklearn.tree import export_graphviz
#import pydot
#from yellowbrick.model_selection import FeatureImportances
#from sklearn.linear_model import Lasso

# modelling 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBClassifier
#from bayes_opt import BayesianOptimization


# model evaluation
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import roc_curve, auc,  RocCurveDisplay
from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#saving models
import joblib


# cross validation 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

###### ------- MUVR ------- ######
#from py_muvr.feature_selector import FeatureSelector
from concurrent.futures import ProcessPoolExecutor

import shap


def usage():

    sys.exit()


def get_opts_muvr():
    if len(sys.argv) != 4:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3]


def get_opts_extract():
    if len(sys.argv) != 5:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4]

def get_opts_model():
    if len(sys.argv) != 7:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], sys.argv[6]

def load_feat_ann(feature_file):
    label_kmer_df = pd.read_csv(feature_file, sep='\t')
    return label_kmer_df

def load_features(feature_file,meta_file):
    
    print ("load feature file")
    kmer_df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0)

    print ("load metadata file")
    metadata = pd.read_csv(meta_file, sep='\t', header=0, index_col=0)

    print("merging data")
    merge_pd =pd.merge(kmer_df,metadata, left_index=True, right_index=True)



    #merge_pd.loc[merge_pd['SYMP'] == 'HUS', 'SYMP'] = 'BD'


    #label_kmer_df = label_kmer_df.set_index('SRA')

    #labels = label_kmer_df['SYMP']

    #label_kmer_df = label_kmer_df.astype(float)

    #change all values > 1 to 1
    #label_kmer_df[label_kmer_df > 1.0] = 1
    #label_kmer_df[label_kmer_df < 1.0] = 0

    #remove kmers < 80bp
    # keep only kmer with > 80 dna base
    #to_keep = []

    #for i in label_kmer_df.columns:
    #    if len(i) >= 80:
    #        to_keep.append(i)   
    # change dataframe to include those kmers only 
    #label_kmer_df = label_kmer_df[to_keep].copy()

    #label_kmer_df = label_kmer_df.join(labels)
    #print (label_kmer_df)

    return merge_pd

def split_dataset(meta_file):
        RSEED=50
        #Load the features table
        #label_kmer_df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0) - #REMOVED WHEN MUVR
        #label_kmer_df = label_kmer_df.drop('SYMP',axis=1) - #REMOVED WHEN MUVR

        #load the metadata
        metadata = pd.read_csv(meta_file, sep='\t', header=0)
        metadata = metadata.set_index('SRA')
        #metadata=metadata.iloc[:,:16] - #REMOVED WHEN MUVR

        #all_pd = pd.merge(label_kmer_df, metadata, left_index=True, right_index=True) - #REMOVED WHEN MUVR

        #Create new dataframes for each sublineage
        lineage_categories = metadata['LINEAGE'].unique()
            #Create an empty dictoniary to store the DataFrames
        lineage_dfs = {}

            # Iterate over each 'LINEAGE' category and create a separate DataFrame for each
        for lineage in lineage_categories:
            mask = metadata['LINEAGE'] == lineage
            lineage_dfs[lineage] = metadata[mask]

        #Create an empty dictoniary to store the train-test splits
        train_test_splits={}

        #Iterate over the LINEAGE dictonary
        for lineage in lineage_categories:
            #Select the correct lineage dataframe
            current_lineage_df = lineage_dfs[lineage]
            #stablish the label column
            labels = np.array(current_lineage_df['SYMP'])
            #stablish the t5 clusters
            t5=np.array(current_lineage_df['t5'])

            #Create the iterator for the Splits - here we use 5 splits to get an approximate 80/20 split
            sfgs=StratifiedGroupKFold(n_splits=5)
            cv_iterator = list(sfgs.split(current_lineage_df, labels, groups=t5))

            #randomly chose one of the splits
            random.seed(42)
            random_split=random.choice(cv_iterator)

            #Get two different dataframes with the test and train data.
            test_data = current_lineage_df.iloc[random_split[1]]
            train_data = current_lineage_df.iloc[random_split[0]]
            #Add everything to a dictonary
            train_test_splits[lineage] = train_data,test_data


#Combine all train and test data into a single dataset
        train_dfs=train_dfs = [split[0] for split in train_test_splits.values()]
        final_train = pd.concat(train_dfs, ignore_index=False)

        test_dfs = train_dfs = [split[1] for split in train_test_splits.values()]
        final_test = pd.concat(test_dfs, ignore_index=False)

        #THIS WILL BE REMOVED===============
        filtered_train = final_train[final_train['SYMP'] == 'HUS']

        filtered_test = final_test[final_test['SYMP'] == 'HUS']

        specific_columns = ['LINEAGE', 'SYMP', 't5']

        for column in specific_columns:
            if column in final_test.columns:
                value_counts = final_test[column].value_counts()
                print(f"Metric Test Set: {column}\n{value_counts}\n")
            else:
                print(f"Column {column} not found in the DataFrame.\n")

        for column in specific_columns:
            if column in final_train.columns:
                value_counts = final_train[column].value_counts()
                print(f"Metric Train Set: {column}\n{value_counts}\n")
            else:
                print(f"Column {column} not found in the DataFrame.\n")

        for column in specific_columns:
            if column in filtered_train.columns:
                value_counts = filtered_train[column].value_counts()
                print(f"Metric Filtered Train Set: {column}\n{value_counts}\n")
            else:
                print(f"Column {column} not found in the DataFrame.\n")

        for column in specific_columns:
            if column in filtered_test.columns:
                value_counts = filtered_test[column].value_counts()
                print(f"Metric Filtered Test Set: {column}\n{value_counts}\n")
            else:
                print(f"Column {column} not found in the DataFrame.\n")

        final_train.to_csv(r'2023_train_set_jp.tsv', sep='\t')
        final_test.to_csv(r'2023_test_set_jp.tsv', sep='\t')

        return final_train, final_test

def prepare_data_muvr(train_data):

    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)

    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=['t5', 'SYMP'],
                                                       keep='last')  # remove samples that contain a +

    train_data_muvr.to_csv(r'2023_train_data_filtered.tsv', sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model):

    train_data_muvr = pd.read_csv(train_data_muvr, sep='\t', header=0)
    train_data_muvr= train_data_muvr.set_index('SRA')
    columns_to_drop = ['MOLIS', 'LINEAGE','STX','SNP ADDRESS','t5','SYMP H/L']  # Replace with the actual column names
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

    to_predict = ['SYMP']

    X_muvr = model_input.drop('SYMP', axis = 1).to_numpy()
    y_muvr = model_input['SYMP'].values.ravel()

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



    feature_names = model_input.drop(columns=["SYMP"]).columns


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

    print('something')

    #df_muvr_min.to_csv(r'2023_jp_muvr_min.tsv', sep='\t')
    #df_muvr_mid.to_csv(r'2023_jp_muvr_mid.tsv', sep='\t')
    #df_muvr_max.to_csv(r'2023_jp_muvr_max.tsv', sep='\t')

    return df_muvr_max

def feature_extraction(min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file):
    min_features_columns = pd.read_csv(min_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()
    mid_features_columns = pd.read_csv(mid_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()
    max_features_columns = pd.read_csv(max_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()

    #Get column names
    min_features_columns = ['Unnamed: 0'] + min_features_columns
    mid_features_columns = ['Unnamed: 0'] + mid_features_columns
    max_features_columns = ['Unnamed: 0'] + max_features_columns

    #Get data from chisq file
    min_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=min_features_columns)
    mid_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=mid_features_columns)
    max_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=max_features_columns)


    min_chisq_data.to_csv(r'2023_jp_complete_muvr_min.tsv', sep='\t')
    mid_chisq_data.to_csv(r'2023_jp_complete_muvr_mid.tsv', sep='\t')
    max_chisq_data.to_csv(r'2023_jp_complete_muvr_max.tsv', sep='\t')


def tune_group_oversampling(model_name, model_input, sampling, block_strategy, fit_parameter, features_sizes):
    RSEED = 50

    #clean up the data
    train_labels = np.array(model_input['SYMP'])
    train= model_input.iloc[:, :-7]

    #Set Model
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=RSEED)

    if model_name == 'XGBC':
        # Encode the labels
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(train_labels)
        #Calculate sample weights - they will only be used when no oversampling happens
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=train_labels
        )
        # Define the label
        model = XGBClassifier(objective='multi:softmax',random_state=RSEED)



    #SET UP GRID VALUES FOR HYPER-PARAMETER TUNNING
    #===RF-SPECIFIC

        # number of features at every split
    max_features = ['log2', 'sqrt']

        # max depth
    max_depth = [int(x) for x in np.linspace(100, 500, num=11)]

    #===SMOTE
        #K-neighbors for smote
    k_neighbors = [1,2,3,4]

    #==XGBC
    eta = np.linspace(0.01, 0.2, 10)
    gamma = [0,3,5,7,9]
    max_depth_xgbc = [3,4,5,6,7,8,9,10]
    min_child_weight = [1,2,3,4,5]
    subsample = [0.6, 0.7, 0.8, 0.9, 1]
    #scale_pos_weight = sample_weights #only for binary classification
    colsample_bytree = [0.7, 0.8, 0.9, 1]

    #==COMMON MODEL PARAMETERS
    # number of trees
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]



    #Create a imblearn Pipeline to tune hyper-parameters with oversampling included

    # Oversampling strategy, random grid and Pipeline
    if sampling == 'random':
        oversampler = RandomOverSampler(random_state=RSEED)
        # create random grid
        if model_name == 'RF':
            random_grid = {
                'model__n_estimators': n_estimators,
                'model__max_features': max_features,
                'model__max_depth': max_depth
            }

        if model_name == 'XGBC':
            random_grid = {
                'model__eta': eta,
                'model__gamma': gamma,
                'model__max_depth': max_depth_xgbc,
                'model__min_child_weight': min_child_weight,
                'model__subsample': subsample,
                'model__colsample_bytree': colsample_bytree,
                'model__n_estimators': n_estimators
            }

        tunning_pipeline = Pipeline([
            ('oversampler', oversampler),
            ('model', model)
        ])

    if sampling == 'smote':
        oversampler = SMOTE(random_state=RSEED)
        # create random grid
        if model_name == 'RF':
            random_grid = {
                'oversampler__k_neighbors': k_neighbors,
                'model__n_estimators': n_estimators,
                'model__max_features': max_features,
                'model__max_depth': max_depth
            }

        if model_name == 'XGBC':
            random_grid = {
                'oversampler__k_neighbors': k_neighbors,
                'model__eta': eta,
                'model__gamma': gamma,
                'model__max_depth': max_depth_xgbc,
                'model__min_child_weight': min_child_weight,
                'model__subsample': subsample,
                'model__colsample_bytree': colsample_bytree,
                'model__n_estimators': n_estimators
            }


        tunning_pipeline = Pipeline([
            ('oversampler', oversampler),
            ('model', model)
        ])

    if sampling == 'none':
        if model_name == 'RF':
            # create random grid
            random_grid = {
                'model__n_estimators': n_estimators,
                'model__max_features': max_features,
                'model__max_depth': max_depth
        }
            tunning_pipeline = Pipeline([
                ('model', model)
            ])

        if model_name == 'XGBC':
            random_grid = {
                'eta': eta,
                'gamma': gamma,
                'max_depth': max_depth_xgbc,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'n_estimators': n_estimators
            }

            tunning_pipeline = model
            #tunning_pipeline = Pipeline([
            #    ('model', model)
            #])




    #create iterator list according to the blocking strategy
    if block_strategy=='t5':
        groups = np.array(model_input['t5'])
        sfgs = StratifiedGroupKFold(n_splits=10)
        cv_iterator = list(sfgs.split(train, train_labels, groups=groups))

    if block_strategy=='lineage':
        groups = np.array(model_input['LINEAGE'])
        logo = LeaveOneGroupOut()
        cv_iterator = list(logo.split(train, train_labels, groups=groups))


    # Fitting the pipeline and obtain the best parameters using random-search
    #Determine the scoring strategy
    scoring_strategy = ['accuracy', 'balanced_accuracy']

    #GridSearch
    model_tunning = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions=random_grid, n_iter=100,
                                       cv=cv_iterator, scoring=scoring_strategy, refit=fit_parameter, verbose=2,
                                       random_state=RSEED, n_jobs=-1)

    if model_name == 'XGBC' and sampling=='none':
        model_tunning.fit(train, train_labels, sample_weight=sample_weights, verbose=False)

    else:
        model_tunning.fit(train, train_labels)



    best_params= model_tunning.best_params_
    best_score=model_tunning.best_score_
    cv_results=pd.DataFrame(model_tunning.cv_results_)
    best_model=model_tunning.best_estimator_

    # EXPORT THE CV RESULTS
    cv_file_name = f'results/03_cv/{model_name}_{sampling}_{block_strategy}_{features_sizes}.tsv'  # Using f-string formatting
    cv_results.to_csv(cv_file_name, sep='\t')


    #Filter the results of the classifier
    best_index = model_tunning.best_index_
    best_cv_results=pd.DataFrame(cv_results.iloc[best_index,:])

    #EXPORT THE ML MODEL
    model_file_name = f'results/04_models/{fit_parameter}/{model_name}_{sampling}_{block_strategy}_{features_sizes}.joblib'  # Using f-string formatting
    joblib.dump(best_model, model_file_name)

    return best_params,best_cv_results,best_model


    # print results of the best parameters
    #best_param= model_tunning.best_params_
    #best_score=model_tunning.best_score_
    #print("Best params:",best_param)
    #print("Best score:", best_score)
    #return best_param

def cross_model_balanced_blocked(model_input, best_params, label_df, sampling, block_strategy, model_name, features_size, fit_parameter):

    #1. Import the data
    all_labels = np.array(model_input['SYMP'])
    features= model_input.iloc[:,:-7]


    #set up the random seed
    RSEED = 50

    #Create an iterator to separate files in groups
    if block_strategy=='t5':
        groups = np.array(model_input['t5'])
        sgkf=StratifiedGroupKFold(n_splits=10)
        cv_iterator = list(sgkf.split(features, all_labels, groups=groups))

    if block_strategy=='lineage':
        groups = np.array(model_input['LINEAGE'])
        logo = LeaveOneGroupOut()
        cv_iterator = list(logo.split(features, all_labels, groups=lineages))


    #to hold the max accuracy
    acc_muvr_max = []

    # data frame for storing probabilities of classification
    final_res = pd.DataFrame()
    # data frame for storing feature importance
    final_imp = pd.DataFrame()
    list_shap_values = list()
    list_test_sets = pd.DataFrame()

    # Test-set true lables
    list_test_labels = []
    # Test-set predictions
    list_test_pred = []

    for x in range(len(cv_iterator)):
        test = model_input.iloc[cv_iterator[x][1]]
        samples = test.index.values.tolist()
        test_features = test.iloc[:,:-7]
        test_labels = test['SYMP'].values.ravel()

        train = model_input.iloc[cv_iterator[x][0]]
        train_features = train.iloc[:,:-7]
        train_labels = train['SYMP'].values.ravel()

        if model_name =='XGBC':
            encoder = LabelEncoder()
            test_labels = encoder.fit_transform(test_labels)
            train_labels = encoder.fit_transform(train_labels)

        if model_name == 'RF':
            if sampling=='none':
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               random_state=RSEED,
                                               n_jobs=-1, verbose=1)

            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               random_state=RSEED,
                                               n_jobs=-1, verbose=1)
            if sampling=='smote':
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=1).fit_resample(train_features,train_labels)
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               random_state=RSEED,
                                               n_jobs=-1, verbose=1)

        if model_name == 'XGBC':

            if sampling=='none':
                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=train_labels
                )

                #THIS WILL BE REMOVED#
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=best_params.subsample,
                                      n_estimators=best_params.n_estimators,
                                      min_child_weight=best_params.min_child_weight,
                                      max_depth=best_params.max_depth,
                                      gamma=best_params.gamma,
                                      eta=best_params.kwargs['eta'],
                                      colsample_bytree=best_params.colsample_bytree,
                                      random_state=RSEED,
                                      n_jobs=-1)

                # THIS WILL BE RE-INCORPORATED#
                # model = XGBClassifier(objective='multi:softmax',
                #                       subsample=best_params['subsample'],
                #                       n_estimators=best_params['n_estimators'],
                #                       min_child_weight=best_params['min_child_weight'],
                #                       max_depth=best_params['max_depth'],
                #                       gamma=best_params['gamma'],
                #                       eta=best_params['eta'],
                #                       colsample_bytree=best_params['colsample_bytree'],
                #                       random_state=RSEED,
                #                       n_jobs=-1)

            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=best_params['subsample'],
                                      n_estimators=best_params['n_estimators'],
                                      min_child_weight=best_params['min_child_weight'],
                                      max_depth=best_params['max_depth'],
                                      gamma=best_params['gamma'],
                                      eta=best_params['eta'],
                                      colsample_bytree=best_params['colsample_bytree'],
                                      random_state=RSEED,
                                      n_jobs=-1)
            if sampling=='smote':
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=1).fit_resample(train_features,train_labels)
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=best_params['subsample'],
                                      n_estimators=best_params['n_estimators'],
                                      min_child_weight=best_params['min_child_weight'],
                                      max_depth=best_params['max_depth'],
                                      gamma=best_params['gamma'],
                                      eta=best_params['eta'],
                                      colsample_bytree=best_params['colsample_bytree'],
                                      random_state=RSEED,
                                      n_jobs=-1)

        if sampling == 'none' and model_name=='XGBC':
            model.fit(train_features,train_labels, sample_weight=sample_weights)

        elif sampling == 'none' and model_name=='RF':
            model.fit(train_features,train_labels)

        else:
            model.fit(features_resampled, labels_resampled)

        # test the model on test data
        test_model = model.predict(test_features)
        test_probs = model.predict_proba(test_features) #Formerly: test_rf_probs

        # If model is XGBC, reverse the encoder label
        if model_name == 'XGBC':
            test_labels = encoder.inverse_transform(test_labels)
            test_model = encoder.inverse_transform(test_model)

        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)

        # 4. extract feature importance
        feature_model = pd.DataFrame({'feature': test_features.columns,
                                      x: model.feature_importances_}). \
            sort_values(x, ascending=False)
        feature_model = feature_model.set_index('feature')

        final_imp = pd.concat([final_imp, feature_model], axis=1)

        #Create a dataframe with the predictions and the probabilities
        res_df = pd.DataFrame(
            {'samples': samples, 'labels': test_labels, 'predictions': test_model})

        #Merge with probabilities
        res_df = pd.merge(res_df, pd.DataFrame(test_probs), how='left',
                                  left_index=True, right_index=True)

        col_names = np.array(['samples', 'labels', 'predictions', 'BD', 'D', 'HUS'])
        res_df.columns = col_names

        final_res = pd.concat([final_res, res_df])


        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)

        list_shap_values.append(shap_values)
        list_test_sets = pd.concat([list_test_sets, test_features])

        acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max) / len(acc_muvr_max))

    #WRITE PREDICTIONS OUTPUTS AND THEIR PROBABILITY
    #predictions_file_name = f'results/07_predictions/{fit_parameter}/training_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    #final_res.to_csv(predictions_file_name, sep='\t')

    #WRITE IMPORTANCES OF EACH FEATURE - SORTED BY AVERAGE IMPORTANCE
    final_imp['average'] = final_imp.mean(numeric_only=True, axis=1)
    final_imp = final_imp.sort_values('average', ascending=False)

    #importances_file_name = f'results/06_feature_importances/{fit_parameter}/training_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    #final_imp.to_csv(importances_file_name, sep='\t')

    #   combining results from all iterations
    shap_values = np.array(list_shap_values[0])
    for i in range(1, len(list_shap_values)):
        shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)

    # print shap uncomment
    label_dict = dict(zip(label_df['Sequence'], label_df['ANN']))
    fest = list_test_sets.columns.values.tolist()

    anns = []
    for f in fest:
        anns.append(label_dict[f])

    #print(model.classes_)

    shap.summary_plot(shap_values[0], list_test_sets, feature_names=anns)
    shap.summary_plot(shap_values[1], list_test_sets,feature_names=anns)
    shap.summary_plot(shap_values[2], list_test_sets, feature_names=anns)

    # shap.summary_plot(shap_values[0], list_test_sets)
    # shap.summary_plot(shap_values[1], list_test_sets)
    # shap.summary_plot(shap_values[2], list_test_sets)

    return list_test_pred, list_test_labels, final_imp


def predict_test_set(best_model, val_input, model_name, sampling, block_strategy, features_size, fit_parameter):
    RSEED = 50

    #Clean up the test-data
    test_features = val_input.iloc[:, :-7]
    test_labels = val_input['SYMP'].values.ravel()

    #Encode labels when model is XGBC
    if model_name == 'XGBC':
        encoder = LabelEncoder()
        test_labels = encoder.fit_transform(test_labels)

    #Run model on the test data
    test_predictions = best_model.predict(test_features)
    test_probabilities = best_model.predict_proba(test_features)

    # Get the sample names into a list
    samples = test_features.index.values.tolist()

    #Decode the labels of the test set and predictions if the model is XGBC
    if model_name == 'XGBC':
        test_predictions = encoder.inverse_transform(test_predictions)
        test_labels = encoder.inverse_transform(test_labels)


    # Combine everything to a dataframe
    predictions_df = pd.DataFrame(
        {'samples': samples, 'labels': test_labels, 'predictions': test_predictions})

    predictions_df = pd.merge(predictions_df, pd.DataFrame(test_probabilities), how='left',
                                    left_index=True, right_index=True)
    #Change column names
    col_names = np.array(['samples', 'labels','predictions','BD','D','HUS'])
    predictions_df.columns = col_names

    test_predictions_file_name = f'results/07_predictions/{fit_parameter}/test_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'  # Using f-string formatting
    predictions_df.to_csv(test_predictions_file_name, sep='\t')


    return test_predictions, test_labels


def calc_accuracy(preds, labels, experiment_type, model_name, sampling, block_strategy, features_size, fit_parameter):

    #CREATE AND EXPORT CLASSIFICATION REPORT
    report_ = classification_report(
            digits=6,
            y_true= labels, 
            y_pred= preds,
            output_dict=True)

    classification_report_df = pd.DataFrame(report_).transpose()

    classification_report_file_name = f'results/05_classification_reports/{fit_parameter}/{experiment_type}_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'  # Using f-string formatting
    classification_report_df.to_csv(classification_report_file_name, sep='\t')


    #CREATE AND EXPORT CONFUSION MATRIX
    label_names_array=np.array(['BD','D','HUS'])
    cm = confusion_matrix(labels, preds, labels=label_names_array)
    cm_df=pd.DataFrame(cm, index=label_names_array, columns=label_names_array)

    # Export DataFrame to CSV
    confusion_matrix_file_name=f'results/09_confusion_matrices/{fit_parameter}/{experiment_type}_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    cm_df.to_csv(confusion_matrix_file_name, sep='\t')


    disp = ConfusionMatrixDisplay.from_predictions(cmap=plt.cm.Blues, y_true = labels, y_pred = preds, display_labels=['Bloody Diarrhoea', 'Diarrhoea', 'HUS'], normalize='true', values_format='.1g')
    disp.plot()
    plt.show()

def create_fasta(final_imp,model_name, sampling, block_strategy, features_size, fit_parameter):
    features =  final_imp.index.values.tolist()
    #print (features)
    features_file_name = f'results/08_fasta_features/{fit_parameter}/training_{model_name}_{sampling}_{block_strategy}_{features_size}.fasta'  # Using f-string formatting
    with open(features_file_name, 'w') as f:
        for x, feat in enumerate(features):
            f.write('>Feature'+str(x+1))
            f.write('\n')
            f.write(feat)
            f.write('\n')



################ MAIN ##############

#1. create a train-test split - considering population structre and blocking hihgly similar isoaltes to avoid data-leakage
#train_data,validation_data=split_dataset(meta_file)

#2. Create a sub-set of the train set, which will be used for doing muvr.
    #In this subset, only one isolate within the same t5 cluster are retained
#train_data_muvr=prepare_data_muvr(train_data)

#3. MUVR step
 #This will have to be run on an HPC (size of chisq_file=4GBs)
 #model can refer to: "RFC" or "XGBC"
#train_data_muvr, chisq_file,model= get_opts_muvr()
#print ("MUVR feature reduction")
#feature_df = feature_reduction(train_data_muvr, chisq_file, model)

#4. FEATURE EXTRACTION STEP
    #Extract relevant features from all samples
#min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file = get_opts_extract()
#feature_df = feature_extraction(min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file)

#5. LOAD AND WRANGLE TRAIN DATA FOR MODELS
#print ("load training data")
rf_feature_file, xgbc_feature_file, train_meta_file, test_meta_file, rf_ann_file, xgbc_ann_file = get_opts_model()
#Process RF data
rf_train_data = load_features(rf_feature_file, train_meta_file)
rf_test_data = load_features(rf_feature_file, test_meta_file)
rf_label_df=load_feat_ann(rf_ann_file)
#Process XGBC data
xgbc_train_data = load_features(xgbc_feature_file, train_meta_file)
xgbc_test_data = load_features(xgbc_feature_file, test_meta_file)
xgbc_label_df=load_feat_ann(xgbc_ann_file)


#################################################

###         RF MODEL                          ##

###################################################

##### NO OVERSAMPLING - T5 BLOCKING - MAX FEATURES ####

# ###ACCURACY
##Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'none', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'none','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','accuracy')



###BALANCED ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'none', 't5', 'balanced_accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','RF', 'max', 'balanced_accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','balanced_accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'none','t5','max','balanced_accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','balanced_accuracy')

# #### RANDOM OVERSAMPLING - T5 BLOCKING - MAX FEATURES ####
# # Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'random', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'random','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','random','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'random','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','random','t5','max','accuracy')

#### SMOTE OVERSAMPLING - T5 BLOCKING - MAX FEATURES ####
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'smote', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'smote','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','smote','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'smote','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','smote','t5','max','accuracy')



#################################################

###         old code                         ##

###################################################



#7. ====== EVALUATE RF =======
#7.1 WITH TRAINING DATA - T5 BLOCKING
#final_res_t5, final_imp_t5, preds_t5, labels_t5 = cross_model_balanced_blocked(train_data, label_df,'random','t5','RF')
#calc_accuracy(preds_t5, labels_t5)

#7.2 EXTRACT IMPORTANT FEATURES AS FASTA FILE
#imp_fasta = create_fasta(final_imp_t5)

#7.3 LOAD AND WRANGLE TEST DATA FOR RF - this could be removed
#print ("load test data")
#feature_file, train_meta_file, test_meta_file, ann_file= get_opts_rf()
#train_data = load_features(feature_file, train_meta_file)
#test_data = load_features(feature_file, test_meta_file)
#label_df=load_feat_ann(ann_file)

#7.4. EVALUATE RF WITH TEST DATA - T5 BLOCKING
#test_preds, test_labels = predict_test_set(train_data, test_data)
#calc_accuracy(test_preds, test_labels)
#====

#-Import model - optional
#xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed


#################################################

###         XGBC MODEL                          ##

###################################################


##### NO OVERSAMPLING - T5 BLOCKING - MAX FEATURES ####

##ACCURACY
## Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', xgbc_train_data, 'none', 't5', 'accuracy','max')
# ## Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(xgbc_train_data, best_params, xgbc_label_df,'none','t5','XGBC', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max','accuracy')
# #Predicting the test set
# #Import model - optional
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
# test_preds, test_labels= predict_test_set(best_model, xgbc_test_data, 'XGBC', 'none','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max','accuracy')


##BALANCED ACCURACY
# ## Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', xgbc_train_data, 'none', 't5', 'balanced_accuracy','max')
# ## Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(xgbc_train_data, best_params, xgbc_label_df,'none','t5','XGBC', 'max', 'balanced_accuracy')
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max','balanced_accuracy')
# #Predicting the test set
# #Import model - optional
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
# test_preds, test_labels= predict_test_set(best_model, xgbc_test_data, 'XGBC', 'none','t5','max','balanced_accuracy')
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max','balanced_accuracy')

#xgbc_best_params_random_t5 = tune_group_oversampling('XGBC', train_data, 'random', 't5')
#xgbc_best_params_smote_t5 = tune_group_oversampling('XGBC', train_data, 'smote', 't5')



#################################################

###       XGBC MODEL with RF features        ####

#################################################


##### NO OVERSAMPLING - T5 BLOCKING - MAX FEATURES ####

# #ACCURACY
# # Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'none', 't5', 'accuracy','max_RF')
# ## Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','XGBC', 'max_RF', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','accuracy')
# #Predicting the test set
# #Import model - optional
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'none','t5','max_RF','accuracy')
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','accuracy')


#BALANCED ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'none', 't5', 'balanced_accuracy','max_RF')
# ## Feature importance and training set predictions

#IMPORT MODEL - THIS WILL BE REMOVED
best_params = joblib.load("results/04_models/balanced_accuracy/XGBC_none_t5_max_RF.joblib") #this will be removed
##
train_preds, train_labels, feature_importances= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','XGBC', 'max_RF', 'balanced_accuracy')
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','balanced_accuracy')
# #Predicting the test set
# #Import model - optional
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'none','t5','max_RF','balanced_accuracy')
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','balanced_accuracy')
#Create fasta file with important features
#create_fasta(feature_importances,'XGBC','none','t5','max_RF','balanced_accuracy')







#print ("load validation data")
#val_df = load_features(val_file, meta_file,"V")

#print ("load feature annotation")
#label_df = load_feat_ann(ann_file)

#print ("feature reduction")
#feature_df = feature_reduction(feature_df)

#print ("simple rf")
#simple_rf(feature_df)


#print ("cv rf")
#final_res, final_imp, preds, labels = cross_model_balanced(train_data, label_df)
#calc_accuracy(preds, labels)


#Using t5 as block -


#Using sublineage as block
#final_res_sublineage, final_imp_sublineage, preds_sublineage, labels_sublineage = cross_model_balanced_blocked(train_data, label_df,'random','t5')
#calc_accuracy(preds_sublineage, labels_sublineage)


#imp_fasta = create_fasta(final_imp)

#final_model(feature_df, val_df)




