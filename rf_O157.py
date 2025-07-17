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
#from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#saving models
import joblib


# cross validation 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
#from yellowbrick.model_selection import CVScores
#from yellowbrick.model_selection import RFECV
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

def get_opts_model():
    if len(sys.argv) != 9:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], sys.argv[6],sys.argv[7],sys.argv[8]

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
    return merge_pd



def tune_group_oversampling(model_name, model_input, sampling, block_strategy, fit_parameter, features_sizes, class_type):
    RSEED = 50

    #1. Import and clean up the data
    if class_type=='multilabel':
        train_labels = np.array(model_input['SYMP'])
        train= model_input.iloc[:, :-7]
    else:
        train_labels = np.array(model_input['SYMP H/L'])
        train= model_input.iloc[:, :-7]
    print("something")
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
        if class_type == 'multilabel':
            model = XGBClassifier(objective='multi:softmax',random_state=RSEED)
        else:
            model = XGBClassifier(objective='binary:logistic', random_state=RSEED)




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
    cv_file_name = f'results/03_cv/{class_type}/{model_name}_{sampling}_{block_strategy}_{features_sizes}.tsv'  # Using f-string formatting
    cv_results.to_csv(cv_file_name, sep='\t')


    #Filter the results of the classifier
    best_index = model_tunning.best_index_
    best_cv_results=pd.DataFrame(cv_results.iloc[best_index,:])

    #EXPORT THE ML MODEL
    model_file_name = f'results/04_models/{class_type}/{fit_parameter}/{model_name}_{sampling}_{block_strategy}_{features_sizes}_v2.joblib'  # Using f-string formatting
    joblib.dump(best_model, model_file_name)

    return best_params,best_cv_results,best_model


def cross_model_balanced_blocked(model_input, best_params, label_df, sampling, block_strategy, model_name, features_size, fit_parameter, class_type):
    # set up the random seed
    RSEED = 50

    #1. Import and clean up the data
    if class_type=='multilabel':
        all_labels = np.array(model_input['SYMP'])
    else:
        all_labels = np.array(model_input['SYMP H/L'])

    features= model_input.iloc[:,:-7]

    # Get the sample weights === MIGHT BE REMOVED
    if model_name == 'XGBC':
        if sampling=='none':
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=all_labels
            )

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
    #data frame for storing shap values
    list_shap_values = list()
    #Lis for storing shap values arrays as annotated dataframes
    shap_values_bd_list = []
    shap_values_d_list = []
    shap_values_hus_list = []
    #Data frame for storing test sets in each iteration
    list_test_sets = pd.DataFrame()

    # Test-set true lables
    list_test_labels = []
    # Test-set predictions
    list_test_pred = []

    for x in range(len(cv_iterator)):
        test = model_input.iloc[cv_iterator[x][1]]
        samples = test.index.values.tolist()
        test_features = test.iloc[:,:-7]
        if class_type=='multilabel':
            test_labels = test['SYMP'].values.ravel()
        else:
            test_labels = test['SYMP H/L'].values.ravel()

        train = model_input.iloc[cv_iterator[x][0]]
        train_features = train.iloc[:,:-7]
        if class_type == 'multilabel':
            train_labels = train['SYMP'].values.ravel()
        else:
            train_labels = train['SYMP H/L'].values.ravel()

        if model_name =='XGBC':
            encoder = LabelEncoder()
            test_labels = encoder.fit_transform(test_labels)
            train_labels = encoder.fit_transform(train_labels)

        if model_name == 'RF':
            if sampling=='none':
                #THIS WILL BE REMOVED
                # model = RandomForestClassifier(n_estimators=best_params._final_estimator.n_estimators,
                #                                max_features=best_params._final_estimator.max_features,
                #                                max_depth=best_params._final_estimator.max_depth,
                #                                random_state=RSEED,
                #                                n_jobs=-1, verbose=1)
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
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=best_params['oversampler__k_neighbors']).fit_resample(train_features,train_labels)
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               random_state=RSEED,
                                               n_jobs=-1, verbose=1)

        if model_name == 'XGBC':

            if sampling=='none':
                #Get the sample weights for the training set
                sample_weights_train = sample_weights[cv_iterator[x][0]]

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

                ##THIS WILL BE RE-INCORPORATED#
                # if class_type=='multilabel':
                #     model = XGBClassifier(objective='multi:softmax',
                #                           subsample=best_params['subsample'],
                #                           n_estimators=best_params['n_estimators'],
                #                           min_child_weight=best_params['min_child_weight'],
                #                           max_depth=best_params['max_depth'],
                #                           gamma=best_params['gamma'],
                #                           eta=best_params['eta'],
                #                           colsample_bytree=best_params['colsample_bytree'],
                #                           random_state=RSEED,
                #                           n_jobs=-1)
                # else:
                #     model = XGBClassifier(objective='binary:logistic',
                #                           subsample=best_params['subsample'],
                #                           n_estimators=best_params['n_estimators'],
                #                           min_child_weight=best_params['min_child_weight'],
                #                           max_depth=best_params['max_depth'],
                #                           gamma=best_params['gamma'],
                #                           eta=best_params['eta'],
                #                           colsample_bytree=best_params['colsample_bytree'],
                #                           random_state=RSEED,
                #                           n_jobs=-1)


            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=best_params['model__subsample'],
                                      n_estimators=best_params['model__n_estimators'],
                                      min_child_weight=best_params['model__min_child_weight'],
                                      max_depth=best_params['model__max_depth'],
                                      gamma=best_params['model__gamma'],
                                      eta=best_params['model__eta'],
                                      colsample_bytree=best_params['model__colsample_bytree'],
                                      random_state=RSEED,
                                      n_jobs=-1)
            if sampling=='smote':
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=best_params['oversampler__k_neighbors']).fit_resample(train_features,train_labels)
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=best_params['model__subsample'],
                                      n_estimators=best_params['model__n_estimators'],
                                      min_child_weight=best_params['model__min_child_weight'],
                                      max_depth=best_params['model__max_depth'],
                                      gamma=best_params['model__gamma'],
                                      eta=best_params['model__eta'],
                                      colsample_bytree=best_params['model__colsample_bytree'],
                                      random_state=RSEED,
                                      n_jobs=-1)

        if sampling == 'none' and model_name=='XGBC':
            model.fit(train_features,train_labels, sample_weight=sample_weights_train)

        elif sampling == 'none' and model_name=='RF':
            model.fit(train_features,train_labels)

        else:
            model.fit(features_resampled, labels_resampled)

        # test the model on test data
        test_model = model.predict(test_features)
        test_probs = model.predict_proba(test_features) #Formerly: test_rf_probs

        # Calculate accuracy of the model
        acc_muvr_max += [model.score(test_features, test_labels)]

        # If model is XGBC, reverse the encoder label
        if model_name == 'XGBC':
            test_labels = encoder.inverse_transform(test_labels)
            test_model = encoder.inverse_transform(test_model)

        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)

        # 4. extract feature importances  (Gini index)
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

        if class_type=='multilabel':
            col_names = np.array(['samples', 'labels', 'predictions', 'BD', 'D', 'HUS'])
        else:
            col_names = np.array(['samples', 'labels', 'predictions', 'H','L'])

        res_df.columns = col_names
        final_res = pd.concat([final_res, res_df])

        #Get shapley values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)
        #Append them to the
        list_shap_values.append(shap_values)

        #Separate the resulting shap array into individual dataframes
        shap_values_bd_tmp = pd.DataFrame(shap_values[0], index=test_features.index, columns=test_features.columns)
        shap_values_d_tmp = pd.DataFrame(shap_values[1], index=test_features.index, columns=test_features.columns)
        shap_values_hus_tmp = pd.DataFrame(shap_values[2], index=test_features.index, columns=test_features.columns)

        # Append the current iteration's DataFrames to the lists
        shap_values_bd_list.append(shap_values_bd_tmp)
        shap_values_d_list.append(shap_values_d_tmp)
        shap_values_hus_list.append(shap_values_hus_tmp)

        # Concatenate the test data from the current iteration
        list_test_sets = pd.concat([list_test_sets, test_features])

        # acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max) / len(acc_muvr_max))

    #WRITE PREDICTIONS OUTPUTS AND THEIR PROBABILITY

    predictions_file_name = f'results/07_predictions/{class_type}/{fit_parameter}/training_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    final_res.to_csv(predictions_file_name, sep='\t')

    #WRITE IMPORTANCES OF EACH FEATURE - SORTED BY AVERAGE IMPORTANCE
    final_imp['average'] = final_imp.mean(numeric_only=True, axis=1)
    final_imp = final_imp.sort_values('average', ascending=False)

    importances_file_name = f'results/06_feature_importances/{class_type}/{fit_parameter}/training_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    final_imp.to_csv(importances_file_name, sep='\t')

    #   combining SHAP values arrays from all iterations
    ##This will be used for plotting - see below.
    if class_type=='binary':
        shap_values = np.array(list_shap_values[0])
        for i in range(1, len(list_shap_values)):
            shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)
    else:
        shap_values = np.array(list_shap_values[0])
        for i in range(1, len(list_shap_values)):
            shap_values = np.concatenate((shap_values, np.array(list_shap_values[i])), axis=1)



    # Add
    label_dict = dict(zip(label_df['Sequence'], label_df['ANN']))
    fest = list_test_sets.columns.values.tolist()

    anns = []
    for f in fest:
        anns.append(label_dict[f])

    if class_type=='binary':
        shap.summary_plot(shap_values, list_test_sets, feature_names=anns)

    else:
        shap.summary_plot(shap_values[0], list_test_sets, feature_names=anns)
        shap.summary_plot(shap_values[1], list_test_sets,feature_names=anns)
        shap.summary_plot(shap_values[2], list_test_sets, feature_names=anns)


    #   combining SHAP values individual dataframes from all iterations
    ##This will be used for co-occurence analysis (in R)
    shap_values_bd = pd.concat(shap_values_bd_list, axis=0)
    shap_values_d = pd.concat(shap_values_d_list, axis=0)
    shap_values_hus = pd.concat(shap_values_hus_list, axis=0)
    #Write out these dataframes
    shap_bd_file_name = f'results/06_feature_importances/{class_type}/{fit_parameter}/SHAP_BD_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    shap_d_file_name = f'results/06_feature_importances/{class_type}/{fit_parameter}/SHAP_D_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    shap_hus_file_name = f'results/06_feature_importances/{class_type}/{fit_parameter}/SHAP_HUS_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    shap_values_bd.to_csv(shap_bd_file_name, sep='\t')
    shap_values_d.to_csv(shap_d_file_name, sep='\t')
    shap_values_hus.to_csv(shap_hus_file_name, sep='\t')

    return list_test_pred, list_test_labels


def predict_test_set(best_model, val_input, model_name, sampling, block_strategy, features_size, fit_parameter, class_type):
    RSEED = 50

    #Clean up the test-data
    test_features = val_input.iloc[:, :-7]
    if class_type=='multilabel':
        test_labels = val_input['SYMP'].values.ravel()
    else:
        test_labels = val_input['SYMP H/L'].values.ravel()

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
    if class_type =='multilabel':
        col_names = np.array(['samples', 'labels','predictions','BD','D','HUS'])
    else:
        col_names = np.array(['samples', 'labels', 'predictions', 'H', 'L'])

    predictions_df.columns = col_names

    test_predictions_file_name = f'results/07_predictions/{class_type}/{fit_parameter}/test_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'  # Using f-string formatting
    predictions_df.to_csv(test_predictions_file_name, sep='\t')


    return test_predictions, test_labels


def calc_accuracy(preds, labels, experiment_type, model_name, sampling, block_strategy, features_size, fit_parameter, class_type):

    #CREATE AND EXPORT CLASSIFICATION REPORT
    report_ = classification_report(
            digits=6,
            y_true= labels, 
            y_pred= preds,
            output_dict=True)

    classification_report_df = pd.DataFrame(report_).transpose()

    classification_report_file_name = f'results/05_classification_reports/{class_type}/{fit_parameter}/{experiment_type}_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'  # Using f-string formatting
    classification_report_df.to_csv(classification_report_file_name, sep='\t')


    #CREATE AND EXPORT CONFUSION MATRIX
    if class_type=='multilabel':
        label_names_array=np.array(['BD','D','HUS'])
    else:
        label_names_array = np.array(['H', 'L'])

    cm = confusion_matrix(labels, preds, labels=label_names_array)
    cm_df=pd.DataFrame(cm, index=label_names_array, columns=label_names_array)

    # Export DataFrame to CSV
    confusion_matrix_file_name = f'results/09_confusion_matrices/{class_type}/{fit_parameter}/{experiment_type}_{model_name}_{sampling}_{block_strategy}_{features_size}.tsv'
    cm_df.to_csv(confusion_matrix_file_name, sep='\t')

    if class_type == 'multilabel':
        disp = ConfusionMatrixDisplay.from_predictions(cmap=plt.cm.Blues, y_true = labels, y_pred = preds,
                                                       display_labels=['Bloody Diarrhoea', 'Diarrhoea', 'HUS'],
                                                       normalize='true', values_format='.1g')
    else:
        disp = ConfusionMatrixDisplay.from_predictions(cmap=plt.cm.Blues, y_true=labels, y_pred=preds,
                                                       display_labels=['High Risk', 'Low Risk'],
                                                       normalize='true', values_format='.1g')

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

#5. LOAD AND WRANGLE TRAIN DATA FOR MODELS
#print ("load training data")

rf_feature_file, rf_feature_file_binary, xgbc_feature_file, train_meta_file, test_meta_file, rf_ann_file, rf_ann_file_binary, xgbc_ann_file = get_opts_model()
#Process RF data multilabel
rf_train_data = load_features(rf_feature_file, train_meta_file)
rf_test_data = load_features(rf_feature_file, test_meta_file)
rf_label_df=load_feat_ann(rf_ann_file)
#Process RF data binary
rf_train_data_binary = load_features(rf_feature_file_binary, train_meta_file)
rf_test_data_binary = load_features(rf_feature_file_binary, test_meta_file)
rf_label_df_binary=load_feat_ann(rf_ann_file_binary)

#Process XGBC data
xgbc_train_data = load_features(xgbc_feature_file, train_meta_file)
xgbc_test_data = load_features(xgbc_feature_file, test_meta_file)
xgbc_label_df=load_feat_ann(xgbc_ann_file)


#################################################

###         RF MODEL                          ##

###################################################

###====== MULTILABEL ====================####

#### NO OVERSAMPLING - ACCURACY
##Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'none', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'none','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','accuracy')

### NO OVERSAMPLING - BALANCED ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'none', 't5', 'balanced_accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','RF', 'max', 'balanced_accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','balanced_accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'none','t5','max','balanced_accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','balanced_accuracy')

#### RANDOM OVERSAMPLING - ACCURACY
# # Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'random', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'random','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','random','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'random','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','random','t5','max','accuracy')

#### SMOTE OVERSAMPLING - ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data, 'smote', 't5', 'accuracy','max')
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'smote','t5','RF', 'max', 'accuracy')
# calc_accuracy(train_preds, train_labels,'training','RF','smote','t5','max','accuracy')
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'RF', 'smote','t5','max','accuracy')
# calc_accuracy(test_preds, test_labels,'test','RF','smote','t5','max','accuracy')

###======= BINARY =============================###

#### NO OVERSAMPLING - ACCURACY
##Hyp optimization and model selection
#best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data_binary, 'none', 't5', 'accuracy','max', "binary")

#Import model - optional
#best_params = joblib.load("results/04_models/binary/accuracy/RF_none_t5_max.joblib") #this will be removed

# Feature importance and training set predictions
#train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'none','t5','RF', 'max', 'accuracy', "binary")
#calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','accuracy',"binary")
# Predicting the test set
#Import model - optional
#best_model = joblib.load("results/04_models/binary/accuracy/RF_none_t5_max.joblib") #this will be removed
#test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'RF', 'none','t5','max','accuracy',"binary")
#calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','accuracy',"binary")

### NO OVERSAMPLING - BALANCED ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data_binary, 'none', 't5', 'balanced_accuracy','max',"binary")
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'none','t5','RF', 'max', 'balanced_accuracy',"binary")
# calc_accuracy(train_preds, train_labels,'training','RF','none','t5','max','balanced_accuracy',"binary")
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'RF', 'none','t5','max','balanced_accuracy',"binary")
# calc_accuracy(test_preds, test_labels,'test','RF','none','t5','max','balanced_accuracy',"binary")

#### RANDOM OVERSAMPLING - ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data_binary, 'random', 't5', 'accuracy','max',"binary")
# # Feature importance and training set predictions
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'random','t5','RF', 'max', 'accuracy',"binary")
# calc_accuracy(train_preds, train_labels,'training','RF','random','t5','max','accuracy',"binary")
# # Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'RF', 'random','t5','max','accuracy',"binary")
# calc_accuracy(test_preds, test_labels,'test','RF','random','t5','max','accuracy',"binary")
#
#### SMOTE OVERSAMPLING - ACCURACY
# Hyp optimization and model selection
#best_params,best_cv_results, best_model = tune_group_oversampling('RF', rf_train_data_binary, 'smote', 't5', 'accuracy','max',"binary")
# Feature importance and training set predictions
#train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'smote','t5','RF', 'max', 'accuracy',"binary")
#calc_accuracy(train_preds, train_labels,'training','RF','smote','t5','max','accuracy',"binary")
# Predicting the test set
#test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'RF', 'smote','t5','max','accuracy',"binary")
#calc_accuracy(test_preds, test_labels,'test','RF','smote','t5','max','accuracy',"binary")




#-Import model - optional
#xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed


#################################################

###         XGBC MODEL                          ##

###################################################


#### NO OVERSAMPLING - ACCURACY
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


#### NO OVERSAMPLING - BALANCED ACCURACY
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

#===== MULTILABEL =======

#### NO OVERSAMPLING - ACCURACY
# # Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'none', 't5', 'accuracy','max_RF')
# ## Feature importance and training set predictions
# #Import model - optional
#best_params = joblib.load("results/04_models/multilabel/balanced_accuracy/XGBC_none_t5_max_RF.joblib") #this will be removed
#train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','XGBC', 'max_RF', 'accuracy', 'multilabel')
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','accuracy')
# #Predicting the test set
# #Import model - optional
#best_model = joblib.load("results/04_models/multilabel/balanced_accuracy/XGBC_none_t5_max_RF.joblib") #this will be removed
#test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'none','t5','max_RF','accuracy',"multilabel")
#calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','accuracy')

### NO OVERSAMPLING - BALANCED ACCURACY
# Hyp optimization and model selection
#best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'none', 't5', 'balanced_accuracy','max_RF', 'multilabel')
# ## Feature importance and training set predictions
##=================
#IMPORT MODEL - THIS WILL BE REMOVED
#best_params = joblib.load("results/04_models/multilabel/balanced_accuracy/XGBC_none_t5_max_RF_v2.joblib") #this will be removed
#===========
#train_preds, train_labels = cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'none','t5','XGBC', 'max_RF', 'balanced_accuracy', 'multilabel')
#calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','balanced_accuracy', 'multilabel')
#=====Import model - this will be removed ====#
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
# #=====                                     ====#
# #Predicting the test set
# # import model this will be removed
# best_model = joblib.load("results/04_models/multilabel/balanced_accuracy/XGBC_none_t5_max_RF_v2.joblib") #this will be removed
# ##===
# test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'none','t5','max_RF','balanced_accuracy',"multilabel")
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','balanced_accuracy',"multilabel"       )
# #Create fasta file with important features
# #create_fasta(feature_importances,'XGBC','none','t5','max_RF','balanced_accuracy')

#### RANDOM OVERSAMPLING - ACCURACY
# Hyp optimization and model selection
#best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'random', 't5', 'accuracy','max_RF',"multilabel")
# ## Feature importance and training set predictions
#train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'random','t5','XGBC', 'max_RF', 'accuracy', 'multilabel')
#calc_accuracy(train_preds, train_labels,'training','XGBC','random','t5','max_RF','accuracy', 'multilabel')
# #Predicting the test set
# #Import model - optional
# #xgbc_best_model_none_t5_max = joblib.load("results/04_models/XGBC_none_t5_balanced_accuracy_max.joblib") #this will be removed
#test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'random','t5','max_RF','accuracy', "multilabel")
#calc_accuracy(test_preds, test_labels,'test','XGBC','random','t5','max_RF','accuracy', "multilabel")

#### SMOTE - ACCURACY
#Hyp optimization and model selection
best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data, 'smote', 't5', 'accuracy','max_RF',"multilabel")
## Feature importance and training set predictions
train_preds, train_labels= cross_model_balanced_blocked(rf_train_data, best_params, rf_label_df,'smote','t5','XGBC', 'max_RF', 'accuracy', 'multilabel')
calc_accuracy(train_preds, train_labels,'training','XGBC','smote','t5','max_RF','accuracy', 'multilabel')
#Predicting the test set
test_preds, test_labels= predict_test_set(best_model, rf_test_data, 'XGBC', 'smote','t5','max_RF','accuracy', "multilabel")
calc_accuracy(test_preds, test_labels,'test','XGBC','smote','t5','max_RF','accuracy', "multilabel")


#=====BINARY LABELS==========
###NO OVERSAMPLING - ACCURACY
# Hyp optimization and model selection
#best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data_binary, 'none', 't5', 'accuracy','max_RF', 'binary')
## Feature importance
#IMPORT MODEL - THIS WILL BE REMOVED
#best_params = joblib.load("results/04_models/binary/accuracy/XGBC_none_t5_max_RF.joblib") #this will be removed
##===========
#train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'none','t5','XGBC', 'max_RF', 'accuracy', 'binary')
#Training set accuracy
#calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','accuracy','binary')
#Predicting the test set
#Import model - optional
#best_model = joblib.load("results/04_models/binary/accuracy/XGBC_none_t5_max_RF.joblib") #this will be removed
#test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'XGBC', 'none','t5','max_RF','accuracy', 'binary')
#calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','accuracy','binary')

###NO OVERSAMPLING - BALANCED ACCURACY
# Hyp optimization and model selection
# best_params,best_cv_results, best_model = tune_group_oversampling('XGBC', rf_train_data_binary, 'none', 't5', 'balanced_accuracy','max_RF', 'binary')
# ## Feature importance
# train_preds, train_labels= cross_model_balanced_blocked(rf_train_data_binary, best_params, rf_label_df_binary,'none','t5','XGBC', 'max_RF', 'balanced_accuracy', 'binary')
# #Training set accuracy
# calc_accuracy(train_preds, train_labels,'training','XGBC','none','t5','max_RF','balanced_accuracy','binary')
# #Predicting the test set
# test_preds, test_labels= predict_test_set(best_model, rf_test_data_binary, 'XGBC', 'none','t5','max_RF','balanced_accuracy', 'binary')
# calc_accuracy(test_preds, test_labels,'test','XGBC','none','t5','max_RF','balanced_accuracy','binary')
#
#







