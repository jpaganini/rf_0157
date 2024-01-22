# IMPORT PACKAGE 

# Stats
import numpy as np
import random 
#from collections import Counter
#import itertools
#from scipy import interp
#from itertools import cycle
import sys
#import math

# pandas
import pandas as pd

#dask dataframes for large files manipulation
import dask.dataframe as dd

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
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import LeaveOneGroupOut
#from sklearn.model_selection import StratifiedGroupKFold
#from sklearn.model_selection import GroupKFold
#from sklearn.preprocessing import label_binarize
#from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder


# model evaluation
#from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score
#from sklearn.metrics import roc_curve, auc,  RocCurveDisplay
#from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report

# cross validation 
#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import StratifiedKFold
#from yellowbrick.model_selection import CVScores
#from yellowbrick.model_selection import RFECV
#from sklearn.model_selection import cross_val_score

#from imblearn.over_sampling import RandomOverSampler
#from imblearn.pipeline import Pipeline
#from imblearn.over_sampling import SMOTE

###### ------- MUVR ------- ######
from py_muvr.feature_selector import FeatureSelector
#from concurrent.futures import ProcessPoolExecutor

#import shap


def usage():

    sys.exit()

def get_opts():
    if len(sys.argv) != 6:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def get_opts_muvr():
    if len(sys.argv) != 3:
        usage()
    else:
        return sys.argv[1], sys.argv[2]

def get_opts_extract():
    if len(sys.argv) != 5:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4]
def load_feat_ann(feature_file):
    label_kmer_df = pd.read_csv(feature_file, sep='\t')
    return label_kmer_df

def load_features(feature_file):
    
    print ("load feature file")
    label_kmer_df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0)

    try:
        label_kmer_df = label_kmer_df.set_index('SRA')
    except:
        print ("index not found")
    try:
        label_kmer_df= label_kmer_df.drop('SYMP', axis = 1)
    except:
        print ("label not found")

    print ("load meta file")



    if flag == 'T':
        all_pd = pd.read_csv(meta_file, sep='\t', header=0)
        all_pd = all_pd[['SRA', 'SYMP','LINEAGE','t5']]
        all_pd = all_pd.set_index('SRA')
        all_pd = pd.concat([all_pd, all_pd['SNP ADDRESS'].str.split('.', expand=True)], axis=1)
        all_pd = all_pd.sort_values('SRA').drop_duplicates(subset=[5, 'SYMP'], keep='last') #remove samples that contain a +
        all_pd = all_pd[['SYMP','LINEAGE']]

    else:
        all_pd = pd.read_csv(meta_file, sep='\t', header=0)
        all_pd = all_pd[['SRA', 'SYMP','LINEAGE']]
        all_pd = all_pd.set_index('SRA')

    print ("merge features and meta data")

    merge_pd =pd.merge(label_kmer_df,all_pd, left_index=True, right_index=True)
    symptom_counts_train = merge_pd['SYMP'].value_counts()
    print('Symptomps count training', symptom_counts_train)


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

def feature_reduction(train_data_muvr,chisq_file):

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

    feature_names = model_input.drop(columns=["SYMP"]).columns


    feature_selector = FeatureSelector(
        n_repetitions=10,
        n_outer=5,
        n_inner=4,
        estimator="RFC",
        metric="MISS",
        features_dropout_rate=0.9
    )

    feature_selector.fit(X_muvr, y_muvr)
    selected_features = feature_selector.get_selected_features(feature_names=feature_names)

    # Obtain a dataframe containing MUVR selected features
    df_muvr_min = model_input[to_predict+list(selected_features.min)]
    df_muvr_mid = model_input[to_predict+list(selected_features.mid)]
    df_muvr_max = model_input[to_predict+list(selected_features.max)]

    print('something')

    df_muvr_min.to_csv(r'2023_jp_muvr_min.tsv', sep='\t')
    df_muvr_mid.to_csv(r'2023_jp_muvr_mid.tsv', sep='\t')
    df_muvr_max.to_csv(r'2023_jp_muvr_max.tsv', sep='\t')

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


def simple_rf(label_kmer_df):

    labels = np.array(label_kmer_df['SYMP'])
    features= label_kmer_df.drop(['SYMP','LINEAGE'], axis = 1)


    RSEED = 50

    #train, test split the data
    train_features, test_features, train_labels, test_labels = train_test_split(features, 
                                                                                labels, 
                                                                                test_size = 0.3, 
                                                                                random_state = RSEED)
    
    #over sample
    train_features, train_labels = RandomOverSampler().fit_resample(train_features, train_labels) #random state?


    #build the model 
    model = RandomForestClassifier(n_estimators=822, 
                                   random_state=RSEED, 
                                   max_features = 'sqrt',
                                   n_jobs=-1, verbose = 1)
    # fit the trainning data to the model 
    model_fit = model.fit(train_features, train_labels)

    #test the model on test data
    test_model = model.predict(test_features)

    test_rf_predictions = model.predict(test_features)
    test_rf_probs = model.predict_proba(test_features)


    # 5. get classification report
    report_ = classification_report(
        digits=6,
        y_true= test_labels, 
        y_pred= test_model)

    print('Classification Report Simple RF: ' ,
          report_)


    
    # 1. Confusion Matrix
    visualizer = ConfusionMatrix(model, classes = ['BD', 'D', 'HUS'], size = (1000,800))

    visualizer.fit(train_features, train_labels)  
    visualizer.score(test_features, test_labels) 
    visualizer.finalize()

    visualizer.ax.set_xlabel('Predicted Class', fontsize=20, fontweight = 'bold')
    visualizer.ax.set_ylabel('True Class', fontsize=20, fontweight = 'bold')
    visualizer.ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    for label in visualizer.ax.texts:
        label.set_size(25)

    visualizer.show(outpath='confusion_matrix_model_1') 



def tune_rf(model_input):
    # create a hyperparameter grid 
    RSEED = 50

    train_labels = np.array(model_input['SYMP'])
    train= model_input.drop(['SYMP','LINEAGE'], axis = 1)

    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
    print (n_estimators)

    # number of features at every split
    max_features = ['log2', 'sqrt']

    # max depth
    max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
    print (max_depth)
    max_depth.append(None)

    # create random grid
    random_grid = {
     'n_estimators': n_estimators,
     'max_features': max_features,
     'max_depth': max_depth
     }

    # generate model 
    estimator = RandomForestClassifier()

    # Random search of parameters
    #This hyper parameter tunning uses Stratified K-fold CV. In this method, each K-Fold retains the proportion of each class (D,BD, HUS) in the entire dataset.
    model_tunning = RandomizedSearchCV(estimator, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=RSEED, n_jobs = -1)

    # Fit the model
    model_tunning.fit(train, train_labels)

    # print results
    best_param = model_tunning.best_params_
    print('Best params no oversampling', best_param)


def tune_rf_group_oversampling(model_input, sampling, block_strategy):
    RSEED = 50

    train_labels = np.array(model_input['SYMP'])
    train= model_input.iloc[:, :-16]

    #Model
    model = RandomForestClassifier()

    #SET UP GRID VALUES FOR HYPER-PARAMETER TUNNING
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
    print(n_estimators)

    # number of features at every split
    max_features = ['log2', 'sqrt']

    # max depth
    max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
    # print (max_depth)
    # max_depth.append(None)

    #Create a imblearn Pipeline to tune hyper-parameters with oversampling included



    # Oversampling strategy, random grid and Pipeline
    if sampling == 'random':
        oversampler = RandomOverSampler(random_state=RSEED)
        # create random grid
        random_grid = {
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth
        }

        tunning_pipeline = Pipeline([
            ('oversampler', oversampler),
            ('model', model)
        ])

    if sampling == 'smote':
        oversampler = SMOTE(random_state=RSEED)
        # create random grid
        random_grid = {
            'oversampler__k_neighbors': k_neighbors,
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth
        }

        tunning_pipeline = Pipeline([
            ('oversampler', oversampler),
            ('model', model)
        ])

    if sampling == 'none':
        oversampler = SMOTE(random_state=RSEED)
        # create random grid
        random_grid = {
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth
        }

        tunning_pipeline = Pipeline([
            ('model', model)
        ])

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
    #GridSearch
    model_tunning_oversampling = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions = random_grid, n_iter = 100, cv = cv_iterator, verbose=2, random_state=RSEED, n_jobs = -1)
    model_tunning_oversampling.fit(train, train_labels)

    # print results of the best parameters
    best_param_oversampling= model_tunning_oversampling.best_params_
    print("Best params:",best_param_oversampling)
    return best_param_oversampling

def final_model(model_input, val_input):
    RSEED = 50

    test_features = val_input.drop('SYMP', axis = 1)
    test_labels = val_input['SYMP'].values.ravel()

    #print (test_features)


    train_labels = model_input['SYMP'].values.ravel()
    train_features= model_input.drop('SYMP', axis = 1)

    #print (train_features)


    train_features, train_labels = RandomOverSampler().fit_resample(train_features, train_labels)


    #print (train_features, train_labels)

    model = RandomForestClassifier(n_estimators=200, 
                                   random_state=RSEED, 
                                   max_features = 'sqrt',
                                   n_jobs=-1, verbose = 1, max_depth=460)

    model.fit(train_features, train_labels)


    #test the model on test data
    test_model = model.predict(test_features)

    test_rf_predictions = model.predict(test_features)
    test_rf_probs = model.predict_proba(test_features)


    # 5. get classification report
    report_ = classification_report(
        digits=6,
        y_true= test_labels, 
        y_pred= test_model)

    print('Classification Report Validation Data: ' ,
          report_)


    
    # 1. Confusion Matrix
    #visualizer = ConfusionMatrix(model, classes = ['BD', 'D', 'HUS'], size = (1000,800))

    #visualizer.fit(train_features, train_labels)  
    #visualizer.score(test_features, test_labels) 
    #visualizer.finalize()

    #visualizer.ax.set_xlabel('Predicted Class', fontsize=20, fontweight = 'bold')
    #visualizer.ax.set_ylabel('True Class', fontsize=20, fontweight = 'bold')
    #visualizer.ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    #for label in visualizer.ax.texts:
    #    label.set_size(25)

    #visualizer.show(outpath='confusion_matrix_model_1') 

def cross_model_balanced(model_input, label_df):

    #columns = model_input.columns.tolist()
    #ßprint (columns)
    #columns = columns.remove('SYMP')

    # Create 10-folds for cross validation
    row_nums = list(range(len(model_input)))
    random.shuffle(row_nums)
    splits = np.array_split(row_nums, 10)
    splits_indeces = []
    for array in splits:
        splits_indeces += [model_input.index[array]]

    acc_muvr_max = []
    RSEED = 50

    #data frame for storing probabilities of classification
    final_res = pd.DataFrame()
    #data frame for storing feature importance
    final_imp = pd.DataFrame()
    list_shap_values = list()
    list_test_sets = pd.DataFrame()

    #Test-set true lables
    list_test_labels = []
    #Test-set predictions
    list_test_pred = []
    
    for x in range(10):
        
        test = model_input.iloc[splits[x]]
        samples = test.index.values.tolist()
        test_features = test.drop(['SYMP','LINEAGE'], axis = 1)
        test_labels = test['SYMP'].values.ravel()
        train = model_input.drop(splits_indeces[x],axis=0)
        train_features = train.drop(['SYMP','LINEAGE'], axis = 1)
        train_labels = train['SYMP'].values.ravel()
        features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features, train_labels)



        model = RandomForestClassifier(n_estimators=200, 
                                   random_state=RSEED, 
                                   max_features = 'log2',
                                   n_jobs=-1, verbose = 1, max_depth=260)
        
        model.fit(features_resampled, labels_resampled)
        #test the model on test data
        
        test_model = model.predict(test_features)
        test_rf_probs = model.predict_proba(test_features)
        
     
        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)


        # 4. extract feature importance 
        feature_model = pd.DataFrame({'feature': test_features.columns,
                   x: model.feature_importances_}).\
                    sort_values(x, ascending = False)
        feature_model = feature_model.set_index('feature')

        final_imp = pd.concat([final_imp,feature_model], axis=1)

        res_df = pd.DataFrame({'samples': samples,'labels': test_labels,'predictions': test_model, 'probabilties' : zip(test_rf_probs) })


        final_res = pd.concat([final_res, res_df])

        # 5. get classification report
        report_ = classification_report(
            digits=6,
            y_true= test_labels, 
            y_pred= test_model)
        



        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)


        list_shap_values.append(shap_values)
        list_test_sets = pd.concat([list_test_sets, test_features])



        acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max)/len(acc_muvr_max))
    
    #outputs commented out
    final_res.to_csv(r'final_pred_standard_fixed.tsv', sep='\t')
    final_imp['average'] = final_imp.mean(numeric_only=True, axis=1)
    final_imp = final_imp.sort_values('average', ascending=False)
    final_imp.to_csv(r'final_standard_fixed.tsv', sep='\t')
    

#   combining results from all iterations
    shap_values = np.array(list_shap_values[0])
    for i in range(1,len(list_shap_values)):
        shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
    
    #print shap uncomment
    label_dict = dict(zip(label_df['Sequence'], label_df['ANN']))
    fest = list_test_sets.columns.values.tolist()

    anns = []
    for f in fest:
        anns.append(label_dict[f])


    print (model.classes_)

    #shap.summary_plot(shap_values[0], list_test_sets, feature_names=anns)
    #shap.summary_plot(shap_values[1], list_test_sets,feature_names=anns)
    shap.summary_plot(shap_values[2], list_test_sets,feature_names=anns)

    #shap.summary_plot(shap_values[0], list_test_sets)
    #shap.summary_plot(shap_values[1], list_test_sets)
    #shap.summary_plot(shap_values[2], list_test_sets)

    return final_res, final_imp, list_test_pred, list_test_labels

def cross_model_balanced_blocked(model_input, label_df,sampling,block_strategy):

    #1. Import the data
    all_labels = np.array(model_input['SYMP'])
    features= model_input.iloc[:,:-16]


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
        test_features = test.iloc[:,:-16]
        test_labels = test['SYMP'].values.ravel()

        train = model_input.iloc[cv_iterator[x][0]]
        train_features = train.iloc[:,:-16]
        train_labels = train['SYMP'].values.ravel()

        if sampling=='random':
            features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
            model = RandomForestClassifier(n_estimators=733,
                                           random_state=RSEED,
                                           max_features='sqrt',
                                           n_jobs=-1, verbose=1, max_depth=300)
        if sampling=='smote':
            features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=2).fit_resample(train_features,train_labels)
            model = RandomForestClassifier(n_estimators=644,
                                           random_state=RSEED,
                                           max_features='sqrt',
                                           n_jobs=-1, verbose=1, max_depth=460)

        model.fit(features_resampled, labels_resampled)
        # test the model on test data

        test_model = model.predict(test_features)
        test_rf_probs = model.predict_proba(test_features)

        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)

        # 4. extract feature importance
        feature_model = pd.DataFrame({'feature': test_features.columns,
                                      x: model.feature_importances_}). \
            sort_values(x, ascending=False)
        feature_model = feature_model.set_index('feature')

        final_imp = pd.concat([final_imp, feature_model], axis=1)

        res_df = pd.DataFrame(
            {'samples': samples, 'labels': test_labels, 'predictions': test_model, 'probabilties': zip(test_rf_probs)})

        final_res = pd.concat([final_res, res_df])

        # 5. get classification report
        report_ = classification_report(
            digits=6,
            y_true=test_labels,
            y_pred=test_model)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)

        list_shap_values.append(shap_values)
        list_test_sets = pd.concat([list_test_sets, test_features])

        acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max) / len(acc_muvr_max))

    # outputs commented out
    #final_res.to_csv(r'final_pred_grouped_test.tsv', sep='\t')
    #final_imp['average'] = final_imp.mean(numeric_only=True, axis=1)
    #final_imp = final_imp.sort_values('average', ascending=False)
    #final_imp.to_csv(r'final_grouped_test.tsv', sep='\t')

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

    print(model.classes_)

    shap.summary_plot(shap_values[0], list_test_sets, feature_names=anns)
    shap.summary_plot(shap_values[1], list_test_sets,feature_names=anns)
    shap.summary_plot(shap_values[2], list_test_sets, feature_names=anns)

    # shap.summary_plot(shap_values[0], list_test_sets)
    # shap.summary_plot(shap_values[1], list_test_sets)
    # shap.summary_plot(shap_values[2], list_test_sets)

    return final_res, final_imp, list_test_pred, list_test_labels

def cross_model_balanced_sublineage(model_input, label_df):
    # columns = model_input.columns.tolist()
    # ßprint (columns)
    # columns = columns.remove('SYMP')

    # Create 10-folds for cross validation
    #row_nums = list(range(len(model_input)))
    #random.shuffle(row_nums)
    #splits = np.array_split(row_nums, 10)
    #splits_indeces = []
    #for array in splits:
    #    splits_indeces += [model_input.index[array]]

    #1. Import the data
    all_labels = np.array(model_input['SYMP'])
    features= model_input.drop(['SYMP','LINEAGE'], axis = 1)
    lineages=np.array(model_input['LINEAGE'])

    #set up the random seed
    RSEED = 50

    #Create an iterator to separate files in groups
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
        test_features = test.drop(['SYMP', 'LINEAGE'], axis=1)

        test_labels = test['SYMP'].values.ravel()
        train = model_input.iloc[cv_iterator[x][0]]
        train_features = train.drop(['SYMP', 'LINEAGE'], axis=1)
        train_labels = train['SYMP'].values.ravel()

        features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,
                                                                                                  train_labels)

        model = RandomForestClassifier(n_estimators=200,
                                       random_state=RSEED,
                                       max_features='log2',
                                       n_jobs=-1, verbose=1, max_depth=420)

        model.fit(features_resampled, labels_resampled)
        # test the model on test data

        test_model = model.predict(test_features)
        test_rf_probs = model.predict_proba(test_features)

        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)

        # 4. extract feature importance
        feature_model = pd.DataFrame({'feature': test_features.columns,
                                      x: model.feature_importances_}). \
            sort_values(x, ascending=False)
        feature_model = feature_model.set_index('feature')

        final_imp = pd.concat([final_imp, feature_model], axis=1)

        res_df = pd.DataFrame(
            {'samples': samples, 'labels': test_labels, 'predictions': test_model, 'probabilties': zip(test_rf_probs)})

        final_res = pd.concat([final_res, res_df])

        # 5. get classification report
        report_ = classification_report(
            digits=6,
            y_true=test_labels,
            y_pred=test_model)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_features)

        list_shap_values.append(shap_values)
        list_test_sets = pd.concat([list_test_sets, test_features])

        acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max) / len(acc_muvr_max))

    # outputs commented out
    #final_res.to_csv(r'final_pred_grouped_test.tsv', sep='\t')
    #final_imp['average'] = final_imp.mean(numeric_only=True, axis=1)
    #final_imp = final_imp.sort_values('average', ascending=False)
    #final_imp.to_csv(r'final_grouped_test.tsv', sep='\t')

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

    print(model.classes_)

    # shap.summary_plot(shap_values[0], list_test_sets, feature_names=anns)
    # shap.summary_plot(shap_values[1], list_test_sets,feature_names=anns)
    shap.summary_plot(shap_values[2], list_test_sets, feature_names=anns)

    # shap.summary_plot(shap_values[0], list_test_sets)
    # shap.summary_plot(shap_values[1], list_test_sets)
    # shap.summary_plot(shap_values[2], list_test_sets)

    return final_res, final_imp, list_test_pred, list_test_labels


def create_fasta(final_imp):
    features =  final_imp.index.values.tolist()
    #print (features)
    with open('features_2023_new.fasta', 'w') as f:
        for x, feat in enumerate(features):
            f.write('>Feature'+str(x+1))
            f.write('\n')
            f.write(feat)
            f.write('\n')

def calc_accuracy(preds, labels):
    report_ = classification_report(
            digits=6,
            y_true= labels, 
            y_pred= preds)
    print('Classification Report Training Data: ' ,
          report_)
    cm = confusion_matrix(labels, preds, labels=['BD','D','HUS'])
    disp = ConfusionMatrixDisplay.from_predictions(cmap=plt.cm.Blues, y_true = labels, y_pred = preds, display_labels=['Bloody Diarrhoea', 'Diarrhoea', 'HUS'], normalize='true', values_format='.1g')
    disp.plot()
    plt.show()

    #RocCurveDisplay.from_predictions(y_true = labels, y_pred = preds)
    #plt.show()



################ MAIN ##############

#get all opts - this will have to be modified
#feature_file, ann_file, meta_file, val_file, chisq_file = get_opts()

#create a train-test split - considering population structre and blocking hihgly similar isoaltes to avoid data-leakage
#train_data,validation_data=split_dataset(meta_file,feature_file)
#train_data,validation_data=split_dataset(meta_file)

#Create a sub-set of the train set, which will be used for doing muvr.
    #In this subset, only one isolate within the same t5 cluster are retained
#train_data_muvr=prepare_data_muvr(train_data)

#======MUVR step =====#
#train_data_muvr, chisq_file = get_opts_muvr()
#print ("MUVR feature reduction")
#feature_df = feature_reduction(train_data_muvr, chisq_file)
#===========

#-====FEATURE EXTRACTION STEP =====
###In this step, we will extract relevant features from all samples
min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file = get_opts_extract()
feature_df = feature_extraction(min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file)
#=========
#print ("load training data")
#feature_df = load_features(train_data)

#print ("load validation data")
#val_df = load_features(val_file, meta_file,"V")

#print ("load feature annotation")
#label_df = load_feat_ann(ann_file)

#print ("feature reduction")
#feature_df = feature_reduction(feature_df)

#print ("simple rf")
#simple_rf(feature_df)

#Check hyper-parameters optimization
#print ("hyperparam optimisation")
#tune_rf(feature_df)

#HYPER-PARAMETER OPTIMIZATION
#best_params_smote_t5 = tune_rf_group_oversampling(train_data, 'smote', 't5')
#best_params_random_t5 = tune_rf_group_oversampling(train_data, 'random', 't5')


#print ("cv rf")
#final_res, final_imp, preds, labels = cross_model_balanced(train_data, label_df)
#calc_accuracy(preds, labels)


#Using t5 as block -
#final_res_t5, final_imp_t5, preds_t5, labels_t5 = cross_model_balanced_blocked(train_data, label_df,'random','t5')
#calc_accuracy(preds_t5, labels_t5)

#Using sublineage as block
#final_res_sublineage, final_imp_sublineage, preds_sublineage, labels_sublineage = cross_model_balanced_blocked(train_data, label_df,'random','t5')
#calc_accuracy(preds_sublineage, labels_sublineage)


#imp_fasta = create_fasta(final_imp)

#final_model(feature_df, val_df)




