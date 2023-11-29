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
import seaborn as sns
from IPython.display import Image
from subprocess import call
from sklearn.tree import export_graphviz
import pydot
from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import Lasso

# modelling 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# model evaluation
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import roc_curve, auc,  RocCurveDisplay
from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

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
from py_muvr.feature_selector import FeatureSelector
from concurrent.futures import ProcessPoolExecutor

import shap


def usage():

    sys.exit()

def get_opts():
    if len(sys.argv) != 5:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def load_feat_ann(feature_file):
    label_kmer_df = pd.read_csv(feature_file, sep='\t')
    return label_kmer_df

def load_features(feature_file, meta_file, flag):
    
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
        all_pd = all_pd[['SRA', 'SYMP','LINEAGE', 'SNP ADDRESS']]
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


def feature_reduction(model_input):

    to_predict = ["SYMP"]
    #executor = ProcessPoolExecutor(max_workers=6)

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

    df_muvr_min.to_csv(r'df_muvr_min_2017d_merged.tsv', sep='\t')
    df_muvr_mid.to_csv(r'df_muvr_mid_2017d_merged.tsv', sep='\t')
    df_muvr_max.to_csv(r'df_muvr_max_2017d_merged.tsv', sep='\t')

    return df_muvr_max


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

def tune_rf_oversampling(model_input):
    RSEED = 50

    train_labels = np.array(model_input['SYMP'])
    train= model_input.drop(['SYMP','LINEAGE'], axis = 1)

    #Oversampling strategy
    oversampler = RandomOverSampler(random_state=RSEED)

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

    tunning_pipeline=Pipeline([
        ('oversampler', oversampler),
        ('model',model)
    ])

    #create random grid
    random_grid = {
     'model__n_estimators': n_estimators,
     'model__max_features': max_features,
     'model__max_depth': max_depth
     }

    # Fitting the pipeline and obtain the best parameters using random-search
    #GridSerac
    model_tunning_oversampling = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=RSEED, n_jobs = -1)
    model_tunning_oversampling.fit(train, train_labels)

    # print results of the best parameters
    best_param_oversampling = model_tunning_oversampling.best_params_
    print("Best params oversampling: ",best_param_oversampling)

def tune_rf_group(model_input):
    # create a hyperparameter grid
    RSEED = 50

    train_labels = np.array(model_input['SYMP'])
    train= model_input.drop(['SYMP','LINEAGE'], axis = 1)
    lineages=np.array(model_input['LINEAGE'])

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

    #create iterator list according to lineages
    logo = LeaveOneGroupOut()
    cv_iterator = list(logo.split(train, train_labels, groups=lineages))

    #for train, test in logo.split(train, train_labels, groups=lineages):
    #    print("%s %s" % (train, test))

    # generate model
    estimator = RandomForestClassifier()

    # Random search of parameters
    model_tunning = RandomizedSearchCV(estimator, param_distributions = random_grid, n_iter = 100, cv = cv_iterator, verbose=2, random_state=RSEED, n_jobs = -1)

    # Fit the model
    model_tunning.fit(train, train_labels)

    # print results
    best_param = model_tunning.best_params_
    print('Best parameters, considering sub-lineages', best_param)

def tune_rf_group_oversampling(model_input):
    RSEED = 50

    train_labels = np.array(model_input['SYMP'])
    train= model_input.drop(['SYMP','LINEAGE'], axis = 1)
    lineages = np.array(model_input['LINEAGE'])

    #Oversampling strategy
    oversampler = RandomOverSampler(random_state=RSEED)

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

    tunning_pipeline=Pipeline([
        ('oversampler', oversampler),
        ('model',model)
    ])

    #create random grid
    random_grid = {
     'model__n_estimators': n_estimators,
     'model__max_features': max_features,
     'model__max_depth': max_depth
     }

    #create iterator list according to lineages
    logo = LeaveOneGroupOut()
    cv_iterator = list(logo.split(train, train_labels, groups=lineages))

    # Fitting the pipeline and obtain the best parameters using random-search
    #GridSerac
    model_tunning_oversampling = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions = random_grid, n_iter = 100, cv = cv_iterator, verbose=2, random_state=RSEED, n_jobs = -1)
    model_tunning_oversampling.fit(train, train_labels)

    # print results of the best parameters
    best_param_oversampling = model_tunning_oversampling.best_params_
    print("Best params oversampling, sublineages: ",best_param_oversampling)

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
                                   max_features = 'sqrt',
                                   n_jobs=-1, verbose = 1, max_depth=460)
        
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

    


    RocCurveDisplay.from_predictions(y_true = labels, y_pred = preds)
    plt.show()



################ MAIN ##############


feature_file, ann_file, meta_file, val_file = get_opts()

print ("load training data")
feature_df = load_features(feature_file, meta_file,"T")

print ("load validation data")
val_df = load_features(val_file, meta_file,"V")

print ("load feature annotation")
label_df = load_feat_ann(ann_file)

#print ("feature reduction")
#feature_df = feature_reduction(feature_df)

#print ("simple rf")
#simple_rf(feature_df)

#Check hyper-parameters optimization
#print ("hyperparam optimisation")
#tune_rf(feature_df)

#print ("hyperparam optimisation")
#tune_rf_oversampling(feature_df)

#print ("hyperparam optimisation")
#tune_rf_group(feature_df)

#print ("hyperparam optimisation")
#tune_rf_group_oversampling(feature_df)

#print ("hyperparam optimisation considering sub-lineages")
#tune_rf_group(feature_df)


#print ("cv rf")
final_res, final_imp, preds, labels = cross_model_balanced(feature_df, label_df)


#calc_accuracy(preds, labels)

#imp_fasta = create_fasta(final_imp)

#final_model(feature_df, val_df)




