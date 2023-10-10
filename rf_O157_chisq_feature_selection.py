 #imports
import sys
#from os import system
from datetime import datetime
import numpy as np
#from math import sin, cos, sqrt, atan2, radians
#from pyproj import Proj, transform
import pandas as pd
#from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from numpy import array 


########## FUNCTIONS ###############


def usage():

    sys.exit()

def get_opts():
    if len(sys.argv) != 4:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3]



def read_meta(meta_file):
    all_pd = pd.read_csv(meta_file, sep='\t', header=0)
    all_pd = all_pd[['SRA', 'SYMP']]

    #sym_dict  = all_pd.groupby("SYMP")["SRA"].apply(list).to_dict()

    all_pd = all_pd.set_index('SRA')
    return all_pd

def read_features(feature_file1, feature_file2, meta_pd):
    
    print ("load dataframe 1")

    all_pd = pd.read_csv(feature_file1, sep='\t', header=0, index_col=0)
    all_pd = all_pd.loc[:, ~all_pd.columns.str.contains('^Unnamed')]
    #set >0 values to 1
    all_pd[all_pd > 0] = 1
    all_pd = all_pd.T

    print ("get long columns")
    to_keep = []
    for i in all_pd.columns:
        if len(i) >= 80:
            to_keep.append(i)   
    # change dataframe to include those kmers only 
    all_pd = all_pd[to_keep]



    all_pd.info(memory_usage = "deep")
    icols = all_pd.select_dtypes('integer').columns
    all_pd = all_pd.astype("int8")
    all_pd.info(memory_usage = "deep")

    print ("load dataframe 2")

    all_pd2 = pd.read_csv(feature_file2, sep='\t', header=0, index_col=0)
    all_pd2 = all_pd2.loc[:, ~all_pd2.columns.str.contains('^Unnamed')]
    #set >0 values to 1
    all_pd2[all_pd2 > 0] = 1
    all_pd2 = all_pd2.T

    print ("get long columns")

    to_keep2 = []
    for i in all_pd2.columns:
        if len(i) >= 80:
            to_keep2.append(i)   
    # change dataframe to include those kmers only 
    all_pd2 = all_pd2[to_keep2]

    all_pd2.info(memory_usage = "deep")
    icols = all_pd2.select_dtypes('integer').columns
    all_pd2 = all_pd2.astype("int8")
    all_pd2.info(memory_usage = "deep")

    all_pd = all_pd.T
    all_pd2 = all_pd2.T

    
    print ("create empty dfs")

    pd1_index = list(all_pd.index.values)
    pd2_index = list(all_pd2.index.values)
    to_add1 = set(pd1_index) - set(pd2_index)
    to_add2 = set(pd2_index) - set(pd1_index)
    df_temp = pd.DataFrame(columns=all_pd.columns.to_list(), index=list(to_add2))
    df_temp = df_temp.fillna(0).astype("int8")
    df_temp2 = pd.DataFrame(columns=all_pd2.columns.to_list(), index=list(to_add1))
    df_temp2 = df_temp2.fillna(0).astype("int8")  


    print ("add dfs to Original dfs")

    pd_merge1=pd.concat([all_pd, df_temp], axis=0)
    pd_merge2=pd.concat([all_pd2, df_temp2], axis=0)


    print ("Merge new dataframe")

    # pandas join two DataFrames
    pd3=pd_merge1.join(pd_merge2, how='inner')
    pd3.info(memory_usage = "deep")   

    pd3 = pd3.T

    print ("merge meta")

    merge_pd =pd.merge(pd3,meta_pd, left_index=True, right_index=True)
    merge_pd.info(memory_usage = "deep")


    label_encoder = LabelEncoder()
    merge_pd['SYMP'] = label_encoder.fit_transform(merge_pd['SYMP'])
 
    #shuffle columns
    #merge_pd = merge_pd[np.random.default_rng(seed=42).permutation(merge_pd.columns.values)]


    print ("do chi2")


    res = []


    X = merge_pd.drop('SYMP',axis=1)
    y = merge_pd['SYMP']

    chi_scores = chi2(X,y)
        
    #print (chi_scores)

    # Two features with highest chi-squared statistics are selected
    chi2_features = SelectKBest(chi2, k = 100000)
    X_kbest_features = chi2_features.fit_transform(X, y)
      
    # Reduced features
    print('Original feature number:', X.shape[1])
    print('Reduced feature number:', X_kbest_features.shape[1])


    filter = chi2_features.get_support()
    features = array(X.columns)

    #print (features[filter])

    good_df_100k = merge_pd[features[filter]]
    good_df_100k.to_csv(r'all_chisq_100k.tsv', sep='\t')

    p_values = pd.Series(chi_scores[1],index = X.columns)
    p_values.sort_values(ascending = True , inplace = True)

    good_df_pvalue = merge_pd[p_values[p_values<=0.05].index.to_list()]
    good_df_pvalue.to_csv(r'all_chisq_pvalue.tsv', sep='\t')


    #print(p_values<=0.05)


    #with open("chi_features.txt", "w") as txt_file:
    #    for line in features[filter]:
    #        txt_file.write(line + "\n")

############ MAIN #############



meta_file, feature_file1, feature_file2  = get_opts()

all_pd = read_meta(meta_file)

read_features(feature_file1, feature_file2, all_pd)


