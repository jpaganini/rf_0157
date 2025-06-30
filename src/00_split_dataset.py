# IMPORT PACKAGE

# Stats
import numpy as np
import random
from scipy import interp
import sys

# pandas
import pandas as pd



# modelling
from sklearn.model_selection import StratifiedGroupKFold


def usage():

    sys.exit()

def get_opts():
    if len(sys.argv) != 2:
        usage()
    else:
        return sys.argv[1]



def split_dataset(meta_file):
        RSEED=50

        #load the metadata
        metadata = pd.read_csv(meta_file, sep='\t', header=0)
        metadata = metadata.set_index('SRA')
        #metadata=metadata.iloc[:,:16] - #REMOVED WHEN MUVR

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

        #write the final output
        final_train.to_csv(r'TMP_2023_train_set_jp.tsv', sep='\t')
        final_test.to_csv(r'TMP_2023_test_set_jp.tsv', sep='\t')

        return final_train, final_test


################ MAIN ##############

#1. Load metadata
meta_file = get_opts()

#2. create a train-test split - considering population structre and blocking hihgly similar isoaltes to avoid data-leakage
train_data,validation_data=split_dataset(meta_file)