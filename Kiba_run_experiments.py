from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import random as rn
import pandas as pd
import time
import os
os.environ['PYTHONHASHSEED'] = '0'
#import tensorflow as tf
np.random.seed(1)
rn.seed(1)
import glob
from sklearn.pipeline import Pipeline
from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA

import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error, r2_score
from lifelines.utils import concordance_index
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr

import xgboost as xgb
from xgboost import XGBRegressor
import subprocess
print( subprocess.call(["ls", "-l"]))

TABSY = "\t"
figdir = "figures/"


def nfold_1_2_3_setting_sample(XD, XT,  Y, label_row_inds, label_col_inds, measure,  FLAGS, dataset):

    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    print(type(outer_train_sets)) # List >> two Dimensions.
    rows = len(outer_train_sets)
    cols = len(outer_train_sets[0]) if rows > 0 else 0
    print("Shape of outer_train_sets: ", (rows, cols))
    flat = []
    for row in outer_train_sets:
        for item in row:
            flat.append(item)
    # print(flat)
    print(type(flat)) # List >> 1 Dimension.
    rows = len(flat)
    print("Shape of flat: ", (rows))
    
    
    foldinds = len(outer_train_sets)

    test_sets = [test_set]
    ## TRAIN AND VAL
    val_sets = []
    train_sets = [flat]

    train_drugs, train_prots,  train_Y, val_drugs, val_prots,  val_Y = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                        measure, FLAGS, train_sets, test_sets)
    
    X_train = np.hstack((train_drugs, train_prots))
    X_test = np.hstack((val_drugs, val_prots))
    y_train = train_Y
    y_test = val_Y
    return X_train, X_test, y_train, y_test
    

def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
    paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384]
    epoch = FLAGS.num_epoch                                 #100
    batchsz = FLAGS.batch_size                             #256
    input_file = '/content/drive/MyDrive/Drug-Target-Binding-Affinity/UniProt/input_data.csv'
    # Read CSV file into a DataFrame
    input_data2 = pd.read_csv(input_file)
    
    # Check if 'KD_value' column exists
    if 'KD_value' in input_data2.columns:
        # Extract the column and convert to NumPy array
        KD_values = input_data2['KD_value']
        KD_values1 = KD_values.to_numpy()
    else:
        raise KeyError("Column 'KD_value' not found in the input file.")
    
    print("KD values have been read! ")
    logging("---Parameter Search-----", FLAGS)
    print(paramset2)
    print(len(paramset2))
    print(label_row_inds)
    print(type(label_row_inds))
    print(len(label_row_inds))
    print('-------------------')
    print(label_col_inds)
    print(type(label_col_inds))
    print(len(label_col_inds))
    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 
    print(all_predictions)
    print(len(val_sets))
    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]
        print(type(labeledinds))
        print(len(labeledinds))
        # KD_train = KD_values1[labeledinds]
        # KD_test = KD_values1[valinds]
        # print("KD_train", KD_train.shape)
        # print("KD_test", KD_test.shape)
        Y_train = np.asmatrix(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        print(type(trrows))
        print(type(trcols))
        XD_train = XD[trrows]
        XT_train = XT[trcols]
        print("trrows", str(trrows), str(len(trrows)))
        print("trcols", str(trcols), str(len(trcols)))
        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        print("train_drugs", train_drugs.shape[1])
        print("train_prots", train_prots.shape[1])
        print("train_Y", np.array(train_Y).shape)
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        print("terows", str(terows), str(len(terows)))
        print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)
        print("test_Y",np.array(val_Y).shape)
        train_Y1 = np.array(train_Y)
        val_Y1 = np.array(val_Y)        
        
        # Combine KD_train and train_Y into one array with two columns
        # combined_train = np.column_stack((KD_train, train_Y1))
        # combined_test = np.column_stack((KD_test, val_Y1))
        
        # Create DataFrames with column names
        #df_train = pd.DataFrame(combined_train, columns=["KD_train", "train_Y1"])
        #df_test = pd.DataFrame(combined_test, columns=["KD_test", "val_Y1"])

        # Write to CSV
        #df_train.to_csv("/content/drive/MyDrive/Drug-Target-Binding-Affinity/UniProt/combined_train.csv", index=False)
        #df_test.to_csv("/content/drive/MyDrive/Drug-Target-Binding-Affinity/UniProt/combined_test.csv", index=False)

        #results_train = np.allclose(KD_train, train_Y1)
        #results_test = np.allclose(KD_test, val_Y1)   
        #print(results_train)
        #print(results_test)
        #print(KD_train.shape, train_Y1.shape)
        #print(KD_test.shape, val_Y1.shape)
        #print(np.array_equal(KD_train, train_Y1))
        #print(np.array_equal(KD_test, val_Y1))

    return train_drugs, train_prots,  train_Y, val_drugs, val_prots,  val_Y


def plotLoss(history, batchind, epochind, param3ind, foldind):

    figname = "b"+str(batchind) + "_e" + str(epochind) + "_" + str(param3ind) + "_"  + str( foldind) + "_" + str(time.time()) 
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
	#plt.legend(['trainloss', 'valloss', 'cindex', 'valcindex'], loc='upper left')
    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+figname +".png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                    format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()


    ## PLOT CINDEX
    plt.figure()
    plt.title('model concordance index')
    plt.ylabel('cindex')
    plt.xlabel('epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['traincindex', 'valcindex'], loc='upper left')
    plt.savefig("figures/"+figname + "_acc.png" , dpi=None, facecolor='w', edgecolor='w', orientation='portrait', 
                            format=None,transparent=False, bbox_inches=None, pad_inches=0.1)
    plt.close()

def fast_remove_correlated_features(X, threshold=0.95):
    """
    Efficiently removes highly correlated features using a generator pattern.
    """
    X = X.copy()
    to_drop = set()
    columns = X.columns

    for i in range(len(columns)):
        if columns[i] in to_drop:
            continue
        for j in range(i + 1, len(columns)):
            if columns[j] in to_drop:
                continue
            corr = X[columns[i]].corr(X[columns[j]])
            if abs(corr) > threshold:
                to_drop.add(columns[j])
    return X.drop(columns=to_drop)
    
def remove_highly_correlated_features(df, threshold=0.95, max_columns=5000):
    """
    Removes highly correlated features from a large DataFrame by:
    1. Splitting the DataFrame by columns if it exceeds max_columns.
    2. Removing highly correlated features within each chunk.
    3. Merging the remaining features.
    4. Performing a final correlation removal across the merged features.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric features.
        threshold (float): Correlation threshold above which features are removed.
        max_columns (int): Max number of columns per chunk.
    
    Returns:
        pd.DataFrame: DataFrame with reduced features.
    """

    def drop_correlated(data, threshold):
        corr_matrix = data.corr().abs()
        upper = np.triu(corr_matrix, k=1)
        to_drop = [data.columns[i] for i in range(len(data.columns))
                   if any(upper[i, :] > threshold)]
        return data.drop(columns=to_drop)

    # Step 1: Split if needed
    n_cols = df.shape[1]
    if n_cols <= max_columns:
        cleaned_chunks = [drop_correlated(df, threshold)]
    else:
        chunks = [df.iloc[:, i:i+max_columns] for i in range(0, n_cols, max_columns)]
        cleaned_chunks = [drop_correlated(chunk, threshold) for chunk in chunks]

    # Step 2: Combine remaining columns
    combined = pd.concat(cleaned_chunks, axis=1)
    print(combined.shape)

    # Step 3: Final correlation filtering
    final_cleaned = drop_correlated(combined, threshold)
    print(final_cleaned.shape)
    
    return final_cleaned    
def preprocess_data(X):
    """
    Combined preprocessing including:
    1. Removing all-zero columns
    2. Removing zero-variance columns
    3. Removing highly correlated features (default threshold = 0.95)

    Parameters:
    -----------
    X : array-like or pd.DataFrame
        Input feature matrix (n_samples, n_features)

    correlation_threshold : float
        Threshold for removing highly correlated features

    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed feature matrix
    """

    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Replace NaNs and Infs with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 1. Remove all-zero columns
    X = X.loc[:, (X != 0).any(axis=0)]

    # 2. Remove zero-variance columns
    selector = VarianceThreshold()
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
    
    # X = remove_highly_correlated_features(X, threshold=0.90, max_columns=10000)

    print("Original shape: ", X.shape)
    # print("threshold: ", threshold)
    print("Filtered shape:", X.shape)
    return X


def select_top_features(XD, XT, Y, top_k=300):
    """Select top features from XD and XT matrices"""

    # Select top drug features (using mean Y per drug as target)
    xd_selector = SelectKBest(score_func=f_regression, k=min(top_k, XD.shape[1]))
    XD_top = xd_selector.fit_transform(XD, np.nanmean(Y, axis=1))

    # Select top target features (using mean Y per target as target)
    xt_selector = SelectKBest(score_func=f_regression, k=min(top_k, XT.shape[1]))
    XT_top = xt_selector.fit_transform(XT, np.nanmean(Y, axis=0))

    print(f"Original shapes - XD: {XD.shape}, XT: {XT.shape}")
    print(f"Selected top {top_k} features - XD: {XD_top.shape}, XT: {XT_top.shape}")

    return XD_top, XT_top

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[]
    print("prepare_interaction_pairs")
    print(rows.shape)
    print(cols.shape)
    print(Y.shape)
    print("drug shape = ", XD.shape)
    print("protein shape = ", XT.shape)
    
    # feature selection ... should be done here because the features are a lot.

    # Apply preprocessing
    print("starting preprocessing  drug data ...")
    start_time = time.time()    
    XD = preprocess_data(XD)
    print(XD.shape)
    end_time = time.time()
    print(f"Execution preprocessing drug time: {end_time - start_time:.2f} seconds")
    print("starting preprocessing protein data ...")
    start_time = time.time()  
    XT = preprocess_data(XT)
    print(XT.shape)
    end_time = time.time()
    print(f"Execution preprocessing protein time: {end_time - start_time:.2f} seconds")
    #check shapes after:
    print(f"zero cols and std.dev. removal -- XD shape: {XD.shape}")
    print(f"zero cols and std.dev. removal -- XT shape: {XT.shape}")



        
    XD_reduced, XT_reduced = select_top_features(XD, XT, Y, top_k=10000)

    XD = XD_reduced
    XT = XT_reduced
    print(XD.shape)
    print(XT.shape)
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)
    print(drug_data.shape)
    print(target_data.shape)
    print(len(affinity))
    return drug_data,target_data,affinity



def reading_our_features(csv_files):
    print(type(csv_files))
    # Verify we found files
    if not csv_files:
        raise ValueError(f"No CSV files found in {base_path}")

    # Load and concatenate all descriptor files
    descriptors = []
    for csv_file in sorted(csv_files):  # Sort for consistent ordering
        
        # time.sleep(4)  # Waits for 4 seconds
        # print(csv_file)
        df = pd.read_csv(csv_file)
        if "sequence_id" in df.columns:
            df = df.drop(columns=["sequence_id"])
            print("sequence_id column removed.")
        if "sequence" in df.columns:
            df = df.drop(columns=["sequence"])
            print("sequence column removed.")
        if "SMILES" in df.columns:
            df = df.drop(columns=["SMILES"])
            print("SMILES column removed.")  
        if "smiles" in df.columns:
            df = df.drop(columns=["smiles"])
            print("smiles column removed.")
        if "smile" in df.columns:
            df = df.drop(columns=["smile"])
            print("smile column removed.")  
        # Convert to numpy array and add to list
        descriptors.append(df.values)

    # Concatenate all descriptors horizontally
    all_descriptors = np.concatenate(descriptors, axis=1)
    
    # Convert to DataFrame to use replace and fillna
    df_descriptors = pd.DataFrame(all_descriptors)
    
    # Replace inf, -inf with NaN, then fill NaNs with 0
    df_descriptors.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_descriptors.fillna(0.0, inplace=True)
    
    # If you want to convert back to NumPy array (optional)
    all_descriptors = df_descriptors.to_numpy()
    
    # Verify no NaN values
    if np.isnan(all_descriptors).any():
        raise ValueError("Descriptor matrices contain NaN values")
        
    return all_descriptors

from sklearn.metrics import roc_auc_score, average_precision_score

# ---- Helper: binarize for AUC/AUPR
def binarize_for_auc(y, q=0.5, higher_is_better=True):
    """
    Returns y_binary and the threshold used.
    If higher_is_better=True: positives are y >= threshold (top-q).
    If higher_is_better=False: positives are y <= threshold (bottom-q).
    """
    y = np.asarray(y).ravel()
    thr = np.quantile(y, q)
    if higher_is_better:
        y_bin = (y >= thr).astype(int)
        score = y  # use predictions as-is
    else:
        y_bin = (y <= thr).astype(int)
        score = -y  # flip convention so 'higher score = more positive'
    return y_bin, thr
       
thresh = 12.1


def get_aupr(Y, P,threshold):
    
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.array(Y)   # convert list to numpy array
    Y = np.where(Y>threshold, 1, 0)
    Y = Y.ravel()
    P = P.ravel()
    
    f = open("aupr_kiba/P_Y.txt", 'w')
    for i in range(Y.shape[0]):
        f.write("%f %d\n" %(P[i], Y[i]))
    f.close()
    
    f = open("aupr_kiba/aupr_metric.txt", 'w')
    subprocess.call(["java", "-jar", "aupr_kiba/auc.jar", "aupr_kiba/P_Y.txt", "list"], stdout=f)
    f.close()
    
    f = open("aupr_kiba/aupr_metric.txt")
    lines = f.readlines()
    aucpr = float(lines[-2].split()[-1])
    f.close()
    
    return aucpr


def experiment(FLAGS, perfmeasure, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print("drug count--: ", XD.shape)
    targetcount = XT.shape[0]
    print("target count: ", XT.shape)
    
    # our features.
    ## -----------Test_iFeatureOmega_colab
    base_path_prot_iFeature = '/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/'
    csv_files = glob.glob(os.path.join(base_path_prot_iFeature, "*.csv"))
    all_descriptors_prot_iFeature = reading_our_features(csv_files)
    print("protein features from domain from IfeatureOmega = ", all_descriptors_prot_iFeature.shape)
    
    base_path_drug_iFeature = '/content/drive/MyDrive/Kd_Meshari/features/Kiba_ligand_features/'
    csv_files = glob.glob(os.path.join(base_path_drug_iFeature, "*.csv"))
    all_descriptors_drug_iFeature = reading_our_features(csv_files)
    print("drug features from IfeatureOmega = ", all_descriptors_drug_iFeature.shape)

    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_ligand_features/Babel_Chemicals/ligand_descriptors.csv']
    all_descriptors_drug_Babel = reading_our_features(csv_files)
    print("drug features from Babel and chem. desc. = ", all_descriptors_drug_Babel.shape)

    # ProtBERT
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/separated_features/ProtBERT/protbert_embeddings.csv']
    ProtBERT = reading_our_features(csv_files)
    print("ProtBERT features = ", ProtBERT.shape)
    # UniRep
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/separated_features/UniRep/UniRep_jax_embeddings.csv']
    UniRep = reading_our_features(csv_files)
    print("UniRep features = ", UniRep.shape)
    # UniRef50
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/separated_features/UniRef50/protein_embeddings.csv']
    UniRef50 = reading_our_features(csv_files)
    print("UniRef50 features = ", UniRef50.shape)
    
    # ESM-1b
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/separated_features/ESM-1b/esm1b_embeddings2.csv']
    ESM_1b = reading_our_features(csv_files)
    print("ESM-1b features = ", ESM_1b.shape)
    # ESM-2
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_features_from_domain_seq/separated_features/ESM-2/esm2_embeddings.csv']
    ESM_2 = reading_our_features(csv_files)
    print("ESM-2 features = ", ESM_2.shape)
    # Morgan_ECFP
    base_path_drug_Morgan_ECFP = '/content/drive/MyDrive/Kd_Meshari/features/Kiba_ligand_features/Morgan_ECFP/'
    csv_files = glob.glob(os.path.join(base_path_drug_Morgan_ECFP, "*.csv"))
    all_descriptors_drug_Morgan_ECFP = reading_our_features(csv_files)
    print("drug features from Morgan_ECFP = ", all_descriptors_drug_Morgan_ECFP.shape)
    # Mordred
    csv_files = ['/content/drive/MyDrive/Kd_Meshari/features/Kiba_ligand_features/Mordred/mordred_additional_features2.csv']
    all_descriptors_drug_mordred = reading_our_features(csv_files)
    print("drug features from mordred = ", all_descriptors_drug_mordred.shape)
    
    # Concatenate features
    # ESM_1b, ESM_2, UniRef50, UniRep, and ProtBERT
    XT = np.concatenate([all_descriptors_prot_iFeature, ESM_1b, ESM_2], axis=1)
    # XT = all_descriptors_prot_iFeature

    # all_descriptors_drug_mordred, all_descriptors_drug_Morgan_ECFP, OR all_descriptors_drug_Babel
    XD = np.concatenate([all_descriptors_drug_iFeature, all_descriptors_drug_Babel, all_descriptors_drug_mordred, all_descriptors_drug_Morgan_ECFP], axis=1)
    # XD = all_descriptors_drug_iFeature

    print("Hi, here is the number of features....")
    print("Drugs features = ", XD.shape)
    print("Proteins features = ", XT.shape)
    
    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    X_train, X_test, y_train, y_test = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                     perfmeasure, FLAGS, dataset)
                                                                     
    # Step 2: Create pipeline
    top_k = 10000  # adjust this value based on your total number of features
    n_estimators = 2500
    learning_rate1=0.1
    reg_alpha1=0.1
    reg_lambda1=10
    # 'reg_alpha': [0, 0.1, 1, 5, 10],
    # 'reg_lambda': [1, 10, 50, 100],
    print(X_train.shape)
    print("features selected count = ",top_k)
    print("n_estimators = ",n_estimators)

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('select', SelectKBest(score_func=f_regression, k=top_k)),
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=n_estimators,
            tree_method='hist', # to make it parallel, use 'hist' OR you can set it to 'auto' if you want CPU.
            device='cuda',
            learning_rate=learning_rate1,
            max_depth=8,
            # subsample=1.0,
            # colsample_bytree=1.0,
            reg_alpha=reg_alpha1,
            reg_lambda=reg_lambda1
        ))
    ])
    print(f"learning_rate = {learning_rate1:.2f}, reg_alpha={reg_alpha1:.1f}, reg_lambda= {reg_lambda1:.1f} ... ")


    start_time = time.time()    
    
    pipe.fit(X_train, y_train)
    end_time = time.time()
    print(f"Execution training time: {end_time - start_time:.2f} seconds")
    
    # Predict
    y_pred = pipe.predict(X_test)
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # cindex = cindex_score(y_test, y_pred)  # Assuming you already have this function
    
    # Print results
    print(f"Test MSE: {mse:.4f}")

    #print(f"Test cindex: {cindex:.4f}")

    mean= mean_absolute_error(y_test,y_pred)
    median= median_absolute_error(y_test,y_pred)
    pearson= pearsonr(y_test,y_pred)[0]
    spearman= spearmanr(y_test,y_pred)[0]
    ci= concordance_index(y_test,y_pred)
    rm2= get_rm2(y_test,y_pred)
    print(f"Test mean: {mean:.4f}")
    print(f"Test median: {median:.4f}")
    print(f"Test pearson: {pearson:.4f}")    
    print(f"Test spearman: {spearman:.4f}")
    print(f"Test RÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²: {r2:.4f}")
    print(f"Test ci: {ci:.4f}")
    print(f"Test rm2: {rm2:.4f}")
    
    # ---- AUC & AUPR settings ----
    # Set this based on your domain:
    # If LOWER target = better (e.g., Kd), set to False. If HIGHER = better, set True.
    HIGHER_IS_BETTER = False
    # Choose the cutoff quantile for positives (0.5 = median split)
    Q = 0.5
    
    # Binarize ground truth based on y_test only (no peeking at y_pred)
    y_bin, thr = binarize_for_auc(y_test, q=Q, higher_is_better=HIGHER_IS_BETTER)
    
    # Scores for AUC/AUPR should align with "higher score => positive".
    # If lower-is-better, flip y_pred so that larger score implies stronger/positive.
    scores_for_auc = y_pred if HIGHER_IS_BETTER else -y_pred
    
    auc  = roc_auc_score(y_bin, scores_for_auc)
    aupr = average_precision_score(y_bin, scores_for_auc)
    print(f"Test auc: {auc:.4f}")
    print(f"Test aupr: {aupr:.4f}")
    print('aupr: %.4f' % get_aupr(y_test, y_pred,thresh))
    
    
    # Compute residuals
    residuals = y_test - y_pred
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left: Predicted vs Actual ---
    axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label="Ideal fit")
    axes[0].set_title(f"Predicted vs Actual\nMSE={mse:.3f}, R²={r2:.3f}", fontsize=13)
    axes[0].set_xlabel("Actual Binding Affinity", fontsize=12)
    axes[0].set_ylabel("Predicted Binding Affinity", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # --- Right: Residuals vs Actual ---
    axes[1].scatter(y_test, residuals, alpha=0.6, edgecolor="k")
    axes[1].axhline(0, color='r', linestyle='--', lw=2)
    axes[1].set_title("Residuals vs Actual", fontsize=13)
    axes[1].set_xlabel("Actual Binding Affinity", fontsize=12)
    axes[1].set_ylabel("Residuals (y_test - y_pred)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    
    # Ensure "figures" directory exists
    os.makedirs("figures", exist_ok=True)
    
    # Save figure inside "figures/"
    save_path = os.path.join("figures", "Kiba_binding_affinity_results.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Figure saved as '{save_path}'")
def run_regression( FLAGS ): 

    perfmeasure = get_cindex

    experiment(FLAGS, perfmeasure)




if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )
