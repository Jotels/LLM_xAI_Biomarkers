import pandas as pd
from scipy.io import mmread
import scanpy as sc
import numpy as np
import anndata as ad
import logging
import gprofiler
import os
from scipy import sparse
from functions.mofa_candidate import mofa_candidates
from functions.partition_assignemnt import partition_assignment

sc.settings.verbosity = 1

def load_and_split(dataset_number, min_cells):
    load_path = 'standard_anndata/UC%s_final.h5ad' %( dataset_number)
    matrix_ = sc.read_h5ad(load_path)#, backed=None, *, as_sparse=(), as_sparse_fmt=<class 'scipy.sparse._csr.csr_matrix'>, chunk_size=6000)
    info_ = matrix_.obs[['sample', 'disease_state']].reset_index()[['sample', 'disease_state']]
    matrix_.layers["log_transformed"] = matrix_.X
    annotation_scBert = '../SingleCell2/scBERT_Annotation/UC%s_Predictions.txt' % (dataset_number)
    annotation_scBert_df = pd.read_table(annotation_scBert, header=None)
    annotation_scBert_df.columns = ['Cell_type']
    matrix_.obs["Cell_type"] = list(annotation_scBert_df['Cell_type'])
    #print(matrix_.obs.Cell_type.value_counts())
    df1  = pd.DataFrame(matrix_.obs.Cell_type.value_counts())
    df1  = df1.reset_index()
    df1.columns =['Cell_type','Cell_Counts']
    df1.Cell_Counts = df1.Cell_Counts.astype(int)
    min_cells = 10#int(len(matrix_.obs)/100)
    print('Keeping cell types with at least {} cells for UC{}'.format(min_cells, i))

    cell_toKeep = list(df1.query(f'Cell_Counts >= {min_cells}').Cell_type)
    
    matrix_ = matrix_[matrix_.obs.Cell_type.isin(cell_toKeep)]
    cl_numbers = partition_assignment(np.array(matrix_.obs.PatientID), np.array(matrix_.obs.UC_state), 1, 2)
    matrix_.obs['CV_SPLIT'] = cl_numbers
    train = matrix_[matrix_.obs.CV_SPLIT == 0.0]
    sc.pp.highly_variable_genes(train, flavor='seurat', n_top_genes=4000)
    print('\n','Number of highly variable genes: {:d}'.format(np.sum(train.var['highly_variable'])))
    print('\n','Number of kept cell types: {:d}'.format(len(cell_toKeep)))
    matrix_var_filtered = matrix_[:,train.var.highly_variable].copy()
    var_genes = list(matrix_var_filtered.var.index)
    
    return matrix_, var_genes, cell_toKeep


#%%capture
from functools import reduce
all_genes = list()
all_cells = list()
df_dict = dict()
for i in [1,2,3,5]:
    adata, var_genes_, cell_toKeep_ = load_and_split(i, 500)
    all_genes.append(var_genes_)
    all_cells.append(cell_toKeep_)
    df_dict_tmp = dict()
    for cell_type in cell_toKeep_:
        n_classes = adata[adata.obs['Cell_type']==cell_type].obs['UC_state'].value_counts()
        if len(n_classes) == 2:
            df_dict_tmp[cell_type] = adata[adata.obs['Cell_type']==cell_type].to_df()
            df_dict_tmp[cell_type]['UC_state'] = adata[adata.obs['Cell_type']==cell_type].obs['UC_state']
            df_dict_tmp[cell_type]['CV_SPLIT'] = adata[adata.obs['Cell_type']==cell_type].obs['CV_SPLIT']
        
    df_dict["UC{}".format(i)] = df_dict_tmp
    
final_list_feats= list(set.intersection(*map(set,all_genes)))
final_list_feats = final_list_feats +['UC_state']
final_list_cells = []
for i, cell_list_1 in enumerate(all_cells):
    for j, cell_list_2 in enumerate(all_cells[i+1:]):
        final_list_cells+=list(set(cell_list_1) & set(cell_list_2) )
final_list_cells = list(set(final_list_cells))

import feyn

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#import wikipedia
import collections
from functions.tpm import counts2tpm
from functions.model_eval import *


import os
performance_dict_train = dict()
for i, cells_list in enumerate(all_cells):
    if i == 3:
        current_dataset = 5
    else:
        current_dataset = i + 1
        if current_dataset != 1:

            performance_dict_train["UC_{}".format(current_dataset)] = dict()

            for cell_type in cells_list:

                target = "UC_state"
                save_string = "/home/jupyter-marco.salvatore/jupyter_lab_files/single_cell_new/models/UC{}/UC{}_{}".format(current_dataset, current_dataset, cell_type)
                os.mkdir(save_string)
                stypes = {}

                random_seed = 42
                df_1 = df_dict['UC{}'.format(current_dataset)][cell_type]

                train = df_1
                train = train.drop(columns=['CV_SPLIT'])

                sw = np.where(train[target] == 1, np.sum(train[target] == 0)/sum(train[target]), 1)    

                ql = feyn.QLattice(random_seed)
                models = ql.auto_run(
                    data=train[final_list_feats],
                    output_name=target,
                    kind="classification",
                    criterion='bic',
                    stypes=stypes,
                    sample_weights=sw,
                    max_complexity=4,
                    n_epochs=30,
                )
                performance_dict_train["UC_{}".format(current_dataset)][cell_type] = models[0].roc_auc_score(train)
                for n, model in enumerate(models):          
                    model.save(file=save_string+"/UC{}_{}_model_{}".format(current_dataset, cell_type, n))