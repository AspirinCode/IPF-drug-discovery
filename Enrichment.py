# coding: utf-8


import pandas as pd
import os
import json
from scipy.stats import hypergeom
from statsmodels.sandbox.stats.multicomp import multipletests as mt
import numpy as np


class Enrichment:
    '''
    Enrichment analysis using a pre compiled library
    Parameters
    ==========
    lib_path: path of directory that has all library files, ending with '/'
    '''
    def __init__(self,lib_path):
        '''
        Load several library files including the followings
        1. Gene id conversion table
        2. Geneid to Term id table
        3. Termid to Term name table
        '''
        self.geneid_table = pd.read_csv(lib_path+'Master Gene Conversion Table.csv',dtype = str)
        self.geneid_table.set_index(self.geneid_table.columns[0],inplace = True)
        self.gene2id = pd.read_table(lib_path+'Gene2Term.txt',index_col = 0,dtype=str)
        self.gene2id['gene_id'] = self.gene2id.index.astype(str)
        self.termid2termname = pd.read_table(lib_path+'termid2termname.txt',header = None, dtype = str)
        self.termid2termname.columns = ['Term_id','Term_name']
        self.termid2termname.set_index('Term_id',inplace = True)
        # get gene2term lookup matrix
        cats = self.gene2id.Term_category.unique().tolist()
        self.dict_g2t = dict(zip(cats,['']*len(cats)))
        for cat in cats:
            tmp = self.gene2id[self.gene2id.Term_category==cat].copy()
            tmp['included'] = 1
            g2t_matrix = tmp.pivot_table(index='gene_id',columns='Term_id',values='included')
            g2t_matrix.dropna(how='all',axis = 0,inplace = True)
            g2t_matrix.fillna(0,inplace = True)
            g2t_matrix = g2t_matrix.astype(int,copy = False)
            self.dict_g2t[cat] = g2t_matrix
        
    def get_pvalues(self,query, ref,
                   M = None,
                   gene_type = 'H_entrez'):
        '''
        Gene list enrichment analyis based on hypergeometric test, need input genelist, and reference libraries. Adapted for massive scale analysis. One test costs about 1 seconds to complete. Returned p-values are FDR corrected.

        Parameters
        ========
        query: a list of genes for enrichment analysis, must be human entrez ids or symbols.
        ref: the ref library, it is a pd.DataFrame of gene id to term id
        gene_type: the type of gene identifier used in the query, can be H_entrez, M_symbol, or 'H_symbol'


        '''
        # convert query to strings, then to human entrez ids
        query = [str(x) for x in query]
        if gene_type == 'H_entrez':
            pass
        elif gene_type == 'M_symbol':
            query = self.geneid_table.set_index('mouse_symbol').ix[query,'h_entrez_id'].dropna().tolist()
        elif gene_type == 'H_symbol':
            query = self.geneid_table.ix[query,'h_entrez_id'].dropna().tolist()
        query = list(set(query))    
        # print(query)
        if M is None:
            M = len(set(query)|set(ref.index.tolist()))
        n = ref.sum().values
        N = len(query)
        X = ref.ix[query].sum()-1
        X = X.values
        M = np.repeat(M,ref.shape[1])
        N = np.repeat(N,ref.shape[1])
        pval = hypergeom.sf(X,M,n,N)
        fdr = mt(pval,method='fdr_bh')[1]
        
        report = pd.DataFrame(fdr,index = ref.columns, columns = ['FDR_adjusted_pvalue'])
        report = report[report.FDR_adjusted_pvalue<=0.05]
        report['Term_name'] = self.termid2termname.ix[report.index,0]
        report.sort_values('FDR_adjusted_pvalue',inplace=True)
        return pval,fdr,report
    

