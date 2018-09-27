
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy import stats
from operator import itemgetter
from scipy.stats import sem
from scipy.stats import ks_2samp
from scipy.stats import probplot
import statsmodels.api as sm
import statsmodels.stats.power as smp
import pylab
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


def tcgaTissueSelect(samp_list, samp_name, cnv_df, exp_df, save=False):
    """
    subsamples tcga cnv and exp dataframes into tissue-specific sample columns
    
    :param samp_list: list of str, tissue sample IDs (len 12) from TCGA cases
    :param samp_name: str, name of TCGA tissue source
    :param cnv_df: df int32, thresholded or masked copy number values, gene (index) x samples
    :param exp_df: df float32, thresholded or masked expression values, gene (index) x samples
    :param save: bool or str, str of dir/ where to save, default False
    
    returns tissue_cnv_df, tissue_exp_df
    
    :usage:
        brca_samples = list(pd.read_json('tcga/BRCA_cases_2018-03-23.json')['submitter_id'])
        brca_cnv, brca_exp = tcgaTissueSelect(brca_samples, 'brca', cnv, exp, save='tcga/')
    """
    # list of longer IDs (16) whose first 12 match the IDs in a tissue
    c_samples = [col[:16] for col in cnv_df.columns if col[:12] in samp_list]
    e_samples = [col[:16] for col in exp_df.columns if col[:12] in samp_list]

    # list of intersecting IDs whose first 12 match the IDs in a tissue
    samples = list(set(c_samples).intersection(e_samples))

    # slice down all IDs of DFs so they can match
    cnv_df.columns = [col[:16] for col in cnv_df.columns]
    exp_df.columns = [col[:16] for col in exp_df.columns]

    tissue_cnv_df = cnv_df[samples]
    tissue_exp_df = exp_df[samples]
    
    if save:
        tissue_cnv_df.to_csv(save + 'cnv_' + samp_name + '.tab.gz', sep='\t', compression='gzip')
        tissue_exp_df.to_csv(save + 'exp_' + samp_name + '.tab.gz', sep='\t', compression='gzip')
    
    return tissue_cnv_df, tissue_exp_df


# In[36]:

# subsample exp and cnv dfs by tissue (lung) and histolgical subtype (sclc, adno, sqms)

def ccleTissueSelect(expdf_fn, cnvdf_fn, celldf_fn, tissue, subtype, out_dir=''):
    """
    separates samples by subtype returns subsampled df (and pickles them for later!)
    
    :param expdf_fn: str, filename of pickled expression dataframe, (check in /clove/tissueDf_raw)
    :param cnvdf_fn: str, filename revealer expression calls, (CCLE_DEL_calls.pickle)
    :param celldf_fn: str filename of pickled CCLE cell info df (ccle_cell_info_df.pickle)
    :param tissue: str, name of primary site matching CCLE cell info file
    :param subtype: str, name of histological subtype mathcing CCLE cell info file
    :param outdir: str, name of directory to drop pickled dfs, (clove/dichotomy_*sample*V*sample*)
    
    returns expression df, copy call df
    
    """
    
    exp = pd.read_pickle(expdf_fn)
    cnv = pd.read_pickle(cnvdf_fn)
    celldf = pd.read_pickle(celldf_fn)
    
    cells = celldf.loc[(celldf['Hist Subtype1'] == subtype) & (celldf['Site Primary'] == tissue)].index
    exp = exp[list(set(cells).intersection(exp.columns))]
    cnv = cnv[list(set(cells).intersection(cnv.columns))]
    
    exp.to_pickle(out_dir+'_'.join([tissue,subtype,'exp','df'])+'.p')
    cnv.to_pickle(out_dir+'_'.join([tissue,subtype,'cnv','df'])+'.p')
    
    return (exp, cnv)


# In[33]:


def powerFilter(expdf, cnvdf, var_thresh=0.2, n_thresh=5):
    """
    filter gene expression to var and copy number to n threshold across 
    
    :param expdf: pandas dataframe, expression by samples (floats)
    :param cnvdf: pandas dataframe, copy by samples (binary)
    :param var: float, gene_var with expression < var are excluded
    :param n:: int, context/no-context array minimum n length
    
    :returns: filtered expdf, filtered cnvdf
    """
    
    # filter expression values
    filt_expdf = expdf[expdf.var(axis=1) > var_thresh]
    filt_expdf['exp'] = filt_expdf.index
    filt_expdf.set_index('exp', drop=True, inplace=True)
    
    # filter copy number values
    # min context, min non-context, respectively
    samples = cnvdf.shape[1]
    filt_cnvdf = cnvdf[(cnvdf.sum(axis=1) >= n_thresh) &
                       (samples - cnvdf.sum(axis=1) >= n_thresh)]
    filt_cnvdf['cnv'] = filt_cnvdf.index
    filt_cnvdf.set_index('cnv', drop=True, inplace=True)
    
    return (filt_expdf, filt_cnvdf)


# In[34]:

def expAbberationFilter(expdf, combined=False, amp=False, dele=False, mut=False):
    """
    fitler gene expression confounded by dna-level abberations
    
    :param expdf: pandas dataframe, expression by samples (floats)
    :param combined: bool or pandas dataframe, combined thresholded calls {2,1,0,-1,-2}
    :param amp: bool or str, file handle of ampllification calls df file handle, default False 
    :param dele: bool or str, file handle of deletion calls df, default False
    :param mut: bool or str, file handle of mutation calls df file handle, default False

    :returns: filtered expdf
    """
    
    if combined:
        return expdf.where(combined == 0, np.nan)
    
    else:
        zeros = pd.DataFrame(np.zeros(expdf.shape), index=expdf.index, columns=expdf.columns) == 0
        if amp:
            amp_mask = pd.read_pickle(amp) == 0
        else: 
            amp_mask = zeros
        if dele:
            del_mask = pd.read_pickle(dele) == 0
        else: 
            del_mask = zeros
        if mut:
            mut_mask = pd.read_pickle(mut) == 0
        else:
            mut_mask = zeros

        return expdf.where(amp_mask & del_mask & mut_mask, np.nan)


# In[80]:

def subtypeFilter(subtype_names, tissue_names, dfout):
    """
    returns dfs of exp and cnv restricted to subtypes and or tissue_names
    """
    pass


# In[ ]:

def dfIntersection(reducedf, dftofilter):
    """
    filter a dftofilter on sample and gene and 
    """
    pass


# In[ ]:

def mainFitler(expdf, cnvdf, var=0.2, n=5, amp_fh=False, dele_fh=False, mut_fh=False, save=False):
    """
    returns expdf and cnvdf filtered on power and abberations, used for CCLE data mostly
    
    :param expdf: pd df, expression dataframe, (check in /clove/data/)
    :param cnvdf: pd df, dataframe of copy number calls, (dataCCLE_DEL_calls.pickle)
    :param var: float, minimum gene variance to filter low-variance pairs
    :param amp_fh: str, filename of pickled dataframe of binarized ampflication calls (eg REVEALER)
    :param dele_fh: str, filename of pickled dataframe of binarized deletion calls (eg REVEALER)
    :param mut_fh: str, filename of pickled dataframe of binarized mutation calls (eg REVEALER)
    :param save: bool, True saves the filtered dfs with filter conditions in filename
    
    :returns: filtered expdf, filtered cnvdf
    """
    # filter on power
    expdf, cnvdf = powerFilter(expdf, cnvdf, var, n)
    
    # filter out abberations  
    expdf = expAbberationFilter(expdf, amp_fh, dele_fh, mut_fh)
    
    if save:
        filt_cond = 'v' + ''.join(str(var).split('.')) + 'n' + str(n)
        expdf.to_pickle('_'.join(expdf_fh.split('_')[:-1] + [filt_cond] + [expdf_fh.split('_')[-1]]))
        cnvdf.to_pickle('_'.join(cnvdf_fh.split('_')[:-1] + [filt_cond] + [cnvdf_fh.split('_')[-1]]))   
    
    return (expdf, cnvdf)

####################################################################################################
### GRAPHING TOOLS


def calcstats(_df, list1, list2, list1name='ipl_avg_TC', list2name='ipl_avg_nonTC'):
    df = _df.copy()
    for idx, row in df.iterrows():
        grp1 = df.loc[idx, list1]
        grp2 = df.loc[idx, list2]
        df.loc[idx, 'ks_d'] = stats.ks_2samp(grp1, grp2)[0]
        df.loc[idx, 'ks_p'] = stats.ks_2samp(grp1, grp2)[1]
        df.loc[idx, 'tt_s'] = stats.ttest_ind(grp1, grp2)[0]
        df.loc[idx, 'tt_p'] = stats.ttest_ind(grp1, grp2)[1]
        df.loc[idx, 'avg_grp1'] = grp1.mean()
        df.loc[idx, 'avg_grp2'] = grp2.mean()
        
    df = df.loc[:, 'ks_d':'avg_grp2'].sort(['avg_grp1'], ascending=[False])
    df.columns = ['ks_d', 'ks_p', 'tt_s', 'tt_p', list1name, list2name]

    model = pd.ols(y=df[list1name], x=df[list1name])
    df['y_hat'] = model.y_fitted
    df['resid'] = model.resid

    return df

def clustertxt(fh):
    with open(fh, 'r') as clus:
        lis = [line.strip() for line in clus]
    return lis


def joint_annotate(df, title, x_name='ipl_avg_TC', y_name='ipl_avg_nonTC', annotate_col='resid', annotate_n=5):
    g = sns.jointplot(x_name, y_name, data=df, kind="reg",
                      xlim=(-10, 10), ylim=(-10, 10), color="r", size=7)

    head = df.sort_values(by=[annotate_col], ascending=[False]).head(annotate_n)
    tail = df.sort_values(by=[annotate_col], ascending=[False]).tail(annotate_n)

    def ann(row, x_name,y_name):
        ind = row[0]
        r = row[1]
        plt.gca().annotate(ind, xy=(r[x_name], r[y_name]), 
                xytext=(2,2) , textcoords ="offset points", )

    for row in head.iterrows():
        ann(row,x_name,y_name)
    for row in tail.iterrows():
        ann(row,x_name,y_name)
    g.fig.suptitle(title)
    plt.show()

    
# In[46]:

# to use with vectorized functions on pandas series of numpy arrays

def getMean(arr):
    return arr.mean()

def getLen(arr):
    return len(arr)

def getVar(arr):
    return arr.var()


# In[10]:

def meanVar(df):
    """
    calculates mean gene expression variance for shrinkage
    """
    return df.var(axis=1).mean()


# In[73]:

def getContexts(exp_g, cnv_g, cmask):
    lossTrue = cmask.loc[cnv_g]
    loss = expdf.loc[exp_g].where(lossTrue, np.nan).dropna()
    no_loss = np.array(expdf.loc[exp_g].where(~lossTrue, np.nan).dropna())
    return loss, no_loss


# In[4]:

from sklearn.utils import shuffle
def scrambleDF(df):
    """
    shuffles the interior cells of a pandas dataframe
    
    :params df: pandas dataframe
    returns scrambled df
    """
    
    rows = df.index
    cols = df.columns
   
    df_shuf = shuffle(df).reset_index(drop=True)
    df_shuf = df_shuf.set_index(rows)
    
    df_shuf = df[shuffle(cols)]
    df_shuf.columns = cols
    
    return df_shuf


# In[5]:

# article on effect size 
# http://www.leeds.ac.uk/educol/documents/00002182.htm
# https://stackoverflow.com/a/21532472/4154548

def cohenD(x,y):
    """
    computes Cohen's effect size for two arrays x and y
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1) * np.std(x, ddof=1) ** 2 + (ny-1) * np.std(y, ddof=1) ** 2) / dof)


# In[35]:

def t_welch(nx, ny, mx, my, vx, vy, fudge, tails=2):
    """
    welch's t-test for two unequal-size samples, not assuming equal variances
    
    allows shrinkage via adding average variance 'fudge'factor to sample variances
    https://gist.github.com/jdmonaco/5922991
    """
    vx += fudge
    vy += fudge
    # Welch-Satterthwaite equation
    df = int((vx/nx + vy/ny)**2 / ((vx/nx)**2 / (nx - 1) + (vy/ny)**2 / (ny - 1)))
    t = (mx - my) / np.sqrt(vx/nx + vy/ny)
#     TODO: IMPLEMENT BEST p-val CALCULATION
#     p = tails * st.t.sf(abs(t_obs), df)
#     p = distributions.t.sf(np.abs(t), df) * tails
    return t


# In[36]:

# randomly select 10000 cnv and exp genes to form 10000 pairs


def explicitPairContextStat(expdf, cnvdf, exp_lis=False, cnv_lis=False, cat_df=False, nan_style='omit', permute=False):
    """
    takes exp and cnv genes (either all or explicitand returns pair summary statistics
    
    :param n_samp: int, number of random samples to take
    :param expdf: pandas dataframe, expression by sample 
                    (hopefully filtered with mainFilter, tissue specific, with matching samples in cnv)
    :param cnvdf: pandas dataframe, binarized mask 5(1=del, 0=nodel) deletion by sample 
                    (hopefully filtered with mainFilter, tissue specific, with matching samples in exp)
    :param exp_lis: list of str, HUGO gene names in expdf to restrict to, default is False (use all genes in expdf)
    :param exp_lis: list of str, HUGO gene names in cnvdf to restrict to, default is False (use all genes in cnvdf)
    :param in_df: pandas dataframe, previous calculations to concat new results to, used in while loop to get n_samp
    :param nan_style: str, how the stats.ttest_ind treats NANs, {‘propagate’, ‘raise’, ‘omit’}
    :param permute: bool, True will calculate pairs with randomly permuted expression matrix as null model
    
    returns df[['exp', 'cnv', 'cntxt_pos_mu', 'cntxt_neg_mu', 
                'cntxt_pos_var', 'cntxt_neg_var', 
                'cntxt_pos_n', 'cntxt_neg_n']]
    """
    
    cells = list(set(cnvdf.columns).intersection(expdf.columns))
    expdf = expdf[cells]
    cmask = cnvdf[cells] == 1
    
    if type(exp_lis) != bool:
        exp_samp = set(expdf.index).intersection(exp_lis)
        for gene in exp_lis:
            if gene not in expdf.index:
                print('{} not found in expdf.index.  Omitted'.format(gene))
    else:
        exp_samp = expdf.index
    
    if type(cnv_lis) != bool:
        cnv_samp = set(cnvdf.index).intersection(cnv_lis)
        for gene in cnv_lis:
            if gene not in cnvdf.index:
                print('{} not found in cnvdf.index.  Omitted'.format(gene))
    else:
        cnv_samp = cnvdf.index
    print('attempting {} comparisons with current parameters'.format(len(exp_samp) * len(cnv_samp)))
    r={'exp':[],'cnv':[]}
    for pair in itertools.product(exp_samp, cnv_samp):
        r['exp'].append(pair[0])
        r['cnv'].append(pair[1])
    df = pd.DataFrame(r)
    
    pos_n, neg_n, = [], []
    pos_mu, neg_mu = [], []
    pos_var, neg_var = [], []
    cohens_d = []
    np_t_s, np_p_s = [], []
    np_t_w, np_p_w = [], []
    
    df
    
    for row in df.itertuples():
        # mask cnv contexts onto expression data
        pos = np.array(expdf.loc[row.exp][cmask.loc[row.cnv]])
        neg = np.array(expdf.loc[row.exp][~cmask.loc[row.cnv]])
        
        # calculate n
        pos_n.append(len(pos))
        neg_n.append(len(neg))
        
        # calculate mu
        pos_mu.append(pos.mean())
        neg_mu.append(neg.mean())
        
        # calculate var
        pos_var.append(pos.var())
        neg_var.append(neg.var())
        
        # calculate cohen's d
        cohens_d.append(cohenD(pos, neg))
        
        # calculate t_stat, welch
        t, p = stats.ttest_ind(pos, neg, nan_policy=nan_style, equal_var=True)
        np_t_w.append(t)
        np_p_w.append(p)
        
        if permute:
            pos = np.array(expdf.loc[row.exp][cmask_n.loc[row.cnv]])
            neg = np.array(expdf.loc[row.exp][~cmask_n.loc[row.cnv]])
            t, p = stats.ttest_ind(pos, neg, nan_policy=nan_style, equal_var=True)
            np_t_w_null.append(t)
            np_p_w_null.append(p)
            
    df['pos_n'] = pos_n
    df['neg_n'] = neg_n
    df['pos_mu'] = pos_mu
    df['neg_mu'] = neg_mu
    df['pos_var'] = pos_var
    df['neg_var'] = neg_var
    df['cohens_d'] = cohens_d
    df['np_t_w'] = np_t_w
    df['np_p_w'] = np_p_w
    
    if permute:
        df['np_t_w_null'] = np_t_w_null
        df['np_t_w_null'] = np_t_w_null
    
    df.dropna(inplace=True)

    right = expdf.rename_axis('exp', axis=0) 
    right['gene_var_exp'] = right.var(axis=1)
    right = right.reset_index()
    
    df = pd.merge(df, right[['exp','gene_var_exp']], on='exp')
    
    if cat_df:
        return pd.concat([cat_df, df])
        
    return df
    

def randomPairContextStat(n_samp, expdf, cnvdf, cat_df=False, nan_style='omit', permute=False):
    """
    takes exp and cnv genes and returns pair summary statistics
    
    :param n_samp: int, number of random samples to take
    :param expdf: pandas dataframe, expression by sample (hopefully filtered and tissue specific)
    :param cnvdf: pandas dataframe, binarized mask 5(1=del, 0=nodel) deletion by sample
    :param in_df: pandas dataframe, previous calculations to concat new results to, used in while loop to get n_samp
    :param nan_style: str, how the stats.ttest_ind treats NANs, {‘propagate’, ‘raise’, ‘omit’}
    :param permute: bool, True will calculate pairs with randomly permuted expression matrix as null model
    
    returns df[['exp', 'cnv', 'cntxt_pos_mu', 'cntxt_neg_mu', 
                'cntxt_pos_var', 'cntxt_neg_var', 
                'cntxt_pos_n', 'cntxt_neg_n']]
    """
    
    exp_samp = expdf.sample(n=n_samp, replace=True).index.values
    cnv_samp = cnvdf.sample(n=n_samp, replace=True).index.values
    cells = list(set(cnvdf.columns).intersection(expdf.columns))
    expdf = expdf[cells]
    cmask = cnvdf[cells] == 1
    if permute:
        cmask_n = scrambleDF(cmask)
        np_t_w_null, np_p_w_null = [], []

    df = pd.DataFrame(pd.Series(exp_samp), columns=['exp'])
    df['cnv'] = pd.Series(cnv_samp)
    pos_n, neg_n, = [], []
    pos_mu, neg_mu = [], []
    pos_var, neg_var = [], []
    cohens_d = []
    np_t_s, np_p_s = [], []
    np_t_w, np_p_w = [], []
    
    for row in df.itertuples():
        # mask cnv contexts onto expression data
        pos = np.array(expdf.loc[row.exp][cmask.loc[row.cnv]])
        neg = np.array(expdf.loc[row.exp][~cmask.loc[row.cnv]])
        
        # calculate n
        pos_n.append(len(pos))
        neg_n.append(len(neg))
        
        # calculate mu
        pos_mu.append(pos.mean())
        neg_mu.append(neg.mean())
        
        # calculate var
        pos_var.append(pos.var())
        neg_var.append(neg.var())
        
        # calculate cohen's d
        cohens_d.append(cohenD(pos, neg))
        
        # calculate t_stat, welch
        t, p = stats.ttest_ind(pos, neg, nan_policy=nan_style, equal_var=True)
        np_t_w.append(t)
        np_p_w.append(p)
        
        if permute:
            pos = np.array(expdf.loc[row.exp][cmask_n.loc[row.cnv]])
            neg = np.array(expdf.loc[row.exp][~cmask_n.loc[row.cnv]])
            t, p = stats.ttest_ind(pos, neg, nan_policy=nan_style, equal_var=True)
            np_t_w_null.append(t)
            np_p_w_null.append(p)
            
    df['pos_n'] = pos_n
    df['neg_n'] = neg_n
    df['pos_mu'] = pos_mu
    df['neg_mu'] = neg_mu
    df['pos_var'] = pos_var
    df['neg_var'] = neg_var
    df['cohens_d'] = cohens_d
    df['np_t_w'] = np_t_w
    df['np_p_w'] = np_p_w
    
    if permute:
        df['np_t_w_null'] = np_t_w_null
        df['np_t_w_null'] = np_t_w_null
    
    df.dropna(inplace=True)
    df['t_shrnk_glob'] = np.vectorize(t_welch)(df['pos_n'], df['neg_n'], 
                                               df['pos_mu'], df['neg_mu'], 
                                               df['pos_var'], df['neg_var'], 
                                               meanVar(expdf))
    
    right = expdf.rename_axis('exp', axis=0) 
    right['gene_var_exp'] = right.var(axis=1)
    right = right.reset_index()
    
    df = pd.merge(df, right[['exp','gene_var_exp']], on='exp')
    
    if cat_df:
        return pd.concat([cat_df, df])
        
    return df


from scipy.stats import pearsonr
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

def rolling_similarity(df, group='cnv', partners='exp', data='np_t_w', how='pearson', locus='chromosome'):
    """
    computes similarity between array of data for one gene partners and each successive chromosomal neighbor
    
    :param df: pd df, clove results, sorted in ascending order of chromosome locus
    :param group: str, df column label on which to form gene neigbors, default 'cnv'
    :param members: str, df column label on which to pair partners with gene neighbors, default 'exp'
    :param data: str, df column label of source data to populate arrays, default 'np_t_w' (clove t-stats)
    :param how: str, choice of: {pearson, euclidian, cosine}
    :param align: bool, False compares two arrays which don't necessarily have same gene index
    
    returns pd df of pairs, scores, and chosen similary metric, ordered by locus
    """
    
    unique_genes = df[group].unique()
    df = df[[locus, group, partners, data]].sort_values(by=locus)
    results = []
    for idx, g0 in enumerate(unique_genes):
        if idx < len(unique_genes) - 1:
            for idxk, gk in enumerate(unique_genes[idx+1:]):
                g1 = gk  # unique_genes[idx+1]
                merged = pd.merge(df[df[group] == g0], df[df[group] == g1][[partners,group,data]], how='inner', on=partners)
                merged.columns = [locus,'cnv_g0','exp','clove_g0','cnv_g1','clove_g1']

                if how == 'pearson':
                    # produces NaN
                    cols = [group, partners, how ,'pval']
                    pear, pval = pearsonr(merged['clove_g0'], merged['clove_g1'])
                    results.append([g0,g1,pear,pval])
                elif how == 'euclidian':
                    cols = [group, partners, how]
                    d = distance.euclidean(merged['clove_g0'], merged['clove_g1'])
                    # error: (offx>=0 && offx<len(x)) failed for 2nd keyword offx: dnrm2:offx=0
                    results.append([g0, g1, d])
                elif how == 'cosine':
                    cols = [group, partners, how]
                    # error: (offx>=0 && offx<len(x)) failed for 2nd keyword offx: dnrm2:offx=0
                    d = cosine_similarity(merged['clove_g0'], merged['clove_g1'])[0][0]  
                    results.append([g0, g1, d])

    return pd.DataFrame(results, columns=cols)


def map_locus(df, map_col='cnv', lf_fh='data/hugo_to_locus.txt', lf_col=['Approved Symbol', 'Chromosome'], sort=True):
    """
    maps pandas column of HUGO genes to chromosome locus
    
    :param df: pd df, typically CLOvE results df
    :param map_col: str, columns of above df on which to map chromosome locus, default 'cnv'
    :param lf_fh: str, file path and name of chromosome locations, default 'data/hugo_to_locus.txt'
                       default file downloaded from https://www.genenames.org/cgi-bin/download
    :param lf_col: list of str, collection of cols from lf_fh to keep
    :param sort: bool, sorts by chromosome location, default True
    
    returns df with mapped chromosomal locations in new column 'chromosome'
    """
    chrloc = pd.read_csv(lf_fh, sep = '\t')[lf_col]
    chrloc.columns = [map_col, 'chromosome']
    
    if sort:
        return pd.merge(df, chrloc, on='cnv', how='inner').sort_values(by='chromosome')
    else:
        return pd.merge(df, chrloc, on='cnv', how='inner')


def sample_by_loc(df, chr_num, arm='p', pos_sort=True):
    """
    sample clove pairs by location of cnv gene
    
    :param df: pd df, output of clove computation
    :param chr_num: int, chromosome number to sample on
    :param arm: str, p or q arm
    :param pos_sort: bool, sorts by chr position, default true
                        
    """
    if arm.lower() == 'p':
        df['arm'] = df['chromosome'].str.extract('(p)', expand=True)
    if arm.lower() == 'q':
        df['arm'] = df['chromosome'].str.extract('(q)', expand=True)
    # drop NaN locations (65 default for breast cloves)
    df.dropna(inplace=True)

    # subsample to chromosome number
    df['chr'] = df['chr'].astype(int)
    
    #sort by pos
    if pos_sort:
        return df[df['chr'] == chr_num].sort_values(by='chromosome')
    
    return df[df['chr'] == chr_num]




def prepare_vv(exp, cnv, cloves, sig=0.01):
    """Prepare CLoVE computations df for vulnerability vector
    
    :param cloves: df, pair x calculations from clove output
    :param exp: df, gene x sample microarray or RNAseq, nan where no expression counts exist
    :param cnv: df, gene x sample calls from GISTIC, 0=WT 1=loss
    :param sig: float, p-value threshold for signifiance, default=0.01
    
    returns cutoff'd pair x calculations df from clove output with 'pair' col
    """
    
    if 'pair' not in cloves.columns:
        cloves['pair'] = cloves['exp'] + cloves['cnv']
    if exp.index.dtype != 'object':
        exp.set_index(keys='gene_id', drop=True, inplace=True)
    if cnv.index.dtype != 'object':
        cnv.set_index(keys='gene_id', drop=True, inplace=True)
    
    return exp, cnv, cloves[cloves['np_p_w'] <= sig]


def vulnerability_vector_count(exp, cnv, hits):
    """Counts meaningful clDEGs (copy-loss differential expression genes) from loss contexts
    
    :param exp: df, gene x sample microarray or RNAseq, nan where no expression counts exist
    :param cnv: df, gene x sample calls from GISTIC, 0=WT 1=loss
    :param hits: df, pair x calculations from clove output, cutoff to sig, with 'pair' col
    
    returns gene x sample df of counts of significant expression linkages"""
    
    pat_essential = pd.DataFrame(0, index=exp.index, columns=exp.columns)
    
    for col in exp.columns:
        
        g_exp = exp[col].dropna().index.tolist()  # non-NaN genes from exp matrix
        g_cnv = cnv[cnv[col] == 1][col].index.tolist()  # genes with value 1 from cnv matrix
        
        df = hits.loc[(hits['exp'].isin(g_exp)) & (hits['cnv'].isin(g_cnv))]
        df = pd.DataFrame(df['exp'].value_counts())
        
        if df.shape[0] != 0:
            df.columns=[col]
            pat_essential[col] = df
            
    pat_essential.replace(np.nan, 0, inplace=True)
    return pat_essential.astype(np.int32)


def correlate_df_cols(df_1, df_2, comb=False):
    """
    finds pearson coefficient between columns of two pandas dataframes
    
    :param df_1: pandas dataframe
    :param df_2: pandas dataframe
    :param comb: bool, changes how/which column comparisons are made:
                    True: correlation between all combinations of cols
                    False: correlation between matching column headers (default)
    
    returns pandas dataframe of correlations
    """
    
    idx_1, idx_2 = df_1.index, df_2.index
    labels = idx_1.intersection(idx_2)
    df_1, df_2 = df_1.loc[labels], df_2.loc[labels]
    cldeg, dep_breast = df_1, df_2
    
    if comb:
        cldeg_depBreast_corr = [['sample','cell','pear_coeff','p_val']]
        completed = []
        for pair in itertools.product(cldeg.columns, dep_breast.columns):
            if (pair[0] != pair[1]) & (pair[::-1] not in completed):
                corr = pearsonr(cldeg[pair[0]], dep_breast[pair[1]])
                cldeg_depBreast_corr.append([pair[0], pair[1], corr[0], corr[1]])
                completed.append(pair)
    else:
        cldeg_depBreast_corr = [['cell','pear_coeff','p_val']]
        for cell in cldeg.columns:
            if cell in dep_breast.columns:
                corr = pearsonr(cldeg[cell], dep_breast[cell])
                cldeg_depBreast_corr.append([cell, corr[0], corr[1]])

    cols = cldeg_depBreast_corr.pop(0)
    df_corr = pd.DataFrame(cldeg_depBreast_corr)
    df_corr.columns = cols
    return df_corr


def hire_clust(df, fig_fh=False):
    """
    performs hirearchical clustering on columns in df, plots dendro and heatmap
    
    :param df: pandas dataframe of floats with column headers to cluster on
    :param fig_fh: str, filehandle of figure jpg if saving, defaults to False
    """
    data_dfT = df.T
    data_dist = pdist(data_dfT) # computing the distance
    data_link = linkage(data_dist) # computing the linkage
    
    # Compute and plot first dendrogram.
    fig = plt.figure(figsize=(8,8))
    # x ywidth height
    ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
    Y = linkage(data_dist, method='single')
    Z1 = dendrogram(Y, orientation='right',labels=data_dfT.index) # adding/removing the axes, data.dtype.names
    ax1.set_xticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = squareform(data_dist)
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    if fig_fh:
        plt.savefig(fig_fh)
    else:
        plt.show()



# In[74]:

# randomly select 10000 cnv and exp genes to form 10000 pairs

def stack_process(fh, score='pearson', diff='same_group?'):
    '''
    processes df for graphing
    
    :param: fh: path/file of pandas datatframe of congruent or ttest scores 
                df columns = ['level_0', 'level_1', 'Score', 'same_group', 'tissue']
    '''
    # read in results df with tissue groupings and stuff
    cong = pd.read_pickle(fh)
    cong.columns = ['mrna','cnv',score, diff, 'tissue']
    cong.reset_index(inplace=True, drop=True)
    cong['tissue'].replace(to_replace='haematopoietic_and_lymphoid_tissue', value='haem_and_lymph', inplace=True)
    cong['tissue'].replace(to_replace='central_nervous_system', value='c_n_s', inplace=True)
    cong['tissue'].replace(to_replace='upper_aerodigestive_tract', value='aerodigestive', inplace=True)


    # filter col entries not found in membership dict keys, add module columns
    cong['keep'] = (cong['mrna'].isin(memberships.keys())) & (cong['cnv'].isin(memberships.keys()))
    cong = cong[cong['keep'] == True]
    del cong['keep']

    cong['mrna_mods'] = cong['mrna'].map(memberships)
    cong['cnv_mods'] = cong['cnv'].map(memberships)
    cong['all_mods'] = cong['mrna_mods'] + cong['cnv_mods']
    
    return cong


def get_common(cong):
    '''
    finds common gene module between two genes in stacked congruent file
    
    :param: cong: pandas datatframe of congruent scores
    '''
    def check_row (row_mods, col_mods):
        return set(row_mods).intersection(set(col_mods))

    cong['common'] = ''

    for index in cong.index:
        intersect = check_row(cong.loc[index,'mrna_mods'], cong.loc[index,'cnv_mods'])
        cong.set_value(index, 'common', intersect)
    
    return cong


def plot_violin(cong, x_='tissue', y_='pearson', hue_='same_group', title_='Cancer Cell Line Congruency', save=False):
    # % matplotlib inline
    # ax = sns.boxplot(x='same_group', y="score", data=cs_1) # plots box and wisker for same/notsame group
    plt.figure(figsize=(40,20))
    ax = sns.violinplot(x=x_, y=y_, hue=hue_, data=cong, palette="Set2", inner='quartile',split=True, linewidth=2)

    for item in ax.get_xticklabels():
        item.set_rotation(90)
        item.set_fontsize(20) 
    for item in ax.get_yticklabels():
        item.set_fontsize(20) 
    plt.ylim(-1, 1)
    plt.xlabel(x_,fontsize=20)
    plt.ylabel(y_,fontsize=20)

    fig = ax.get_figure()
    fig.suptitle(title_, fontsize=30)
    if save:
        fig.savefig("congruent_pearson_GSEA_violin.png")
        
        
# unpacks sets in 'common' column into strings.  multiple entries are duplicated into rows
def unpack_pivot(stackdf):
    col_to_unpack = 'common'
    df = stackdf.copy()
    df = df.loc[df['same_group'] == 1]
    df = pd.DataFrame({col:np.repeat(df[col].values, df[col_to_unpack].str.len()) 
                  for col in df.columns.difference([col_to_unpack])
                 }).assign(**{col_to_unpack:np.concatenate(df[col_to_unpack].map(list).values)})[df.columns.tolist()]

    df = df[['tissue', 'common', 'pearson']]
    df = df.groupby(['tissue','common']).mean().reset_index()
    return df.pivot( 'common', 'tissue','pearson')


def printks(df, iterator, by_col='tissue', score='t-test', group='same_group'):
    lol, l = [], []
    header = ['tissue', 't_Tstat', 't_Pval', 'ks_Dval', 'ks_Pval']
    for tissue in iterator:
    #     tisdf = unif_2[unif_2['tissue'] == tissue]
        diffdf = df[(df[group] == 0)&(df[by_col] == tissue)].dropna()
        samedf = df[(df[group] == 1)&(df[by_col] == tissue)].dropna()
        try:
            tt = stats.ttest_ind(samedf[score],diffdf[score])
            ks = stats.ks_2samp(diffdf[score], samedf[score])
            lol.append([tissue, tt[0], tt[1], ks[0], ks[1]])
        except:
            ValueError
    return pd.DataFrame(lol, columns=header)


def summary_sort(cong, by_col='tissue', score='pearson', group='same_group'):
    iterator = [tissue for tissue in cong[by_col].unique()]
    same_module_df = printks(cong, iterator, by_col, score, group)
    same_module_df.sort_values(by=['ks_Dval'], ascending=False, inplace=True)
    same_module_df.reset_index(drop=True, inplace=True)
    same_module_df.sort_values(by='t_Tstat', ascending=False)
    return same_module_df


def plot_exp_dist(tupe):
    """
    :param tupe: tuple of ('gene_exp', 'gene_cnv'), where exp may depend on cnv context
    """
    exp_gene, cnv_gene = tupe[0], tupe[1]
    sep_con = cdel.loc[cnv_gene]
    pos_context, neg_context = sep_con[sep_con == 1].index, sep_con[sep_con == 0].index

    pos_exp, neg_exp = exp[pos_context].loc[exp_gene], exp[neg_context].loc[exp_gene]

    if len(pos_exp) == 0:
        print("no context for ",cnv_gene,", no missing gene")
    elif len(neg_exp) ==0:
        print("no context for ",cnv_gene,", no present gene")
    else:
        sns.kdeplot(pos_exp, label="- "+cnv_gene, color="r").set_title(exp_gene+" Expression in Context of "+cnv_gene)
        sns.rugplot(pos_exp, color="r")
        sns.kdeplot(neg_exp, label="+ "+cnv_gene, color="b")
        sns.rugplot(neg_exp, color="b")
    print(pos_exp.var(),neg_exp.var())
    
    