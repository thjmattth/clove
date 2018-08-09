
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
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

def mainFitler(expdf_fh, cnvdf_fh='CCLE_DEL_calls.pickle', var=0.2, n=5, amp_fh=False, dele_fh=False, mut_fh=False, save=False):
    """
    returns expdf and cnvdf filtered on power and abberations, used for CCLE data mostly
    
    :param expdf_fn: str, filename of pickled expression dataframe, (check in /clove/tissueDf_raw)
    :param cnvdf_fn: str, filename revealer expression calls, (CCLE_DEL_calls.pickle)
    :param var: float, minimum gene variance to filter low-variance pairs
    :param amp_fh: str, filename of pickled dataframe of binarized ampflication calls (eg REVEALER)
    :param dele_fh: str, filename of pickled dataframe of binarized deletion calls (eg REVEALER)
    :param mut_fh: str, filename of pickled dataframe of binarized mutation calls (eg REVEALER)
    :param save: bool, True saves the filtered dfs with filter conditions in filename
    
    :returns: filtered expdf, filtered cnvdf
    """
    # filter on power
    expdf = pd.read_pickle(expdf_fh)
    cnvdf = pd.read_pickle(cnvdf_fh)
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
    
    # right = expdf.reset_index()
    # right['gene_var_exp'] = right.var(axis=1)
    # if 'index' in right.columns:
    #     right.rename(columns={'index':'exp'}, inplace=True)
    # elif 'gene_id' in right.columns:
    #     right.rename(columns={'gene_id':'exp'}, inplace=True)
    
    right = expdf.rename_axis('exp', axis=0) 
    right['gene_var_exp'] = right.var(axis=1)
    right = right.reset_index()
    
    df = pd.merge(df, right[['exp','gene_var_exp']], on='exp')
    
    if cat_df:
        return pd.concat([cat_df, df])
        
    return df


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

def TESTrandomPairContextStat(n_samp, expdf, cnvdf, nan_style='omit', shrink=True, permute=False):
    """
    takes exp and cnv genes and returns pair summary statistics
    
    :param n_samp: int, number of random samples to take
    :param expdf: pandas dataframe, expression by sample (hopefully filtered and tissue specific)
    :param cnvdf: pandas dataframe, binarized (1=del, 0=nodel) deletion by sample
    :param nan_style: str, how the stats.ttest_ind treats NANs, {‘propagate’, ‘raise’, ‘omit’}
    :param shrink: bool, True will enable mean of gene expression variance to be added as shrinkage factor
    :param permute: bool, True will calculate pairs with randomly permuted expression matrix as null model
    
    returns df[['exp', 'cnv', 'cntxt_pos_mu', 'cntxt_neg_mu', 
                'cntxt_pos_var', 'cntxt_neg_var', 
                'cntxt_pos_n', 'cntxt_neg_n']]
    """
    exp_samp = expdf.sample(n=n_samp).index.values
    cnv_samp = cnvdf.sample(n=n_samp).index.values
    cells = list(set(cdel.columns).intersection(expdf.columns))
    cmask = cnvdf[cells] == 1
    
    def getContexts(exp_g, cnv_g):
        lossTrue = cmask.loc[cnv_g]
        print(expdf.loc[exp_g].where(lossTrue, np.nan))
        loss = np.array(expdf.loc[exp_g].where(lossTrue, np.nan).dropna(), dtype=object)
        no_loss = np.array(expdf.loc[exp_g].where(~lossTrue, np.nan).dropna())
        return loss, no_loss
    
    if permute:
        cmask = scrambleDF(cmask)

    df = pd.DataFrame(pd.Series(exp_samp), columns=['exp'])
    df['cnv'] = pd.Series(cnv_samp)
    
    df['loss'], df['no_loss'] = np.vectorize(getContexts)(df['exp'], df['cnv'])
    
    df['loss_n'] = np.vectorize(getLen)(df['loss'])
    df['no_loss_n'] = np.vectorize(getLen)(df['no_loss'])
    
    df['loss_mu'] = np.vectorize(getMean)(df['loss'])
    df['no_loss_mu'] = np.vectorize(getMean)(df['no_loss'])
    
    df['loss_var'] = np.vectorize(getVar)(df['loss'])
    df['no_loss_var'] = np.vectorize(getVar)(df['no_loss'])
    
    df['cohens_d'] = np.vectorize(cohenD)(df['loss'], df['no_loss'])
    
    if shrinkage_factor:
        shrinkage_factor = meanVar(expdf)
    else:
        shrinkage_factor = 0
    df['t_test'] = np.vectorize(t_welch)(df['loss_n'], df['no_loss_n'],
                                         df['loss_mu'], df['no_loss_mu'],
                                         df['loss_var'], df['no_loss_var'],
                                        shrinkage_factor)
    return df
