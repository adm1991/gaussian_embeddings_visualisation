
%matplotlib inline
from word2gm_loader import Word2GM
from quantitative_eval import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from scipy.stats import norm 
from scipy.spatial.distance import cosine
from scipy.linalg import det



%matplotlib inline
from word2gm_loader import Word2GM
from quantitative_eval import *

model_dir = ### Model Directory here ###


word1 = "apple"
word2 = "banana"
word3 = "computer"


def distributions_kl(m0, S0, m1, S1):

    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def jensen_shannon(m0, S0, m1, S1):
    
    mean_mu = ((m0 + m1) /2)
    mean_sig = ((S0 + S1) / 2)
    
    dist_1 = distributions_kl(m0, S0, mean_mu, mean_sig)
    dist_2 = distributions_kl(m1, S1, mean_mu, mean_sig)
    
    return ((dist_1 + dist_2) / 2)


def generate_visualisations(i, decade, word1, word2, word3):
    
    spherical = False
    diag = False

   
    w2gm_2s = Word2GM(model_dir)
    w2gm_2s.visualize_embeddings()



    ######## First word #################

    index = w2gm_2s.words_to_idxs([word1])
    sig1 = w2gm_2s.logsigs[index][0][0]
    mu1 = w2gm_2s.mus[index][0][0]


    ########## Second Word ###############

    index2 = w2gm_2s.words_to_idxs([word2])
    sig2 = w2gm_2s.logsigs[index2][0][0]
    mu2 = w2gm_2s.mus[index2][0][0]



    ########## Third Word ###############


    index3 = w2gm_2s.words_to_idxs([word3])
    sig3 = w2gm_2s.logsigs[index3][0][0]
    mu3 = w2gm_2s.mus[index3][0][0]




    sig1 = np.abs(sig1)
    sig2 = np.abs(sig2)
    sig3 = np.abs(sig3)


    cov1 = sig1*np.eye(50) 
    cov2 = sig2*np.eye(50)  
    cov3 = sig3*np.eye(50)
    
    av_var = np.log(scipy.linalg.det(cov1))
    

    js1 = jensen_shannon(mu1, np.exp(cov1), mu2, np.exp(cov2))
    js2 = jensen_shannon(mu1, np.exp(cov1), mu3, np.exp(cov3))
    
    cos1 = scipy.spatial.distance.cosine(mu1, mu2)
    cos2 = scipy.spatial.distance.cosine(mu1, mu3)
    
    results_list = [js1, js2, cos1, cos2, av_var]
    
    return results_list
    


font = {'family' : 'normal',
        
        'size'   : 14}


index = [str(i) for i in range(1, 11)] #number of models to be averaged 

decades = [i for i in range(1900, 2010, 10)] #number of decades to be included



js1 = [[] for decade in decades]
js2 = [[] for decade in decades]
cos1 = [[] for decade in decades]
cos2 = [[] for decade in decades]
av_var = [[] for decade in decades]


for i in index:
    for decade in decades:
        resultslist = generate_visualisations(i, decade, word1, word2, word3)
        js1[decades.index(decade)].append(resultslist.pop(0))
        js2[decades.index(decade)].append(resultslist.pop(0))
        cos1[decades.index(decade)].append(resultslist.pop(0))
        cos2[decades.index(decade)].append(resultslist.pop(0))
        av_var[decades.index(decade)].append(resultslist.pop(0))
        
        
decade_labels = [str(int(item) + 10) for item in decades]
    
    
label1 = word2
label2 = word3


dist1 = [sum(items) / len(items) for items in js1]
dist2 = [sum(items) / len(items) for items in js2]

cosine1 = [sum(items) / len(items) for items in cos1]
cosine2 = [sum(items) / len(items) for items in cos2]

log_det = [sum(items) / len(items) for items in av_var]

df_js=pd.DataFrame({'decade': decade_labels, 
                    label1 : dist1, 
                    label2 : dist2})
df_js.set_index('decade', inplace=True, drop=True)

df_cos=pd.DataFrame({'decade': decade_labels, 
                    label1 : cosine1, 
                    label2 : cosine2})
df_cos.set_index('decade', inplace=True, drop=True)

df_var=pd.DataFrame({'decade': decade_labels, 
                    "var_label" : log_det})
df_var.set_index('decade', inplace=True, drop=True)


ax = plt.gca()
df_js.plot(kind='line',y=label1, color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_js.plot(kind='line',y=label2, color='green', marker='o', markerfacecolor='green', markersize=6, ax=ax)
plt.title("Divergence of distributions for \"" + word1 + "\" and control words")
plt.legend()
plt.show()


ax = plt.gca()
df_cos.plot(kind='line',y=label1, color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_cos.plot(kind='line',y=label2, color='green', marker='o', markerfacecolor='green', markersize=6, ax=ax)
plt.title("Distance between means of \"" + word1 + "\" and control words")
plt.legend()
plt.show()

ax = plt.gca()
df_var.plot(kind='line',y="var_label", color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
plt.title("Log determinant for \"" + word1+"\"")
ax.get_legend().remove()
plt.show()


    
    
