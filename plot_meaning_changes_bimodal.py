import numpy as np
from scipy import linalg
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine

from word2gm_loader import Word2GM
from quantitative_eval import *






def distributions_kl(m0, S0, m1, S1):

    # store inverse diagonal covariance of SD and difference between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0


    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) 
    quad_term = diff.T @ np.linalg.inv(S1) @ diff 
    return .5 * (tr_term + det_term + quad_term - N) 

def jensen_shannon(m0, S0, m1, S1):
    
    mean_mu = ((m0 + m1) /2)
    mean_sig = ((S0 + S1) / 2)
    
    dist_1 = distributions_kl(m0, S0, mean_mu, mean_sig)
    dist_2 = distributions_kl(m1, S1, mean_mu, mean_sig)
    
    return ((dist_1 + dist_2) / 2)


def generate_visualisations(decade, i, word1, word2, word3):


    model_dir = '/media/moss/11E17D2A139EBC87/b'+i+'/'+decade+'s_bimodal' # <--CHANGE FOR GENERATE_VISUALISATIONS 
    #AND FINAL_DIFFERENCES
    
    
    w2gm_2s = Word2GM(model_dir)
    w2gm_2s.visualize_embeddings()
    
    print("word1 = ", word1)
    print("word2 = ", word2)
    print("word3 = ", word3)


    ######## First word #################

    index1 = w2gm_2s.words_to_idxs([word1])
    sig1 = w2gm_2s.logsigs[index1][0][0]
    mu1 = w2gm_2s.mus[index1][0][0]
    
    ######## First word - Second sense #################

    index1 = w2gm_2s.words_to_idxs([word1])
    sig2 = w2gm_2s.logsigs[index1][0][1]
    mu2= w2gm_2s.mus[index1][0][1]


    ########## Second Word ###############

    index2 = w2gm_2s.words_to_idxs([word2])
    sig3 = w2gm_2s.logsigs[index2][0][0]
    mu3 = w2gm_2s.mus[index2][0][0]



    ########## Second Word - Second Sense ###############


    index2 = w2gm_2s.words_to_idxs([word2])
    sig4 = w2gm_2s.logsigs[index2][0][1]
    mu4 = w2gm_2s.mus[index2][0][1]
    
    
    
    ########## Third Word ###############

    index3 = w2gm_2s.words_to_idxs([word3])
    sig5 = w2gm_2s.logsigs[index3][0][0]
    mu5 = w2gm_2s.mus[index3][0][0]


    ########## Third word - Second Sense ###############


    index3 = w2gm_2s.words_to_idxs([word3])
    sig6 = w2gm_2s.logsigs[index3][0][1]
    mu6 = w2gm_2s.mus[index3][0][1]



    
    sig1 = np.abs(sig1)
    sig2 = np.abs(sig2)
    sig3 = np.abs(sig3)
    sig4 = np.abs(sig4)
    sig5 = np.abs(sig5)
    sig6 = np.abs(sig6)


    cov1 = sig1*np.eye(50) 
    cov2 = sig2*np.eye(50)  
    cov3 = sig3*np.eye(50)
    cov4 = sig4*np.eye(50)
    cov5 = sig5*np.eye(50)
    cov6 = sig6*np.eye(50)
    

    
    
    print("Cosine Distance for ", word2+"[0]", "and", word3+"[0]", "=",  scipy.spatial.distance.cosine(mu3, mu5))
    print("Cosine Distance for ", word2+"[0]", "and", word3+"[1]", "=",  scipy.spatial.distance.cosine(mu3, mu6))
    
    print("Cosine Distance for ", word2+"[1]", "and", word3+"[0]", "=",  scipy.spatial.distance.cosine(mu4, mu5))
    print("Cosine Distance for ", word2+"[1]", "and", word3+"[1]", "=",  scipy.spatial.distance.cosine(mu4, mu6))
    
    
    
    cosine_dict = {}
    
    cosine_dict[scipy.spatial.distance.cosine(mu3, mu5)] = [0, 0]
    cosine_dict[scipy.spatial.distance.cosine(mu3, mu6)] = [0, 1]
    cosine_dict[scipy.spatial.distance.cosine(mu4, mu5)] = [1, 0]
    cosine_dict[scipy.spatial.distance.cosine(mu4, mu6)] = [1, 1]
    
    max_inds = cosine_dict[max(list(cosine_dict.keys()))]
    
    print(max_inds)
    
    

                                                                                                       
    return max_inds      


    
def final_differences(decade, i, word1, word2, word3, ind1, ind2):
    model_dir = '/media/moss/11E17D2A139EBC87/b'+i+'/'+decade+'s_bimodal' # <--CHANGE FOR GENERATE_VISUALISATIONS 
    #AND FINAL_DIFFERENCES
    
    w2gm_2s = Word2GM(model_dir)
    w2gm_2s.visualize_embeddings()


    ######## First word #################

    index1 = w2gm_2s.words_to_idxs([word1])
    sig1 = w2gm_2s.logsigs[index1][0][0]
    mu1 = w2gm_2s.mus[index1][0][0]

    ######## First word - Second sense #################

    index1 = w2gm_2s.words_to_idxs([word1])
    sig2 = w2gm_2s.logsigs[index1][0][1]
    mu2= w2gm_2s.mus[index1][0][1]


    ########## Second Word ###############

    index2 = w2gm_2s.words_to_idxs([word2])
    sig3 = w2gm_2s.logsigs[index2][0][ind1]
    mu3 = w2gm_2s.mus[index2][0][ind1]


    ########## Third Word ###############

    index3 = w2gm_2s.words_to_idxs([word3])
    sig4 = w2gm_2s.logsigs[index3][0][ind2]
    mu4 = w2gm_2s.mus[index3][0][ind2]
    
    
       
    sense1_word3_cosine = (scipy.spatial.distance.cosine(mu1, mu3))
    sense2_word3_cosine = (scipy.spatial.distance.cosine(mu2, mu3))
    
    calculate_distance_list = [sense1_word3_cosine, sense2_word3_cosine]
    
    if max(calculate_distance_list) == sense2_word3_cosine:
        temp_mu2, temp_sig2 = mu2, sig2
        mu2, sig2 = mu1, sig1
        mu1, sig1 = temp_mu2, temp_sig2

    



    sig1 = np.abs(sig1)
    sig2 = np.abs(sig2)
    sig3 = np.abs(sig3)
    sig4 = np.abs(sig4)



    cov1 = sig1*np.eye(50) 
    cov2 = sig2*np.eye(50)  
    cov3 = sig3*np.eye(50)
    cov4 = sig4*np.eye(50)
    
    log_determinant1 = np.log(scipy.linalg.det(cov1))
    log_determinant2 = np.log(scipy.linalg.det(cov2))


        

    js1 = (jensen_shannon(mu1, np.exp(cov1), mu3, np.exp(cov3)))
    js2 = (jensen_shannon(mu1, np.exp(cov1), mu4, np.exp(cov4)))
    js3 = (jensen_shannon(mu2, np.exp(cov2), mu3, np.exp(cov3)))
    js4 = (jensen_shannon(mu2, np.exp(cov2), mu4, np.exp(cov4)))
    
    cos1 = scipy.spatial.distance.cosine(mu1, mu3)
    cos2 = scipy.spatial.distance.cosine(mu1, mu4)
    cos3 = scipy.spatial.distance.cosine(mu2, mu3)
    cos4 = scipy.spatial.distance.cosine(mu2, mu4)
    

    output_list = [js1, js2, js3, js4, cos1, cos2, cos3, cos4, log_determinant1, log_determinant2]
    print("OUTPUT LIST = ", output_list)
    return output_list
    
    
    

word1 = "cell"          ##both modes represented##
word2 = "tissue" 
word3 = "telephone"


decades = ["1900", "1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000"]



maxind_dict = {}

directory_indexes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]



output = [[] for decade in decades]
sense1_orig = [[] for decade in decades]
sense2_orig = [[] for decade in decades]
sense1_acquired = [[] for decade in decades]
sense2_acquired = [[] for decade in decades]

sense1_orig_cos = [[] for decade in decades]
sense2_orig_cos = [[] for decade in decades]
sense1_acquired_cos = [[] for decade in decades]
sense2_acquired_cos = [[] for decade in decades]

log_determinant1 = [[] for decade in decades]
log_determinant2 = [[] for decade in decades]

for i in directory_indexes:
    for decade in decades:
        print(decade, i)
        maxind_dict[decade] = generate_visualisations(decade, i, word1, word2, word3)







    for decade in maxind_dict:
        print(decade, maxind_dict[decade], i)
        output2 = final_differences(decade, i, word1, word2, word3, maxind_dict[decade][0], maxind_dict[decade][1])
        
        
        output = [items for items in output2]

        index = decades.index(decade)
        print("OUTPUT = ", len(output))

        sense1_orig[index].append(output.pop(0))
        sense1_acquired[index].append(output.pop(0))
        sense2_orig[index].append(output.pop(0))
        sense2_acquired[index].append(output.pop(0))


        sense1_orig_cos[index].append(output.pop(0))
        sense1_acquired_cos[index].append(output.pop(0))
        sense2_orig_cos[index].append(output.pop(0))
        sense2_acquired_cos[index].append(output.pop(0))


        log_determinant1[index].append(output.pop(0))
        log_determinant2[index].append(output.pop(0))

        


label1 = word1+'[0] '+word2
label2 = word1+'[0] '+word3
label3 = word1+'[1] '+word2
label4 = word1+'[1] '+word3




label5 = word1+'[0]'
label6 = word1+'[1]'

print(decades)
decades = [str(int(items) + 10) for items in decades]



sense1_orig_mean = [sum(item) / len(item) for item in sense1_orig]
sense1_acquired_mean = [sum(item) / len(item) for item in sense1_acquired]
sense2_orig_mean = [sum(item) / len(item) for item in sense2_orig]
sense2_acquired_mean = [sum(item) / len(item) for item in sense2_acquired]
   
sense1_orig_cos_mean = [sum(item) / len(item) for item in sense1_orig_cos]
sense1_acquired_cos_mean = [sum(item) / len(item) for item in sense1_acquired_cos]
sense2_orig_cos_mean = [sum(item) / len(item) for item in sense2_orig_cos]
sense2_acquired_cos_mean = [sum(item) / len(item) for item in sense2_acquired_cos]
    
log_determinant1_mean = [sum(item) / len(item) for item in log_determinant1]
log_determinant2_mean = [sum(item) / len(item) for item in log_determinant2]

df_js=pd.DataFrame({'decade': decades, 
                    label1 : sense1_orig_mean, 
                    label2 : sense1_acquired_mean,
                    label3 : sense2_orig_mean, 
                    label4 :sense2_acquired_mean})
df_js.set_index('decade', inplace=True, drop=True)



df_cd=pd.DataFrame({'decade': decades, 
                    label1 : sense1_orig_cos_mean,
                    label2 : sense1_acquired_cos_mean,
                    label3 : sense2_orig_cos_mean, 
                    label4 :sense2_acquired_cos_mean})
df_cd.set_index('decade', inplace=True, drop=True)

df_ma=pd.DataFrame({'decade': decades, 
                    label5 : log_determinant1_mean, 
                    label6 : log_determinant2_mean })
df_ma.set_index('decade', inplace=True, drop=True)




ax = plt.gca()
df_js.plot(kind='line',y=label1, color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_js.plot(kind='line',y=label2, color='lightblue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_js.plot(kind='line',y=label3, color='green', marker='o', markerfacecolor='green', markersize=6, ax=ax)
df_js.plot(kind='line',y=label4, color='lightgreen', marker='o', markerfacecolor='green', markersize=6, ax=ax)
plt.title("Divergence of distributions for \"" + word1 + "\" and control words")
plt.legend()
plt.show()



ax = plt.gca()
df_cd.plot(kind='line',y=label1, color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_cd.plot(kind='line',y=label2, color='lightblue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_cd.plot(kind='line',y=label3, color='green', marker='o', markerfacecolor='green', markersize=6, ax=ax)
df_cd.plot(kind='line',y=label4, color='lightgreen', marker='o', markerfacecolor='green', markersize=6, ax=ax)
plt.title("Distance between means of \"" + word1 + "\" and control words")
plt.legend()
plt.show()


ax = plt.gca()
df_ma.plot(kind='line',y=label5, color='blue', marker='o', markerfacecolor='blue', markersize=6, ax=ax)
df_ma.plot(kind='line',y=label6, color='green', marker='o', markerfacecolor='green', markersize=6, ax=ax)
plt.title("Log determinant for \"" + word1+"\"")
plt.legend()
plt.show()

                                            
                                                                                                       
                                                                                                       
                                                                                                   
                                                                                                   
                                                                                                       
    
    
  

