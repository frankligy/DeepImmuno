'''
Using hidden Markov model to predict (peptide-TCR) binding affinity

Notation:
n_samples: the length of a sequence
n_components: the length of state space Z {sunny, rainy,cloudy}
n_features: the length of observation space X {go out, not go out}

three basic problems:
1. scoring problem. Having model, given X(n_samples,), what's the observation probability(model fit X, X meet the underlying structure
of HMM model) of X. Using dynamic programming forward algorithm. Calling score function, score_samples can also give you
the posterior distribution of Z(n_samples, n_components).

2. optimal path. Having model, given X, what's the optimal path, meaning the state sequence(n_samples,), solve via dynamic
programming using Viterbi algorithm. Calling decode function will give you the state sequence, predict_proba will give your
posterior distribution(n_samples,n_components). Or score_samples function as mentioned above.

3. training problem. No model, given X, training the model, solve by EM algorithm(Baum-welch algorithm). Calling fit method.
the four internal parameters will be trained and updated during training:
(1) startprob_   's'
(2) transmat_    't'
(3) means_       'm'
(4) covars_      'c'

means_ and covars_ are to define Gaussian kernel.

After training, you can run score function to assess how good the training phase is.

API: https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn-hmm

'''


'''
Come down to the real question, let's first take one specific epitope at a time, collect all positive TCR alpha and beta
CDR3 sequence seperately. Using HMM to train and see if it will perform well in hold-out dataset.
'''

import numpy as np
from hmmlearn import hmm
import pandas as pd

def select_specie(full,specie):
    cond = []
    for i in range(full.shape[0]):
        col = full['species'].iloc[i]
        if col == specie:
            cond.append(True)
        else:
            cond.append(False)
    data = full.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

def select_epitope(full,epitope):
    cond = []
    for i in range(full.shape[0]):
        col = full['antigen.epitope'].iloc[i]
        if col == epitope:
            cond.append(True)
        else:
            cond.append(False)
    data = full.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

def select_positive(full):
    cond = []
    for i in range(full.shape[0]):
        col = full['vdjdb.score'].iloc[i]
        if col != 0:
            cond.append(True)
        else:
            cond.append(False)
    data = full.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

def select_TCR(full,ab):
    import math
    if ab=='alpha':
        slim = full.drop(columns=['cdr3.beta','v.beta','d.beta','j.beta','cdr3fix.beta'])
        cond = []
        for i in range(slim.shape[0]):
            col = slim['cdr3.alpha'].iloc[i]
            try:
                math.isnan(col)
                cond.append(False)
            except TypeError:
                cond.append(True)
        data = slim.loc[cond]
        data = data.set_index(pd.Index(np.arange(data.shape[0])))
    elif ab == 'beta':
        slim = full.drop(columns=['cdr3.alpha', 'v.alpha', 'j.alpha', 'cdr3fix.alpha'])
        cond = []
        for i in range(slim.shape[0]):
            col = slim['cdr3.beta'].iloc[i]
            try:
                math.isnan(col)
                cond.append(False)
            except TypeError:
                cond.append(True)
        data = slim.loc[cond]
        data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

def string2matrix_onehot(string):
    amino = 'ARNDCQEGHILKMFPSTWYV'
    mat = np.empty([len(string),20])
    eye_mat = np.identity(20)
    for i in range(len(string)):
        mat[i,:] = eye_mat[amino.index(string[i]),:]
    return mat

def string2matrix_plain(string):
    amino = 'ARNDCQEGHILKMFPSTWYV'
    mat = np.empty([len(string),1])
    for i in range(len(string)):
        mat[i,:] = amino.index(string[i])
    return mat

def construct_X_length(cdr3):
    string = ''
    length_array = []
    for item in cdr3:
        length = len(item)
        length_array.append(length)
        string += item
    X = string2matrix_plain(string).astype(np.int)
    return X,length_array



if __name__ == '__main__':
    full = pd.read_csv('/Users/ligk2e/Desktop/immuno3/vdjdb/vdjdb_full.txt',sep='\t')
    full_human = select_specie(full,'HomoSapiens')
    full_human_positive = select_positive(full_human)
    full_human_alpha = select_TCR(full_human,'alpha')
    full_human_positive_beta = select_TCR(full_human_positive,'beta')

    # let's first try a certain epitope's all TCR alpha
    full_human_positive_alpha.to_csv('/Users/ligk2e/Desktop/immuno3/vdjdb/process/full_human_positive_alpha.csv',index=None)
    # let's try FRDYVDRFYKTLRAEQASQE
    data = select_epitope(full_human_alpha,'GILGFVFTL')
    data = select_positive(data)
    cdr3 = data['cdr3.alpha']
    cdr3_train_index = np.random.choice(cdr3.index.values,size=250,replace=False)
    cdr3_test_index = [i for i in cdr3.index if i not in cdr3_train_index]
    cdr3_train = cdr3.iloc[cdr3_train_index]
    cdr3_test = cdr3.iloc[cdr3_test_index]
    X,length = construct_X_length(cdr3_train)
    # let's build the model
    from hmmlearn.hmm import MultinomialHMM
    model = MultinomialHMM(n_components=3,random_state=42,params='e',init_params='e')
    model.startprob_ = [0.16,0.04,0.8]
    model.transmat_ = [[0.67,0.13,0.2],[0,0.5,0.5],[0,0,1]]
    '''
    startprob_:
    V: 
    D:
    J:
    
    trasmat_:
    from each row, row become column (the probability of row become column, each row sum to 1, not column)   
    '''
    model.fit(X,length)
    emission = model.emissionprob_
    model.n_features
    model.transmat_
    model.startprob_
    model.get_stationary_distribution()
    # let's test
    train_score = []
    for i in cdr3_train_index:
        test = string2matrix_plain(cdr3[i]).astype(np.int)
        score = model.score(test)
        train_score.append(score)
    train_score = np.array(train_score)

    test_score = []
    for i in cdr3_test_index:
        test = string2matrix_plain(cdr3[i]).astype(np.int)
        score = model.score(test)
        test_score.append(score)
    test_score = np.array(test_score)

    negative_score = []
    negative = []
    for i in range(data.shape[0]):
        if data['vdjdb.score'].iloc[i] == 0:
            negative.append(data['cdr3.alpha'].iloc[i])
    for i in range(len(negative)):
        test = string2matrix_plain(negative[i]).astype(np.int)
        score = model.score(test)
        negative_score.append(score)
    negative_score = np.array(negative_score)

    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    ax.boxplot([train_score,test_score,negative_score],positions=[1,2,3],labels=['training','testing_pos','testing_neg'])
    x_train = np.random.normal(1,0.02,size=len(train_score))
    ax.plot(x_train,train_score,'r.',alpha=0.2)
    x_test = np.random.normal(2,0.02,size=len(test_score))
    ax.plot(x_test,test_score,'b.',alpha=0.2)
    x_negative = np.random.normal(3,0.02,size=len(negative_score))
    ax.plot(x_negative,negative_score,'k.',alpha=0.2)
    ax.set_title('Observation score from HMM -- "GILGFVFTL"--TCRA')

    import scipy.stats as sc
    sc.ttest_ind(test_score,negative_score)






    model.predict(test1)
    model.predict_proba(test1)
    model.decode(test1)





















