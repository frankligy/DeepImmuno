'''
this script is for machine learning algorithm, eapecially
random forest, ada boost
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve,roc_curve,auc,confusion_matrix
import matplotlib.pyplot as plt


def aaindex(peptide,after_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)   # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded

def aaindex(peptide,after_pca):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    encoded = np.empty([len(peptide),21])
    onehot = np.identity(21)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i,:] = onehot[:,amino.index(query)]
    return encoded

def peptide_data_aaindex(peptide,after_pca):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide,after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode


def dict_inventory(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4]  # A,B,C
        first2 = hla[6:8]  # 01
        last2 = hla[8:]  # 01
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)

    return dic


def rescue_unknown_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    #print(hla)
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])






def hla_data_aaindex(hla_dic,hla_type,after_pca):    # return numpy array [34,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type,dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode

def construct_aaindex(ori,hla_dic,after_pca):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]
        '''
        If 'classfication': ['immunogenicity']
        If 'regression': ['potential']
        '''

        encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]

        encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca)   # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic


def draw_ROC(y_true,y_pred):

    fpr,tpr,_ = roc_curve(y_true,y_pred,pos_label=1)
    area_mine = auc(fpr,tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % area_mine)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def draw_PR(y_true,y_pred):
    precision,recall,_ = precision_recall_curve(y_true,y_pred,pos_label=1)
    area_PR = auc(recall,precision)
    baseline = np.sum(np.array(y_true) == 1) / len(y_true)

    plt.figure()
    lw = 2
    plt.plot(recall,precision, color='darkorange',
            lw=lw, label='PR curve (area = %0.2f)' % area_PR)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="lower right")
    plt.show()

def get_Y(value):
    result = []
    for i in value:
        if i == 'Negative':
            result.append(0)
        else:
            result.append(1)
    return result

def retain_910(ori):
    cond = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        if len(peptide) == 9 or len(peptide) == 10:
            cond.append(True)
        else:
            cond.append(False)
    data = ori.loc[cond]
    data = data.set_index(pd.Index(np.arange(data.shape[0])))
    return data

if __name__ == '__main__':
    # let's first prepare the X, Y
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    ori['immunogenicity'],ori['potential'] = ori['potential'],ori['immunogenicity']
    dataset = construct_aaindex(ori, hla_dic, after_pca)
    X = np.empty((len(dataset), 12 * 56))  # 28581
    Y = ori['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)  # 10*12*1 ---> 120
        y = y.reshape(-1)  # 46*12*1 ---> 552
        X[i, :] = np.concatenate([x, y])  #

    # also, let's define a scorer
    from sklearn.metrics import mean_squared_error,make_scorer
    rmse = make_scorer(mean_squared_error,squared=False)

    # also, let's wrap an evaluation function
    def evaluate(estimator,test_X,test_Y):
        pred = estimator.predict(test_X)
        result = mean_squared_error(test_Y,pred,squared=False)
        return result

    # holder
    holder = {}


    # then, for each of "elasticNet,knn, svr, random forest, ada boost", first tune the hyperparameter, then test for each fold
    # go to each subsection below now







    # how about ada boost regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import AdaBoostRegressor
    space = np.linspace(20, 140, 4)
    for i in space:
        cv_result = cross_validate(AdaBoostRegressor(n_estimators=int(i)), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = AdaBoostRegressor(n_estimators=60)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['adaboost'] = holding





    # have a look at all regression machine learning algoritm, using cross-validation to tune the parameter, test on testing dataset:
    '''
    linear regression
    ridge regression
    lasso regression
    elastic net regression
    SVR
    KNN regressor
    random forest 
    '''




    # elastic net regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import ElasticNet
    space = np.linspace(0.01,1,5)
    for i in space:
        cv_result = cross_validate(ElasticNet(alpha=0.01,l1_ratio=i),X,Y,cv=5,scoring=rmse,n_jobs=-1,verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1,1,1)
    ax1.plot(np.arange(len(space)),y1,marker='o',markersize=5)
    ax1.fill_between(np.arange(len(space)),[y1[i]-y1_e[i] for i in range(len(space))],[y1[i]+y1_e[i] for i in range(len(space))],alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = ElasticNet(alpha=0.01, l1_ratio=0.51)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}
    for fold in fold_indices:
        # split
        train_X,train_Y,test_X,test_Y = X[fold[0],:],np.array(Y)[fold[0]],X[fold[1],:],np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator,test_X,test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth,hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth,hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20]==1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50]==1)  # top50
        holding['cell'].append((result1,result2,result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'],hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'],hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'],hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'],hard)  # unexposed recall
        holding['covid'].append((result1,result2,result3,result4))
    holder['elasticnet'] = holding





    # KNN regressor
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.neighbors import KNeighborsRegressor
    space = np.linspace(1,100,10)
    for i in space:
        cv_result = cross_validate(KNeighborsRegressor(n_neighbors=int(i)),X,Y,cv=5,scoring=rmse,n_jobs=-1,verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1,1,1)
    ax1.plot(np.arange(len(space)),y1,marker='o',markersize=5)
    ax1.fill_between(np.arange(len(space)),[y1[i]-y1_e[i] for i in range(len(space))],[y1[i]+y1_e[i] for i in range(len(space))],alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = KNeighborsRegressor(n_neighbors=23)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}
    for fold in fold_indices:
        # split
        train_X,train_Y,test_X,test_Y = X[fold[0],:],np.array(Y)[fold[0]],X[fold[1],:],np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator,test_X,test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth,hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth,hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20]==1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50]==1)  # top50
        holding['cell'].append((result1,result2,result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'],hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'],hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'],hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'],hard)  # unexposed recall
        holding['covid'].append((result1,result2,result3,result4))
    holder['KNN'] = holding


    # random forest regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestRegressor
    space = np.linspace(1, 9, 3)
    for i in space:
        cv_result = cross_validate(RandomForestRegressor(n_estimators=200,min_samples_leaf=int(i)), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = RandomForestRegressor(n_estimators=200,min_samples_leaf=1)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['randomforest'] = holding

    # SVR
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.svm import LinearSVR
    space = np.logspace(-3, 3, 7)
    for i in space:
        cv_result = cross_validate(LinearSVR(C=i), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = LinearSVR(C=0.01)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['SVR'] = holding



    '''
    
    What if we use one-hot encoding
    
    '''

    # let's first prepare the X, Y
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']
    dataset = construct_aaindex(ori, hla_dic, after_pca)
    X = np.empty((len(dataset), 21 * 56))  # 28581
    Y = ori['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)  # 10*12*1 ---> 120
        y = y.reshape(-1)  # 46*12*1 ---> 552
        X[i, :] = np.concatenate([x, y])  #

    # also, let's define a scorer
    from sklearn.metrics import mean_squared_error, make_scorer

    rmse = make_scorer(mean_squared_error, squared=False)


    # also, let's wrap an evaluation function
    def evaluate(estimator, test_X, test_Y):
        pred = estimator.predict(test_X)
        result = mean_squared_error(test_Y, pred, squared=False)
        return result


    # holder
    holder = {}

    # then, for each of "elasticNet,knn, svr, random forest, ada boost", first tune the hyperparameter, then test for each fold
    # go to each subsection below now

    # how about ada boost regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import AdaBoostRegressor

    space = np.linspace(20, 140, 4)
    for i in space:
        cv_result = cross_validate(AdaBoostRegressor(n_estimators=int(i)), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = AdaBoostRegressor(n_estimators=60)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['adaboost'] = holding


    # elastic net regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import ElasticNet

    space = np.linspace(0.01, 1, 5)
    for i in space:
        cv_result = cross_validate(ElasticNet(alpha=0.01, l1_ratio=i), X, Y, cv=5, scoring=rmse, n_jobs=-1, verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = ElasticNet(alpha=0.01, l1_ratio=0.51)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['elasticnet'] = holding

    # KNN regressor
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.neighbors import KNeighborsRegressor

    space = np.linspace(1, 100, 10)
    for i in space:
        cv_result = cross_validate(KNeighborsRegressor(n_neighbors=int(i)), X, Y, cv=5, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = KNeighborsRegressor(n_neighbors=23)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['KNN'] = holding

    # random forest regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestRegressor

    space = np.linspace(1, 9, 3)
    for i in space:
        cv_result = cross_validate(RandomForestRegressor(n_estimators=200, min_samples_leaf=int(i)), X, Y, cv=3,
                                   scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=1)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['randomforest'] = holding

    # SVR
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.svm import LinearSVR

    space = np.logspace(-3, 3, 7)
    for i in space:
        cv_result = cross_validate(LinearSVR(C=i), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = LinearSVR(C=0.01)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 21 * 56))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['SVR'] = holding

    '''
     
     keep ablation test, let's see aaindex + pseudo34
     
     '''


    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    hla = pd.read_csv('immuno2/data/pseudo34_clean.txt', sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)
    ori['immunogenicity'], ori['potential'] = ori['potential'], ori['immunogenicity']
    dataset = construct_aaindex(ori, hla_dic, after_pca)
    X = np.empty((len(dataset), 12 * 44))  # 28581
    Y = ori['immunogenicity'].values
    for i, (x, y, _) in enumerate(dataset):
        x = x.reshape(-1)  # 10*12*1 ---> 120
        y = y.reshape(-1)  # 46*12*1 ---> 552
        X[i, :] = np.concatenate([x, y])  #

    # also, let's define a scorer
    from sklearn.metrics import mean_squared_error, make_scorer

    rmse = make_scorer(mean_squared_error, squared=False)


    # also, let's wrap an evaluation function
    def evaluate(estimator, test_X, test_Y):
        pred = estimator.predict(test_X)
        result = mean_squared_error(test_Y, pred, squared=False)
        return result


    # holder
    holder = {}

    # then, for each of "elasticNet,knn, svr, random forest, ada boost", first tune the hyperparameter, then test for each fold
    # go to each subsection below now

    # how about ada boost regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import AdaBoostRegressor

    space = np.linspace(20, 140, 4)
    for i in space:
        cv_result = cross_validate(AdaBoostRegressor(n_estimators=int(i)), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = AdaBoostRegressor(n_estimators=60)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['adaboost'] = holding


    # elastic net regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import ElasticNet

    space = np.linspace(0.01, 1, 5)
    for i in space:
        cv_result = cross_validate(ElasticNet(alpha=0.01, l1_ratio=i), X, Y, cv=5, scoring=rmse, n_jobs=-1, verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = ElasticNet(alpha=0.01, l1_ratio=0.51)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['elasticnet'] = holding

    # KNN regressor
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.neighbors import KNeighborsRegressor

    space = np.linspace(1, 100, 10)
    for i in space:
        cv_result = cross_validate(KNeighborsRegressor(n_neighbors=int(i)), X, Y, cv=5, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = KNeighborsRegressor(n_neighbors=23)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['KNN'] = holding

    # random forest regression
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.ensemble import RandomForestRegressor

    space = np.linspace(1, 9, 3)
    for i in space:
        cv_result = cross_validate(RandomForestRegressor(n_estimators=200, min_samples_leaf=int(i)), X, Y, cv=3,
                                   scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = RandomForestRegressor(n_estimators=200, min_samples_leaf=1)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['randomforest'] = holding

    # SVR
    cv_results = []
    from sklearn.model_selection import cross_validate
    from sklearn.svm import LinearSVR

    space = np.logspace(-3, 3, 7)
    for i in space:
        cv_result = cross_validate(LinearSVR(C=i), X, Y, cv=3, scoring=rmse, n_jobs=-1,
                                   verbose=5)
        cv_results.append(cv_result)
    y1 = [item['test_score'].mean() for item in cv_results]
    y1_e = [item['test_score'].std() for item in cv_results]
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(np.arange(len(space)), y1, marker='o', markersize=5)
    ax1.fill_between(np.arange(len(space)), [y1[i] - y1_e[i] for i in range(len(space))],
                     [y1[i] + y1_e[i] for i in range(len(space))], alpha=0.2)
    ax1.set_xticks(np.arange(len(space)))
    ax1.set_xticklabels(['{0:.2f}'.format(i) for i in space])

    estimator = LinearSVR(C=0.01)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(X.shape[0])))
    holding = {'validation': [], 'dengue': [], 'cell': [], 'covid': []}
    for fold in fold_indices:
        # split
        train_X, train_Y, test_X, test_Y = X[fold[0], :], np.array(Y)[fold[0]], X[fold[1], :], np.array(Y)[fold[1]]
        # train
        estimator.fit(train_X, train_Y)
        # test in validation set
        result_validation = evaluate(estimator, test_X, test_Y)
        holding['validation'].append(result_validation)
        # test in dengue
        ori_test_dengue = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        testing_dataset = construct_aaindex(ori_test_dengue, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_dengue['immunogenicity'].values
        from sklearn.metrics import accuracy_score, recall_score, precision_score

        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result = accuracy_score(truth, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        testing_dataset = construct_aaindex(ori_test_cell, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        truth = ori_test_cell['immunogenicity'].values
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(truth, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1, result2, result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        testing_dataset = construct_aaindex(ori_test_covid, hla_dic, after_pca)
        X_test = np.empty((len(testing_dataset), 12 * 44))
        for i, (x, y, _) in enumerate(testing_dataset):
            x = x.reshape(-1)
            y = y.reshape(-1)
            X_test[i, :] = np.concatenate([x, y])
        prediction = estimator.predict(X_test)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
    holder['SVR'] = holding










































