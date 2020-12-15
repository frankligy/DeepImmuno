'''
Immuno project using GCN model, in immuno2 folder
'''

import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection
from sklearn.metrics import confusion_matrix,auc,precision_recall_curve,roc_curve
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from stellargraph import IndexedArray

import itertools
import matplotlib.pyplot as plt


def read_para(path):
    df = pd.read_csv(path,sep='\t',header=None)
    dic = {}
    for i in range(df.shape[0]):
        hla = df[0].iloc[i]
        paratope = df[1].iloc[i]
        try:
            dic[hla] = paratope
        except KeyError:
            dic[hla] = []
            dic[hla].append(paratope)
    return dic

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

class Graph_Constructor():

    @staticmethod
    def combinator(pep,hla):
        source = ['p' + str(i+1) for i in range(len(pep))]
        target = ['h' + str(i+1) for i in range(len(hla))]
        return source,target

    @staticmethod
    def numerical(pep,hla,after_pca,embed=12):   # after_pca [21,12]
        pep = pep.replace('X','-').upper()
        hla = hla.replace('X','-').upper()
        feature_array_pep = np.empty([len(pep),embed])
        feature_array_hla = np.empty([len(hla),embed])
        amino = 'ARNDCQEGHILKMFPSTWYV-'
        for i in range(len(pep)):
            feature_array_pep[i,:] = after_pca[amino.index(pep[i]),:]
        for i in range(len(hla)):
            feature_array_hla[i,:] = after_pca[amino.index(hla[i]),:]
        feature_array = np.concatenate([feature_array_pep,feature_array_hla],axis=0)
        #print(feature_array_pep.shape,feature_array_hla.shape,feature_array.shape)
        return feature_array

    @staticmethod
    def unweight_edge(pep,hla,after_pca):
        source,target = Graph_Constructor.combinator(pep,hla)
        combine = list(itertools.product(source,target))
        weight = itertools.repeat(1,len(source)*len(target))
        edges = pd.DataFrame({'source':[item[0] for item in combine],'target':[item[1] for item in combine],'weight':weight})
        feature_array = Graph_Constructor.numerical(pep,hla,after_pca)
        try:nodes = IndexedArray(feature_array,index=source+target)
        except: print(pep,hla,feature_array.shape)
        graph = StellarGraph(nodes,edges,node_type_default='corner',edge_type_default='line')
        return graph

    @staticmethod
    def weight_anchor_edge(pep,hla,after_pca):
        source, target = Graph_Constructor.combinator(pep, hla)
        combine = list(itertools.product(source, target))
        weight = itertools.repeat(1, len(source) * len(target))
        edges = pd.DataFrame({'source': [item[0] for item in combine], 'target': [item[1] for item in combine], 'weight': weight})
        for i in range(edges.shape[0]):
            col1 = edges.iloc[i]['source']
            col2 = edges.iloc[i]['target']
            col3 = edges.iloc[i]['weight']
            if col1 == 'a2' or col1 == 'a9' or col1 ==  'a10':
                edges.iloc[i]['weight'] = 1.5
        feature_array = Graph_Constructor.numerical(pep, hla, after_pca)
        nodes = IndexedArray(feature_array, index=source + target)
        graph = StellarGraph(nodes, edges, node_type_default='corner', edge_type_default='line')
        return graph

    @staticmethod
    def intra_and_inter(pep,hla,after_pca):
        source, target = Graph_Constructor.combinator(pep, hla)
        combine = list(itertools.product(source, target))
        weight = itertools.repeat(2, len(source) * len(target))
        edges_inter = pd.DataFrame({'source': [item[0] for item in combine], 'target': [item[1] for item in combine], 'weight': weight})
        intra_pep = list(itertools.combinations(source,2))
        intra_hla = list(itertools.combinations(target,2))
        intra = intra_pep + intra_hla
        weight = itertools.repeat(1,len(intra))
        edges_intra = pd.DataFrame({'source':[item[0] for item in intra],'target':[item[1] for item in intra],'weight':weight})
        edges = pd.concat([edges_inter,edges_intra])
        edges = edges.set_index(pd.Index(np.arange(edges.shape[0])))
        feature_array = Graph_Constructor.numerical(pep, hla, after_pca)
        nodes = IndexedArray(feature_array, index=source + target)
        graph = StellarGraph(nodes, edges, node_type_default='corner', edge_type_default='line')
        return graph

    @staticmethod
    def entrance(df,after_pca,hla_dic,dic_inventory):
        graphs = []
        graph_labels = []
        for i in range(df.shape[0]):
            print(i)
            pep = df['peptide'].iloc[i]
            try:
                hla = hla_dic[df['HLA'].iloc[i]]
            except KeyError:
                hla = hla_dic[rescue_unknown_hla(df['HLA'].iloc[i],dic_inventory)]
            label = df['immunogenicity'].iloc[i]
            #if label != 'Negative': label = 0
            #else: label = 1
            #graph = Graph_Constructor.unweight_edge(pep,hla,after_pca)
            #graph = Graph_Constructor.unweight_edge(pep,hla,after_pca)
            graph = Graph_Constructor.intra_and_inter(pep,hla,after_pca)
            graphs.append(graph)
            graph_labels.append(label)
        graph_labels = pd.Series(graph_labels)
        return graphs,graph_labels


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=2, callbacks=[es],)
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]
    return history, test_acc

def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

    return train_gen, test_gen


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

def draw_history(history):
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.show()

def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

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


    ori_train = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/remove0123_sample100.csv')
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    ori_train['immunogenicity'], ori_train['potential'] = ori_train['potential'], ori_train['immunogenicity']





    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    fold_indices = list(kf.split(np.arange(ori_train.shape[0])))
    holding = {'validation':[],'dengue':[],'cell':[],'covid':[]}



    for fold in fold_indices:
        i = 1
        graphs, graph_labels = Graph_Constructor.entrance(ori_train, after_pca, hla_dic, dic_inventory)
        generator = PaddedGraphGenerator(graphs=graphs)

        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=[64, 64],
            activations=["relu", "relu"],
            generator=generator,
            dropout=0.2, )

        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=16, activation="relu")(predictions)
        predictions = Dense(units=1, activation="sigmoid")(predictions)
        model = Model(inputs=x_inp, outputs=predictions)

        model.compile(optimizer=Adam(0.001), loss=mean_squared_error)
        train_gen = generator.flow(
            fold[0],
            targets=graph_labels.iloc[fold[0]].values,
            batch_size=256,)
        test_gen = generator.flow(
            fold[1],
            targets=graph_labels.iloc[fold[1]].values,
            batch_size=1,)
        epochs = 100
        es1 = EarlyStopping(monitor='loss',patience=2,restore_best_weights=False)
        es2 = EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=False)
        history = model.fit(
            train_gen, epochs=epochs, validation_data=test_gen, shuffle=True,callbacks=[es1,es2,],class_weight={0:0.5,1:0.5})
        # test in validation
        pred = model.predict(test_gen)
        from sklearn.metrics import mean_squared_error
        result = mean_squared_error(graph_labels.iloc[fold_indices[0][1]],pred,squared=False)
        holding['validation'].append(result)
        # test in dengue
        ori_test = pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/dengue_test.csv')
        graphs_test, graph_labels_test = Graph_Constructor.entrance(ori_test, after_pca, hla_dic, dic_inventory)
        generator_test = PaddedGraphGenerator(graphs=graphs_test)
        input = generator_test.flow(graphs_test)
        prediction = model.predict(input)
        from sklearn.metrics import accuracy_score,recall_score,precision_score
        hard = [1 if item >= 0.5 else 0 for item in prediction[:,0]]
        result = accuracy_score(graph_labels_test, hard)
        holding['dengue'].append(result)
        # test in cell
        ori_test_cell = pd.read_csv(
            '/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt', sep='\t')
        graphs_test, graph_labels_test = Graph_Constructor.entrance(ori_test_cell, after_pca, hla_dic, dic_inventory)
        generator_test = PaddedGraphGenerator(graphs=graphs_test)
        input = generator_test.flow(graphs_test)
        prediction = model.predict(input)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(graph_labels_test, hard)  # recall
        ori_test_cell['result'] = prediction
        ori_test_cell = ori_test_cell.sort_values(by='result', ascending=False).set_index(
            pd.Index(np.arange(ori_test_cell.shape[0])))
        result2 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:20] == 1)  # top20
        result3 = np.count_nonzero(ori_test_cell['immunogenicity'].values[:50] == 1)  # top50
        holding['cell'].append((result1,result2,result3))
        # test in covid
        ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt', sep='\t')
        ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
        ori_test_covid = retain_910(ori)
        graphs_test, graph_labels_test = Graph_Constructor.entrance(ori_test_covid, after_pca, hla_dic, dic_inventory)
        generator_test = PaddedGraphGenerator(graphs=graphs_test)
        input = generator_test.flow(graphs_test)
        prediction = model.predict(input)
        hard = [1 if item >= 0.5 else 0 for item in prediction]
        result1 = recall_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result2 = recall_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        result3 = precision_score(ori_test_covid['immunogenicity-con'], hard)  # convalescent recall
        result4 = precision_score(ori_test_covid['immunogenicity'], hard)  # unexposed recall
        holding['covid'].append((result1, result2, result3, result4))
        print('round {}, finished covid'.format(i))
        i += 1




    # test
    ori_test =pd.read_csv('/Users/ligk2e/Desktop/immuno3/data/remove_low_negative/complete_data_filter910.txt',sep='\t')
    graphs_test,graph_labels_test = Graph_Constructor.entrance(ori_test,after_pca,hla_dic,dic_inventory)
    generator_test = PaddedGraphGenerator(graphs=graphs_test)
    graph_labels_test = pd.get_dummies(graph_labels_test, drop_first=True)

    input = generator_test.flow(graphs_test)
    result = model.predict(input)
    ori_test['result'] = result[:,0]
    ori_test = ori_test.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori_test.shape[0])))




    # neoantigen
    ori_test = pd.read_csv('immuno2/data/mannual_cancer_testing_fiter910.txt',sep='\t')
    graphs_test,graph_labels_test = Graph_Constructor.entrance(ori_test,after_pca,hla_dic,dic_inventory)
    generator_test = PaddedGraphGenerator(graphs=graphs_test)
    graph_labels_test = pd.get_dummies(graph_labels_test, drop_first=True)

    input = generator_test.flow(graphs_test)
    result = model.predict(input)
    draw_ROC(graph_labels_test,result)
    draw_PR(graph_labels_test,result)
    a = pd.DataFrame({'true':graph_labels_test.values[:,0],'pred':result[:,0]})
    a = a.sort_values('pred',ascending=False)
    a = a.set_index(pd.Index(np.arange(a.shape[0])))



















