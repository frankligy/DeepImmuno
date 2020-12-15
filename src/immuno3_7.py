'''
This is for application on SARS-CoV-2 dataset
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

def seperateCNN():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)  # 9
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # 32
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
    y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2,outputs=y)

    combined = layers.concatenate([x.output,y.output])
    z = layers.Dense(128,activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1,activation='sigmoid')(z)

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model

def pull_peptide_aaindex(dataset):
    result = np.empty([len(dataset),10,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][0]
    return result


def pull_hla_aaindex(dataset):
    result = np.empty([len(dataset),46,12,1])
    for i in range(len(dataset)):
        result[i,:,:,:] = dataset[i][1]
    return result


def pull_label_aaindex(dataset):
    col = [item[2] for item in dataset]
    result = [0 if item == 'Negative' else 1 for item in col]
    result = np.expand_dims(np.array(result),axis=1)
    return result

def pull_label_aaindex(dataset):
    result = np.empty([len(dataset),1])
    for i in range(len(dataset)):
        result[i,:] = dataset[i][2]
    return result

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

def draw_ROC(y_true,y_pred):
    from sklearn.metrics import roc_curve,auc
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
    from sklearn.metrics import precision_recall_curve,auc
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

def read_fasta_and_chop_to_N(path,N):
    with open(path,'r') as f:
        lis = f.readlines()[1:]
    lis = [raw.rstrip('\n') for raw in lis]
    seq = ''.join(lis)
    bucket = []
    for i in range(0,len(seq)-N+1,1):
        frag = seq[i:i+N]
        bucket.append(frag)
    return seq,bucket

def set_query_df(frag):
    from itertools import product
    hla = ['HLA-A*0101','HLA-A*0201','HLA-A*0301','HLA-A*1101','HLA-A*2402','HLA-B*0702','HLA-B*0801','HLA-B*1501','HLA-B*4001','HLA-C*0702']
    combine = list(product(frag,hla))
    col1 = [item[0] for item in combine]  # peptide
    col2 = [item[1] for item in combine]  # hla
    col3 = [0 for item in combine]   # immunogenicity
    df = pd.DataFrame({'peptide':col1,'HLA':col2,'immunogenicity':col3})
    return df

def get_score(ori):
    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)


    result = cnn_model.predict([input1,input2])
    ori['result'] = result[:,0]
    return ori

def prepare_plot_each_region(score_df,count,h=10):  # how many hla you query
    from itertools import repeat
    # x coordinate
    x = []
    for i in range(count):
        x.extend(list(repeat(i,h)))
    # y coordinate
    y = score_df['result'].values
    # color coordiate
    tmp = list(repeat([0,1,2,3,4,5,6,7,8,9],count))
    c = [j for i in tmp for j in i]
    # plot
    fig,ax = plt.subplots()
    ax.scatter(x=x,y=y,c=c,cmap='tab10',alpha=1,s=5)
    plt.show()
    return x,y,c

def prepare_plot_each_region_mean(score_df,count,h=10):
    lis = np.split(score_df['result'].values,count)
    y = np.array([item.mean() for item in lis])
    fig,ax = plt.subplots()
    ax.bar(x=np.arange(count),height=y)
    plt.show()
    return y

def wrapper(frag,count):
    orf_score_df = set_query_df(frag)
    orf_score_df = get_score(orf_score_df)
    x,y,c = prepare_plot_each_region(orf_score_df,count)
    y = prepare_plot_each_region_mean(orf_score_df,count)
    return y

def plot_violin(dataset,xlabel=None,ylabel=None):
    fig,ax = plt.subplots()
    vp = ax.violinplot(dataset = dataset,showextrema=False)
    for part in vp['bodies']:
        part.set(facecolor='#D43F3A',edgecolor='black',alpha=1)

    tmp = [np.percentile(data,[25,50,75]) for data in dataset]
    def get_whisker(tmp,dataset):
        whisker = []
        for item,data in zip(tmp,dataset):
            q1 = item[0]
            median = item[1]
            q3 = item[2]
            iqr = q3-q1
            upper = q3 + 1.5*iqr
            upper = np.clip(upper,q3,data.max())
            lower = q1 - 1.5*iqr
            lower = np.clip(lower,data.min(),q1)
            whisker.append((upper,lower))
        return whisker
    whisker = get_whisker(tmp,[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10])
    x = np.arange(len([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10])) + 1
    ax.scatter(x,[item[1] for item in tmp],marker='o',color='white',s=30,zorder=3)
    ax.vlines(x,[item[0] for item in tmp],[item[2] for item in tmp],color='black',linestyle='-',lw=5)
    ax.vlines(x,[item[0] for item in whisker], [item[1] for item in whisker],color='k',linestyle='-',lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel)
    ax.set_ylabel(ylabel)

if __name__ == '__main__':
    ori = pd.read_csv('/Users/ligk2e/Desktop/sars_cov_2.txt',sep='\t')
    cnn_model = seperateCNN()
    cnn_model.load_weights('immuno2/models/cnn_model_331_3_7/')

    after_pca = np.loadtxt('immuno2/data/after_pca.txt')
    ori = ori.sample(frac=1,replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
    ori = retain_910(ori)
    hla = pd.read_csv('immuno2/data/hla2paratopeTable_aligned.txt',sep='\t')
    hla_dic = hla_df_to_dic(hla)
    inventory = list(hla_dic.keys())
    dic_inventory = dict_inventory(inventory)

    dataset = construct_aaindex(ori,hla_dic,after_pca)
    input1 = pull_peptide_aaindex(dataset)
    input2 = pull_hla_aaindex(dataset)
    label = pull_label_aaindex(dataset)

    result = cnn_model.predict([input1,input2])
    ori['result'] = result[:,0]
    ori = ori.sort_values(by='result',ascending=False).set_index(pd.Index(np.arange(ori.shape[0])))
    draw_ROC(ori['immunogenicity-un'],ori['result'])
    draw_PR(ori['immunogenicity-un'],ori['result'])
    from sklearn.metrics import classification_report
    hard = [1 if item > 0.5 else 0 for item in ori['result']]
    tmp = classification_report(ori['immunogenicity-un'],hard)
    print(tmp)

    ori.to_csv('/Users/ligk2e/Desktop/immuno3/sars_cov_2_result.csv',index=None)


    # compare to deephlapan and iedb
    '''
                      cnn    deeplanpan   iedb
    convalescent:     0.68     0.40        0.52
    unexposed:        0.88     0.14        0.38
    
    if precision
    
                       cnn    deeplanpan   iedb
    convalescent:     0.28     0.28        0.25
    unexposed:        0.11     0.05        0.02
    
    '''

    fig,ax = plt.subplots()
    ax.bar([0,5,10,15],[0.4,0.14,0.28,0.05],color='#E36DF2',label='Deephlapan',width=0.8)
    ax.bar([1,6,11,16],[0.52,0.38,0.25,0.02],color='#04BF7B',label='IEDB',width=0.8)
    ax.bar([2,7,12,17],[0.68,0.88,0.28,0.11],color='#F26D6D',label='DeepImmuno-CNN',width=0.8)
    ax.set_xticks([1,6,11,16])
    ax.set_xticklabels(['Convalescent','Unexposed']*2)
    ax.set_ylabel('Recall')
    ax.legend(frameon=False)
    ax.grid(True,alpha=0.3)
    ax.set_ylim([0,1])
    x = [0,1,2,5,6,7,10,11,12,15,16,17]
    y = [0.4,0.52,0.68,0.14,0.38,0.88,0.28,0.25,0.28,0.05,0.02,0.11]
    for i in range(len(x)):
        ax.text(x[i]-0.3,y[i]+0.02,s=y[i],fontsize=8)
    ax2 = ax.twinx()
    ax2.set_ylim([0,1])
    ax2.set_ylabel('Precision')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/covid_test.pdf')

    # now, Nathan suggested to split them up
    fig = plt.figure(figsize=(6.4,4.8))
    ax1 = plt.axes([0.1,0.1,0.35,0.80])
    ax2 = plt.axes([0.55,0.1,0.35,0.80])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.bar([0,5],[0.4,0.14],color='#E36DF2',label='Deephlapan',width=0.8)
    ax1.bar([1,6],[0.52,0.38],color='#04BF7B',label='IEDB',width=0.8)
    ax1.bar([2,7],[0.68,0.88],color='#F26D6D',label='DeepImmuno-CNN',width=0.8)
    ax1.set_xticks([1,6])
    ax1.set_xticklabels(['Convalescent','Unexposed'])
    ax1.legend(frameon=True)
    ax1.grid(True,alpha=0.3,axis='y')
    ax1.set_ylabel('Recall')

    ax2.bar([10,15],[0.28,0.05],color='#E36DF2',label='Deephlapan',width=0.8)
    ax2.bar([11,16],[0.25,0.02],color='#04BF7B',label='IEDB',width=0.8)
    ax2.bar([12,17],[0.28,0.11],color='#F26D6D',label='DeepImmuno-CNN',width=0.8)
    ax2.set_xticks([11,16])
    ax2.set_xticklabels(['Convalescent','Unexposed'])
    ax2.set_ylabel('Precision')
    ax2.grid(True,alpha=0.3,axis='y')

    x1 = [0,1,2,5,6,7]
    y1 = [0.4,0.52,0.68,0.14,0.38,0.88]

    x2 = [10,11,12,15,16,17]
    y2 = [0.28,0.25,0.28,0.05,0.02,0.11]
    for i in range(len(x1)):
        ax1.text(x1[i]-0.3,y1[i]+0.02,s=y1[i],fontsize=8)
    for i in range(len(x2)):
        ax2.text(x2[i]-0.35,y2[i]+0.002,s=y2[i],fontsize=8)

    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure2/covid_test_split.pdf')

    #### Next step, chop whole SARS-Cov2-proteome
    '''
    orf1ab: polypeptide, nsp, replicase..., length 7096, 9mer: 7088, 10mer: 7087
    orf2: spike, length 1273, 9mer: 1265, 10mer: 1264
    orf3a: accessory, length 275, 9mer: 267, 10mer: 266
    orf4: envelope, length 75, 9mer: 67, 10mer: 66
    orf5: membrane, length 222, 9mer: 214, 10mer: 213
    orf6: accessory, length 61, 9mer: 53, 10mer: 52
    orf7a: accessory, length 121, 9mer 113, 10mer: 112
    orf7b: accessory, length 43, 9mer 35   (missing in nature immunology paper), 10mer: 34
    orf8: accessory, length 121, 9mer 113, 10mer: 112
    orf9: nucleocapside glycoprotein, length 419, 9mer 411, 10mer 410
    orf10: accessory, length 38, 9mer: 30, 10mer: 29
    '''

    # first consider 9-mer
    orf1ab_seq,orf1ab_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF1ab.fa',9)
    orf2_seq, orf2_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF2-spike.fa', 9)
    orf3a_seq, orf3a_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF3a-accessory.fa', 9)
    orf4, orf4_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF4-env.fa', 9)
    orf5, orf5_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF5-mem.fa', 9)
    orf6, orf6_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF6-accessory.fa', 9)
    orf7a, orf7a_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF7a-accessory.fa', 9)
    orf7b,orf7b_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF7b-accessory.fa', 9)
    orf8,orf8_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF8-accessory.fa', 9)
    orf9,orf9_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF9-nuc.fa', 9)
    orf10,orf10_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF10-accessory.fa', 9)

    y1 = wrapper(orf1ab_frag,7088)
    y2 = wrapper(orf2_frag,1265)
    y3 = wrapper(orf3a_frag,267)
    y4 = wrapper(orf4_frag,67)
    y5 = wrapper(orf5_frag,214)
    y6 = wrapper(orf6_frag,53)
    y7 = wrapper(orf7a_frag,113)
    y7b = wrapper(orf7b_frag,35)
    y8 = wrapper(orf8_frag,113)
    y9 = wrapper(orf9_frag,411)
    y10 = wrapper(orf10_frag,30)

    # draw box plot
    fig,ax = plt.subplots()
    bp = ax.boxplot([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10],positions=[0,1,2,3,4,5,6,7,8,9],patch_artist=True,widths=0.8)  # bp is a dictionary
    for box in bp['boxes']:   # box is matplotlib.lines.Line2d object
        box.set(facecolor='#087E8B',alpha=0.6,linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1)
    for median in bp['medians']:
        median.set(color='black',linewidth=1)
    for flier in bp['fliers']:
        flier.set(markersize=1.5)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['ORF1','ORF2','ORF3','ORF4','ORF5','ORF6','ORF7','ORF8','ORF9','ORF10'])
    ax.set_ylabel('Average immunogenicity score')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary figure3/9mer.pdf')

    # draw violin plot
    plot_violin([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10],['ORF1','ORF2','ORF3','ORF4','ORF5','ORF6','ORF7','ORF8','ORF9','ORF10'],'Average immunogenicity score')



    # let's inspect 10mer
    orf1ab_seq,orf1ab_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF1ab.fa',10)
    orf2_seq, orf2_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF2-spike.fa', 10)
    orf3a_seq, orf3a_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF3a-accessory.fa', 10)
    orf4, orf4_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF4-env.fa', 10)
    orf5, orf5_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF5-mem.fa', 10)
    orf6, orf6_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF6-accessory.fa', 10)
    orf7a, orf7a_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF7a-accessory.fa', 10)
    orf7b,orf7b_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF7b-accessory.fa', 10)
    orf8,orf8_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF8-accessory.fa', 10)
    orf9,orf9_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF9-nuc.fa', 10)
    orf10,orf10_frag = read_fasta_and_chop_to_N('/Users/ligk2e/Desktop/immuno3/covid/ORF10-accessory.fa', 10)

    y1 = wrapper(orf1ab_frag,7087)
    y2 = wrapper(orf2_frag,1264)
    y3 = wrapper(orf3a_frag,266)
    y4 = wrapper(orf4_frag,66)
    y5 = wrapper(orf5_frag,213)
    y6 = wrapper(orf6_frag,52)
    y7a = wrapper(orf7a_frag,112)
    y7b = wrapper(orf7b_frag,34)
    y8 = wrapper(orf8_frag,112)
    y9 = wrapper(orf9_frag,410)
    y10 = wrapper(orf10_frag,29)

    # draw box plot
    fig, ax = plt.subplots()
    bp = ax.boxplot([y1, y2, y3, y4, y5, y6, y7a, y8, y9, y10], positions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    patch_artist=True, widths=0.8)  # bp is a dictionary
    for box in bp['boxes']:  # box is matplotlib.lines.Line2d object
        box.set(facecolor='#087E8B', alpha=0.6, linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1)
    for flier in bp['fliers']:
        flier.set(markersize=1.5)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(['ORF1', 'ORF2', 'ORF3', 'ORF4', 'ORF5', 'ORF6', 'ORF7', 'ORF8', 'ORF9', 'ORF10'])
    ax.set_ylabel('Average immunogenicity score')
    plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/supplementary figure3/9mer.pdf')

    # draw violin plot
    plot_violin([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10],
                ['ORF1', 'ORF2', 'ORF3', 'ORF4', 'ORF5', 'ORF6', 'ORF7', 'ORF8', 'ORF9', 'ORF10'],
                'Average immunogenicity score')
























