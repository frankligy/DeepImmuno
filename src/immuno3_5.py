'''
In this script, we decided to remove negative which only been tested in 0,1,2,3 samples
We tried machine learning algorithm, both classify and regression      machine_learning.py
We tried CNN model(classify and regression), resnet is too complicated for only 9000 data instances    CNN_aaindex.py

now we draw a barplot to summarize
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap


if __name__ == '__main__':
    '''
    dengue virus: 408 positive
    rf_classify: 342
    rf_regress: 300
    ada_classify: 316
    ada_regress: 276
    cnn_classify: 345
    cnn_regress: 326
    
    cell_data: 522 total, 35 positive
    rf_classify: 25   top20: 0  top50: 3
    rf_regress: 25    top20: 1  top50: 8
    ada_classify: 26   top20: 0  top50: 5
    ada_regress: 23    top20 2   top50: 4
    cnn_classify: 27   top20: 2   top50: 4
    cnn_regress: 29    top20: 4   top50: 7
    '''

    # draw cell_data
    fig,ax = plt.subplots()
    ax.bar([0,5,10,15,20,25],[0,1,0,2,2,4],color='g',alpha=0.5,label='top20')
    ax.bar([1,6,11,16,21,26],[3,8,5,4,4,7],color='r',alpha=0.5,label='top50')
    ax.bar([2,7,12,17,22,27],[25,25,26,23,27,29],color='b',alpha=0.5,label='cutoff>0.5')
    ax.legend(bbox_to_anchor=(0,1),loc='lower left')
    ax.set_xticks([1,6,11,16,21,26])
    ax.set_xticklabels(['rf_classify','rf_regress','ada_classify','ada_regress','cnn_classify','cnn_regress'],rotation=45)
    x = [0,1,2,5,6,7,10,11,12,15,16,17,20,21,22,25,26,27]
    y = [0,3,25,1,8,25,0,5,26,2,4,23,2,4,27,4,7,29]
    for i in range(len(x)):
        ax.text(x[i]-0.5,y[i]+0.2,y[i])
    ax.plot([-3,30],[3,3],'k--')
    ax.set_title('cell_paper_data')
    plt.show()

    # draw virus data
    fig,ax = plt.subplots()
    ax.bar([0,1,2,3,4,5],[342,300,316,276,345,326],width=0.5)
    x = [0,1,2,3,4,5]
    y = [342,300,316,276,345,326]
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['rf_classify','rf_regress','ada_classify','ada_regress','cnn_classify','cnn_regress'],rotation=45)
    for i in range(len(x)):
        ax.text(x[i] - 0.3, y[i] + 0.2, y[i])
    ax.set_title('dengue virus 408 positive epitopes')
    plt.show()
















