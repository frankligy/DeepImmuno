'''
This script is to implement Wass gan, same as feedback GAN paper
To start with the simplist situation, let's consider generating immunogenic
epitope for HLA-A0201, there are 2046 positive instance collected, we use them as
real image. see if it will work. If work, we can scale up to some more challenging task.
Stay tuned!
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# build the model
class ResBlock(nn.Module):
    def __init__(self,hidden):    # hidden means the number of filters
        super(ResBlock,self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),    # in_place = True
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1),
        )

    def forward(self,input):   # input [N, hidden, seq_len]
        output = self.res_block(input)
        return input + 0.3*output   # [N, hidden, seq_len]  doesn't change anything

class Generator(nn.Module):
    def __init__(self,hidden,seq_len,n_chars,batch_size):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(128,hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden,n_chars,kernel_size=1)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self,noise):  # noise [batch,128]
        output = self.fc1(noise)    # [batch,hidden*seq_len]
        output = output.view(-1,self.hidden,self.seq_len)   # [batch,hidden,seq_len]
        output = self.block(output)  # [batch,hidden,seq_len]
        output = self.conv1(output)  # [batch,n_chars,seq_len]
        '''
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.
        
        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as 
        being created form stracth
        '''
        output = output.transpose(1,2)  # [batch,seq_len,n_chars]
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len,self.n_chars)
        output = F.gumbel_softmax(output,tau=0.75,hard=False)  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size,self.seq_len,self.n_chars)   # [batch,seq_len,n_chars]
        return output

class Discriminator(nn.Module):
    def __init__(self,hidden,n_chars,seq_len):
        super(Discriminator,self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(n_chars,hidden,1)
        self.fc = nn.Linear(seq_len*hidden,1)
        self.hidden = hidden
        self.n_chars = n_chars
        self.seq_len = seq_len

    def forward(self,input):  # input [N,seq_len,n_chars]
        output = input.transpose(1,2)   # input [N, n_chars, seq_len]
        output = output.contiguous()
        output = self.conv1(output)  # [N,hidden,seq_len]
        output = self.block(output)  # [N, hidden, seq_len]
        output = output.view(-1,self.seq_len*self.hidden)  # [N, hidden*seq_len]
        output = self.fc(output)   # [N,1]
        return output

# define dataset
class real_dataset_class(torch.utils.data.Dataset):
    def __init__(self,raw,seq_len,n_chars):  # raw is a ndarray ['ARRRR','NNNNN']
        self.raw = raw
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.post = self.process()


    def process(self):
        result = torch.empty(len(self.raw),self.seq_len,self.n_chars)   # [N,seq_len,n_chars]
        amino = 'ARNDCQEGHILKMFPSTWYV-'
        identity = torch.eye(n_chars)
        for i in range(len(self.raw)):
            pep = self.raw[i]
            if len(pep) == 9:
                pep = pep[0:4] + '-' + pep[4:]
            inner = torch.empty(len(pep),self.n_chars)
            for p in range(len(pep)):
                inner[p] = identity[amino.index(pep[p].upper()), :]
            encode = torch.tensor(inner)   # [seq_len,n_chars]
            result[i] = encode
        return result


    def __getitem__(self,index):
        return self.post[index]

    def __len__(self):
        return self.post.shape[0]

# auxiliary function during training GAN
def sample_generator(batch_size):
    noise = torch.randn(batch_size,128).to(device)  # [N, 128]
    generated_data = G(noise)   # [N, seq_len, n_chars]
    return generated_data

def calculate_gradient_penalty(real_data,fake_data,lambda_=10):
    alpha = torch.rand(batch_size,1,1).to(device)
    alpha = alpha.expand_as(real_data)   # [N,seq_len,n_chars]
    interpolates = alpha * real_data + (1-alpha) * fake_data  # [N,seq_len,n_chars]
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    disc_interpolates = D(interpolates)
    # below, grad function will return a tuple with length one, so only take [0], it will be a tensor of shape inputs, gradient wrt each input
    gradients = torch.autograd.grad(outputs=disc_interpolates,inputs=interpolates,grad_outputs=torch.ones(disc_interpolates.size()),create_graph=True,retain_graph=True)[0]
    gradients = gradients.contiguous().view(batch_size,-1)  # [N, seq_len*n_chars]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)  # [N,]
    gradient_penalty = lambda_* ((gradients_norm - 1) ** 2).mean()     # []
    return gradient_penalty


def discriminator_train(real_data):
    D_optimizer.zero_grad()
    fake_data = sample_generator(batch_size)   # generate a mini-batch of fake data
    d_fake_pred = D(fake_data)    # what's the prediction you get via discriminator
    d_fake_error = d_fake_pred.mean()   # compute mean, return a scalar value
    d_real_pred = D(real_data)      # what's the prediction you get for real data via discriminator
    d_real_error = d_real_pred.mean()   # compute mean
    gradient_penalty = calculate_gradient_penalty(real_data,fake_data)   # calculate gradient penalty
    d_error_total = d_fake_error - d_real_error + gradient_penalty  # []   # total error, you want to minimize this, so you hope fake image be more real
    w_dist =  d_real_error - d_fake_error
    d_error_total.backward()
    D_optimizer.step()
    return d_fake_error,d_real_error,gradient_penalty, d_error_total, w_dist

def generator_train():
    G_optimizer.zero_grad()
    g_fake_data = sample_generator(batch_size)
    dg_fake_pred = D(g_fake_data)
    g_error_total = -torch.mean(dg_fake_pred)
    g_error_total.backward()
    G_optimizer.step()
    return g_error_total



# processing function from previous code
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


# post utils functions
def inverse_transform(hard):   # [N,seq_len]
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    result = []
    for row in hard:
        temp = ''
        for col in row:
            aa = amino[col]
            temp += aa
        result.append(temp)
    return result


if __name__ == '__main__':
    batch_size = 64
    lr = 0.0001
    num_epochs = 100
    seq_len = 10
    hidden = 128
    n_chars = 21
    d_steps = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = pd.read_csv('wassGAN/gan_a0201.csv')
    raw = data['peptide'].values
    real_dataset = real_dataset_class(raw,seq_len,n_chars)


    # training
    G = Generator(hidden,seq_len,n_chars,batch_size).to(device)
    D = Discriminator(hidden,n_chars,seq_len).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.9)) # usually should be (0.9,0.999), (momentum,RMSprop)
    D_optimizer = torch.optim.Adam(D.parameters(),lr=lr,betas=(0.5,0.9))

    counter = 0
    c_epoch = 0
    array1,array2,array3,array4,array5 = [],[],[],[],[]
    for epoch in range(num_epochs):
        '''
        The way I understand this trianing process is:
        you first trian the discriminator to minimize the discrepancy between fake and real data, parameters in generator stay constant.
        Then you train the generator, it will adapt to generate more real image.
        It seems like the purpose is just to generate, not discriminate 
        '''
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]
        real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for mini_batch in real_dataloader:
            d_fake_err,d_real_err,gradient_penalty,d_error_total,w_dist = discriminator_train(mini_batch)
            grad_penalties.append(gradient_penalty.detach().cpu().numpy())
            d_real_losses.append(d_real_err.detach().cpu().numpy())
            d_fake_losses.append(d_fake_err.detach().cpu().numpy())
            D_losses.append(d_error_total.detach().cpu().numpy())
            W_dist.append(w_dist.detach().cpu().numpy())

            if counter % d_steps == 0:
                g_err = generator_train()
                G_losses.append(g_err.detach().cpu().numpy())

            counter += 1

        summary_string = 'Epoch{0}/{1}: d_real_loss-{2:.2f},d_fake_loss-{3:.2f},d_total_loss-{4:.2f},G_total_loss-{5:.2f},W_dist-{6:.2f}'\
                    .format(epoch+1,num_epochs,np.mean(d_real_losses),np.mean(d_fake_losses),np.mean(D_losses),np.mean(G_losses),np.mean(W_dist))
        print(summary_string)
        array1.append(np.mean(d_real_losses))
        array2.append(np.mean(d_fake_losses))
        array3.append(np.mean(D_losses))
        array4.append(np.mean(G_losses))
        array5.append(np.mean(W_dist))

        if epoch % 20 == 19:
            total = []
            for i in range(16):
                generation = sample_generator(64).detach().cpu().numpy()  # [N,seq_len,n_chars]
                hard = np.argmax(generation, axis=2)  # [N,seq_len]
                pseudo = inverse_transform(hard)
                df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                                   'immunogenicity': [1 for i in range(len(pseudo))]})
                total.append(df)
            df_all = pd.concat(total)
            df_all.to_csv('/Users/ligk2e/Desktop/df_all_epoch.csv', index=None)

        c_epoch += 1

        # visulization, draw history/log
        ax0 = plt.subplot(5,1,1)
        ax0.plot(np.arange(num_epochs),array1)
        ax0.set_ylabel('d_real_losses')

        ax1 = plt.subplot(5,1,2)
        ax1.plot(np.arange(num_epochs),array2)
        ax1.set_ylabel('d_fake_losses')

        ax2 = plt.subplot(5,1,3)
        ax2.plot(np.arange(num_epochs),array3)
        ax2.set_ylabel('D_losses')

        ax3 = plt.subplot(5,1,4)
        ax3.plot(np.arange(num_epochs),array4)
        ax3.set_ylabel('G_losses')

        ax4 = plt.subplot(5,1,5)
        ax4.plot(np.arange(num_epochs),array5)
        ax4.set_ylabel('W_dist')

        # visualization, t-sne
        from sklearn.manifold import TSNE
        after_pca = np.loadtxt('../immuno/immuno2/data/after_pca.txt')
        hla = pd.read_csv('../immuno/immuno2/data/hla2paratopeTable_aligned.txt', sep='\t')
        hla_dic = hla_df_to_dic(hla)
        inventory = list(hla_dic.keys())
        dic_inventory = dict_inventory(inventory)
        dataset = construct_aaindex(data, hla_dic, after_pca)
        X = np.empty((len(dataset), 12 * 10))
        for i, (x, y, _) in enumerate(dataset):
            x = x.reshape(-1)
            X[i, :] = x


        X_embedded = TSNE(n_components=2).fit_transform(X)
        fig,ax = plt.subplots()
        ax.scatter(X_embedded[:,0],X_embedded[:,1],color='b')

        total = []
        for i in range(16):
            generation = sample_generator(64).detach().cpu().numpy()  # [N,seq_len,n_chars]
            hard = np.argmax(generation,axis=2)  # [N,seq_len]
            pseudo = inverse_transform(hard)
            df = pd.DataFrame({'peptide':pseudo,'HLA':['HLA-A*0201' for i in range(len(pseudo))],'immunogenicity':[1 for i in range(len(pseudo))]})
            test_dataset = construct_aaindex(df, hla_dic, after_pca)
            test_X = np.empty((len(test_dataset), 12 * 10))
            for i, (x, y, _) in enumerate(test_dataset):
                x = x.reshape(-1)
                test_X[i, :] = x
            total.append(test_X)
        test_X = np.concatenate(total,axis=0)
        coalesed = np.concatenate([X,test_X],axis=0)
        coalesed_embedded = TSNE(n_components=2).fit_transform(coalesed)
        from itertools import repeat
        fig,ax = plt.subplots()
        ax.scatter(coalesed_embedded[:,0],coalesed_embedded[:,1],color=list(repeat('b',X.shape[0]))+list(repeat('r',test_X.shape[0])))

        # compare total noise
        total_noise = np.random.randn(1024,10,21)
        hard = np.argmax(total_noise,axis=2)
        pseudo = inverse_transform(hard)
        df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                           'immunogenicity': [1 for i in range(len(pseudo))]})
        df.to_csv('/Users/ligk2e/Desktop/df_noise.csv',index=None)
        test_dataset = construct_aaindex(df, hla_dic, after_pca)
        test_X = np.empty((len(test_dataset), 12 * 10))
        for i, (x, y, _) in enumerate(test_dataset):
            x = x.reshape(-1)
            test_X[i, :] = x
        coalesed = np.concatenate([X, test_X], axis=0)
        coalesed_embedded = TSNE(n_components=2).fit_transform(coalesed)
        from itertools import repeat
        fig, ax = plt.subplots()
        ax.scatter(coalesed_embedded[:, 0], coalesed_embedded[:, 1],
                   color=list(repeat('b', X.shape[0])) + list(repeat('r', test_X.shape[0])))

        # subject generated epitope for immunogenicity prediction using CNN
        total = []
        for i in range(16):
            generation = sample_generator(64).detach().cpu().numpy()  # [N,seq_len,n_chars]
            hard = np.argmax(generation,axis=2)  # [N,seq_len]
            pseudo = inverse_transform(hard)
            df = pd.DataFrame({'peptide':pseudo,'HLA':['HLA-A*0201' for i in range(len(pseudo))],'immunogenicity':[1 for i in range(len(pseudo))]})
            total.append(df)
        df_all = pd.concat(total)
        df_all.to_csv('/Users/ligk2e/Desktop/df_all.csv',index=None)


        # here we perform a time-series inspection, take random noise(already done), epoch 20, 40, 60, 80, 100,
        # see their t-sne distribution
        # see their CNN prediction

        df = pd.read_csv('/Users/ligk2e/Desktop/immuno3/df/df_all_epoch100.csv')
        test_dataset = construct_aaindex(df, hla_dic, after_pca)
        test_X = np.empty((len(test_dataset), 12 * 10))
        for i, (x, y, _) in enumerate(test_dataset):
            x = x.reshape(-1)
            test_X[i, :] = x
        coalesed = np.concatenate([X, test_X], axis=0)
        coalesed_embedded = TSNE(n_components=2,random_state=42).fit_transform(coalesed)
        from itertools import repeat
        fig, ax = plt.subplots()
        ax.scatter(coalesed_embedded[:, 0], coalesed_embedded[:, 1],
                   color=list(repeat('b', X.shape[0])) + list(repeat('r', test_X.shape[0])),s=3)
        h1 = [ax.plot([], [], color=i, marker='o', markersize=5, ls='')[0] for i in ['b','r']]
        leg = ax.legend(handles=h1,labels=['Real peptides','Pseudo peptides'])
        ax.add_artist(leg)
        ax.set_xlabel('t-sne1')
        ax.set_ylabel('t-sne2')
        plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/Supplementary2/tsne/epoch100.pdf')


        # how about umap
        # pip install umap-learn
        import umap
        reducer = umap.UMAP()
        coalesed_embedded = reducer.fit_transform(coalesed)
        from itertools import repeat
        fig, ax = plt.subplots()
        ax.scatter(coalesed_embedded[:, 0], coalesed_embedded[:, 1],
                   color=list(repeat('b', X.shape[0])) + list(repeat('r', test_X.shape[0])),s=3)

        # how about PCA
        from sklearn.decomposition import PCA
        reducer = PCA()
        reducer.fit(X)
        explained = reducer.explained_variance_ratio_
        total = 0
        for i,v in enumerate(explained):
            total += v
            if total > 0.9:
                print(i)
                break   # PC54
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=54)
        coalesed_embedded = reducer.fit_transform(coalesed)
        from itertools import repeat
        fig, ax = plt.subplots()
        ax.scatter(coalesed_embedded[:, 0], coalesed_embedded[:, 1],
                   color=list(repeat('b', X.shape[0])) + list(repeat('r', test_X.shape[0])),s=3)
        h1 = [ax.plot([], [], color=i, marker='o', markersize=5, ls='')[0] for i in ['b','r']]
        leg = ax.legend(handles=h1,labels=['Real peptides','Pseudo peptides'])
        ax.add_artist(leg)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.savefig('/Users/ligk2e/Desktop/immuno3/figures/figure4/pca/epoch100.pdf')




        # sequence matching
        import pandas as pd
        import numpy as np
        data = pd.read_csv('wassGAN/gan_a0201.csv')
        raw = data['peptide'].values
        from difflib import SequenceMatcher
        def similar(a,b):
            return SequenceMatcher(None,a,b).ratio()
        generate = pd.read_csv('/Users/ligk2e/Desktop/immuno3/df/df_all_epoch100.csv')
        seq = generate['peptide'].values

        whole1 = []  # store mean value
        whole2 = []  # store max value
        for item1 in seq:
            total = []
            for item2 in raw:
                total.append((item2, similar(item1, item2)))
            total = np.array(sorted(total, reverse=True, key=lambda x: x[1]))[:,1].astype(np.float64)
            whole1.append(total.mean())
            whole2.append(total.max())

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Arial'

        import seaborn as sns
        plot = sns.displot(data=pd.DataFrame({'Max Similarity': whole2}),x='Max Similarity',kde=True)
        plt.grid(b=True,axis='y',alpha=0.3)

        plot.savefig('/Users/ligk2e/Desktop/immuno3/figures/similarity.pdf')
        np.sum(np.where(np.array(whole2) > 0.75, 1,0))  # 891- 60%




        with open('/Users/ligk2e/Desktop/log.txt','w') as f:
            for item1 in seq:
                total = []
                for item2 in raw:
                    total.append((item2,similar(item1,item2)))
                total = sorted(total,reverse=True,key=lambda x:x[1])[:5]
                print('{0}:{1}'.format(item1,total),file=f)




        #save model
        torch.save(G.state_dict(),'../immuno/immuno2/models/wassGAN_G.pth')
        torch.save(D.state_dict(),'../immuno/immuno2/models/wassGAN_D.pth')



















