'''
Program to run deepimmuno-GAN to generate Pseudo-immunogenic sequences for HLA-A*0201
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os


# build the model
class ResBlock(nn.Module):
    def __init__(self, hidden):  # hidden means the number of filters
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),  # in_place = True
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, input):  # input [N, hidden, seq_len]
        output = self.res_block(input)
        return input + 0.3 * output  # [N, hidden, seq_len]  doesn't change anything


class Generator(nn.Module):
    def __init__(self, hidden, seq_len, n_chars, batch_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, kernel_size=1)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self, noise):  # noise [batch,128]
        output = self.fc1(noise)  # [batch,hidden*seq_len]
        output = output.view(-1, self.hidden, self.seq_len)  # [batch,hidden,seq_len]
        output = self.block(output)  # [batch,hidden,seq_len]
        output = self.conv1(output)  # [batch,n_chars,seq_len]
        '''
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.

        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as 
        being created form stracth
        '''
        output = output.transpose(1, 2)  # [batch,seq_len,n_chars]
        output = output.contiguous()
        output = output.view(self.batch_size * self.seq_len, self.n_chars)
        output = F.gumbel_softmax(output, tau=0.75,
                                  hard=False)  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size, self.seq_len, self.n_chars)  # [batch,seq_len,n_chars]
        return output


class Discriminator(nn.Module):
    def __init__(self, hidden, n_chars, seq_len):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(n_chars, hidden, 1)
        self.fc = nn.Linear(seq_len * hidden, 1)
        self.hidden = hidden
        self.n_chars = n_chars
        self.seq_len = seq_len

    def forward(self, input):  # input [N,seq_len,n_chars]
        output = input.transpose(1, 2)  # input [N, n_chars, seq_len]
        output = output.contiguous()
        output = self.conv1(output)  # [N,hidden,seq_len]
        output = self.block(output)  # [N, hidden, seq_len]
        output = output.view(-1, self.seq_len * self.hidden)  # [N, hidden*seq_len]
        output = self.fc(output)  # [N,1]
        return output


# define dataset
class real_dataset_class(torch.utils.data.Dataset):
    def __init__(self, raw, seq_len, n_chars):  # raw is a ndarray ['ARRRR','NNNNN']
        self.raw = raw
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.post = self.process()

    def process(self):
        result = torch.empty(len(self.raw), self.seq_len, self.n_chars)  # [N,seq_len,n_chars]
        amino = 'ARNDCQEGHILKMFPSTWYV-'
        identity = torch.eye(n_chars)
        for i in range(len(self.raw)):
            pep = self.raw[i]
            if len(pep) == 9:
                pep = pep[0:4] + '-' + pep[4:]
            inner = torch.empty(len(pep), self.n_chars)
            for p in range(len(pep)):
                inner[p] = identity[amino.index(pep[p].upper()), :]
            encode = torch.tensor(inner)  # [seq_len,n_chars]
            result[i] = encode
        return result

    def __getitem__(self, index):
        return self.post[index]

    def __len__(self):
        return self.post.shape[0]


# auxiliary function during training GAN
def sample_generator(batch_size):
    batch_size = 64
    lr = 0.0001
    num_epochs = 100
    seq_len = 10
    hidden = 128
    n_chars = 21
    d_steps = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(hidden,seq_len,n_chars,batch_size).to(device)
    G.load_state_dict(torch.load('./models/wassGAN_G.pth'))
    noise = torch.randn(batch_size, 128).to(device)  # [N, 128]
    generated_data = G(noise)  # [N, seq_len, n_chars]
    return generated_data


def calculate_gradient_penalty(real_data, fake_data, lambda_=10):
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)  # [N,seq_len,n_chars]
    interpolates = alpha * real_data + (1 - alpha) * fake_data  # [N,seq_len,n_chars]
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)
    # below, grad function will return a tuple with length one, so only take [0], it will be a tensor of shape inputs, gradient wrt each input
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()), create_graph=True,
                                    retain_graph=True)[0]
    gradients = gradients.contiguous().view(batch_size, -1)  # [N, seq_len*n_chars]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)  # [N,]
    gradient_penalty = lambda_ * ((gradients_norm - 1) ** 2).mean()  # []
    return gradient_penalty


def discriminator_train(real_data):
    D_optimizer.zero_grad()
    fake_data = sample_generator(batch_size)  # generate a mini-batch of fake data
    d_fake_pred = D(fake_data)  # what's the prediction you get via discriminator
    d_fake_error = d_fake_pred.mean()  # compute mean, return a scalar value
    d_real_pred = D(real_data)  # what's the prediction you get for real data via discriminator
    d_real_error = d_real_pred.mean()  # compute mean
    gradient_penalty = calculate_gradient_penalty(real_data, fake_data)  # calculate gradient penalty
    d_error_total = d_fake_error - d_real_error + gradient_penalty  # []   # total error, you want to minimize this, so you hope fake image be more real
    w_dist = d_real_error - d_fake_error
    d_error_total.backward()
    D_optimizer.step()
    return d_fake_error, d_real_error, gradient_penalty, d_error_total, w_dist


def generator_train():
    G_optimizer.zero_grad()
    g_fake_data = sample_generator(batch_size)
    dg_fake_pred = D(g_fake_data)
    g_error_total = -torch.mean(dg_fake_pred)
    g_error_total.backward()
    G_optimizer.step()
    return g_error_total


# processing function from previous code
def peptide_data_aaindex(peptide, after_pca):  # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide, after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide, after_pca)
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
    # print(hla)
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


def hla_data_aaindex(hla_dic, hla_type, after_pca):  # return numpy array [34,12,1]
    try:
        seq = hla_dic[hla_type]
    except KeyError:
        hla_type = rescue_unknown_hla(hla_type, dic_inventory)
        seq = hla_dic[hla_type]
    encode = aaindex(seq, after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode


def construct_aaindex(ori, hla_dic, after_pca):
    series = []
    for i in range(ori.shape[0]):
        peptide = ori['peptide'].iloc[i]
        hla_type = ori['HLA'].iloc[i]
        immuno = np.array(ori['immunogenicity'].iloc[i]).reshape(1, -1)  # [1,1]
        '''
        If 'classfication': ['immunogenicity']
        If 'regression': ['potential']
        '''

        encode_pep = peptide_data_aaindex(peptide, after_pca)  # [10,12]

        encode_hla = hla_data_aaindex(hla_dic, hla_type, after_pca)  # [46,12]
        series.append((encode_pep, encode_hla, immuno))
    return series


def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic


def aaindex(peptide, after_pca):
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)  # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded


# post utils functions
def inverse_transform(hard):  # [N,seq_len]
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    result = []
    for row in hard:
        temp = ''
        for col in row:
            aa = amino[col]
            temp += aa
        result.append(temp)
    return result

def main(args):
    #batch= args.batch
    outdir = args.outdir
    print("outdir is {}".format(outdir))
    generation = sample_generator(64).detach().cpu().numpy()  # [N,seq_len,n_chars]
    hard = np.argmax(generation, axis=2)  # [N,seq_len]
    pseudo = inverse_transform(hard)
    df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                       'immunogenicity': [1 for i in range(len(pseudo))]})
    df.to_csv(os.path.join(outdir,'deepimmuno-GAN-result.txt'),sep='\t',index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepImmuno-GAN to generate immunogenic peptide')
    #parser.add_argument('--batch',type=str,default=1,help='how many sequences you''d like? one batch is 64')
    parser.add_argument('--outdir',type=str,default='.',help='specifying your output folder')
    args = parser.parse_args()
    main(args)