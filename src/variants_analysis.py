import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



with open('/Users/ligk2e/Desktop/github/DeepImmuno/files/covid/ORF2-spike.fa','r') as f:
    spike = f.readlines()
spike = ''.join([item.rstrip('\n') for item in spike[1:]])

record = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'


# D614G
region = spike[605:622]
mer9_normal = [region[i:i+9] for i in range(0,9,1)]

mutate = region[0:8] + 'G' + region[9:]
mer9_mutate = [mutate[i:i+9] for i in range(0,9,1)]

def set_query_df(frag):
    from itertools import product
    hla = ['HLA-A*0101','HLA-A*0201','HLA-A*0301','HLA-A*1101','HLA-A*2402','HLA-B*0702','HLA-B*0801','HLA-B*1501','HLA-B*4001','HLA-C*0702']
    combine = list(product(frag,hla))
    col1 = [item[0] for item in combine]  # peptide
    col2 = [item[1] for item in combine]  # hla
    df = pd.DataFrame({'peptide':col1,'HLA':col2})
    return df

set_query_df(mer9_normal).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/D614G/D614G_normal.csv',index=None,
                                 header=None)
set_query_df(mer9_mutate).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/D614G/D614G_mutate.csv',index=None,
                                 header=None)

result_normal = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/D614G/normal_result.txt',sep='\t')
result_mutate = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/D614G/mutate_result.txt',sep='\t')

# plot by each HLA
fig,axes = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw={'hspace':0.5})
n = list(result_normal.groupby(by='HLA'))
m = list(result_mutate.groupby(by='HLA'))
for i,ax in enumerate(axes.flatten()):
    ax.plot(np.arange(9)+1,n[i][1]['immunogenicity'][::-1],label='normal',marker='v',alpha=0.5)
    ax.plot(np.arange(9)+1,m[i][1]['immunogenicity'][::-1],label='mutate',marker='o',linestyle='--')
    ax.legend()
    ax.set_title(n[i][0])
plt.savefig('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/D614G/lineplot.pdf',bbox_inches='tight')

# N501Y mutation
region = spike[492:509]
mer9_normal = [region[i:i+9] for i in range(0,9,1)]

mutate = region[0:8] + 'Y' + region[9:]
mer9_mutate = [mutate[i:i+9] for i in range(0,9,1)]

def set_query_df(frag):
    from itertools import product
    hla = ['HLA-A*0101','HLA-A*0201','HLA-A*0301','HLA-A*1101','HLA-A*2402','HLA-B*0702','HLA-B*0801','HLA-B*1501','HLA-B*4001','HLA-C*0702']
    combine = list(product(frag,hla))
    col1 = [item[0] for item in combine]  # peptide
    col2 = [item[1] for item in combine]  # hla
    df = pd.DataFrame({'peptide':col1,'HLA':col2})
    return df

set_query_df(mer9_normal).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/N501Y/N501Y_normal.csv',index=None,
                                 header=None)
set_query_df(mer9_mutate).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/N501Y/N501Y_mutate.csv',index=None,
                                 header=None)

result_normal = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/N501Y/normal_result.txt',sep='\t')
result_mutate = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/N501Y/mutate_result.txt',sep='\t')

# plot by each HLA
fig,axes = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw={'hspace':0.5})
n = list(result_normal.groupby(by='HLA'))
m = list(result_mutate.groupby(by='HLA'))
for i,ax in enumerate(axes.flatten()):
    ax.plot(np.arange(9)+1,n[i][1]['immunogenicity'][::-1],label='normal',marker='v',alpha=0.5)
    ax.plot(np.arange(9)+1,m[i][1]['immunogenicity'][::-1],label='mutate',marker='o',linestyle='--')
    ax.legend()
    ax.set_title(n[i][0])
plt.savefig('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/N501Y/lineplot.pdf',bbox_inches='tight')

# E484K
region = spike[475:492]
mer9_normal = [region[i:i+9] for i in range(0,9,1)]

mutate = region[0:8] + 'K' + region[9:]
mer9_mutate = [mutate[i:i+9] for i in range(0,9,1)]

def set_query_df(frag):
    from itertools import product
    hla = ['HLA-A*0101','HLA-A*0201','HLA-A*0301','HLA-A*1101','HLA-A*2402','HLA-B*0702','HLA-B*0801','HLA-B*1501','HLA-B*4001','HLA-C*0702']
    combine = list(product(frag,hla))
    col1 = [item[0] for item in combine]  # peptide
    col2 = [item[1] for item in combine]  # hla
    df = pd.DataFrame({'peptide':col1,'HLA':col2})
    return df

set_query_df(mer9_normal).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/E484K/E484K_normal.csv',index=None,
                                 header=None)
set_query_df(mer9_mutate).to_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/E484K/E484K_mutate.csv',index=None,
                                 header=None)

result_normal = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/E484K/normal_result.txt',sep='\t')
result_mutate = pd.read_csv('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/E484K/mutate_result.txt',sep='\t')

# plot by each HLA
fig,axes = plt.subplots(nrows=5,ncols=2,figsize=(10,10),gridspec_kw={'hspace':0.5})
n = list(result_normal.groupby(by='HLA'))
m = list(result_mutate.groupby(by='HLA'))
for i,ax in enumerate(axes.flatten()):
    ax.plot(np.arange(9)+1,n[i][1]['immunogenicity'][::-1],label='normal',marker='v',alpha=0.5)
    ax.plot(np.arange(9)+1,m[i][1]['immunogenicity'][::-1],label='mutate',marker='o',linestyle='--')
    ax.legend()
    ax.set_title(n[i][0])
plt.savefig('/Users/ligk2e/Desktop/github/DeepImmuno/files/variants/E484K/lineplot.pdf',bbox_inches='tight')

