# DeepImmuno
Deep-learning empowered prediction and generation of immunogenic epitopes for T cell immunity. 
> Please refer to **DeepImmuno-CNN** if you want to predict immunogenicity
> Plase refer to **DeepImmuno-GAN** if you want to generate immunogenic peptide

Enjoy and don't hesitate to ask me questions (contact at the bottom), I will be responsive!

## DeepImmuno-CNN

#### Dependencies
python = 3.6
tensorflow = 2.3.0
numpy = 1.18.5
pandas = 1.1.1

#### How to use?

If you want to query a single epitope (peptide + HLA), for example you want to query peptide _**HPPLMNVER**_ along with _**HLA-A*0201**_. You need to

```
python3 deepimmuno-cnn.py --mode "single" --epitope "HPPLMNVER" --HLA "HLA-A*0201"
```

If you want to query multiple epitopes, you just need to prepare a csv file like this:

```
AAAAAAAAA,HLA-A*0201
CCCCCCCCC,HLA-B*5801
DDDDDDDDD,HLA-C*0702
```

Then you run:

```
python3 deepimmuno-cnn.py --mode "multiple" --intdir "/path/to/above/file" --outdir "/path/to/output/folder"
```

A full help prompt is as below:

```
usage: deepimmuno-cnn.py [-h] [--mode MODE] [--epitope EPITOPE] [--hla HLA]
                         [--intdir INTDIR] [--outdir OUTDIR]

DeepImmuno-CNN command line

optional arguments:
  -h, --help         show this help message and exit
  --mode MODE        single mode or multiple mode
  --epitope EPITOPE  if single mode, specifying your epitope
  --hla HLA          if single mode, specifying your HLA allele
  --intdir INTDIR    if multiple mode, specifying the path to your input file
  --outdir OUTDIR    if multiple mode, specifying the path to your output folder
```

## DeepImmuno-GAN

#### Dependencies

python = 3.6
pytorch = 1.4.0
numpy = 1.18.4
pandas = 1.0.5

#### How to use

Pretty simple, just run like this

```
python3 deepimmuno-gan.py --outdir "/path/to/store/output"
```

It will automatically genearte one batch, which is **64** pseudo-immunogenic peptides of **HLA-A*0201** for your. It is worth noting that, because of the way I encode the peptide, there will be a placeholder "-". 

A full help prompt is as below
```
usage: deepimmuno-gan.py [-h] [--outdir OUTDIR]

DeepImmuno-GAN to generate immunogenic peptide

optional arguments:
  -h, --help       show this help message and exit
  --outdir OUTDIR  specifying your output folder
```

## Contact

Guangyuan(Frank) Li

li2g2@mail.uc.edu

PhD student, Biomedical Informatics

Cincinnati Children's Hospital Medical Center(CCHMC)

University of Cincinnati, College of Medicine





