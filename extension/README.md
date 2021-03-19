## Train your own GAN model

GAN in theory can generate sequences that meet user-defined features. For instances, you can generate immunogenic sequences that specifically bind to certain HLA (i.e. HLA-A*2402). To accommodate this need, we encapsulate the training codes and make it fully open-source.

## How to run?

- Step1: prepare a csv file with two column (col1: peptide, col2: HLA), an example file is `gan_a0201.csv` in this folder.
- Step2: run the folowing command
```shell
python deepimmuno-gan-train.py --data gan_a0201.csv --outdir ./gan_result --epoch 3
```

The full help prompt is as below:

```shell
$ ​python deepimmuno-gan-train.py --help
usage: deepimmuno-gan-train.py [-h] [--data DATA] [--outdir OUTDIR]
                               [--epoch EPOCH]

DeepImmuno-GAN train your own

optional arguments:
  -h, --help       show this help message and exit
  --data DATA      path to your training data file, a csv file
  --outdir OUTDIR  path to your output folder
  --epoch EPOCH    how many epochs you want to run
```

## What result you can get?

1. The `generated sequences` using the GAN model trained at the epoch number you provided. (default: 1024 peptides)

2. A `diagnosic plot`. `D_loss` is the loss of discriminator, `G_loss` is the loss of generator.

3. The `model` trained on the epoch you provided.

The examples can be found in `./gan_result` folder


## Note

- In addtion to the basic dependencies shown in main [README.md](https://github.com/frankligy/DeepImmuno) file, you probably need `matplotlib` as well to draw the diagnostic plots.

- We hard-coded the `batch_size=64`, so we hope your training data is at least more than 64 instances. Having said that, you can change this setting at your will.

- we advise to set the `epoch` as 100 or more to assure a good training performance.

- Every 50 epoch, an intermediate file with `generated sequences` will be available in the output folder, it can help you to track the performance improvement as training goes on.

## Contact

Guangyuan(Frank) Li

li2g2@mail.uc.edu

PhD student, Biomedical Informatics

Cincinnati Children's Hospital Medical Center(CCHMC)

University of Cincinnati, College of Medicine
