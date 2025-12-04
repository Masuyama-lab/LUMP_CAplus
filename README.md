# A Clustering-based Sample Selection Method for Improving Replay Buffer Quality in Continual Self-Supervised Learning

This repository contains the implementation of the proposed algorithm in the following paper:

**N. Masuyama, R. Fujii, T. Kinoshita, and Y. Nojima, "A clustering-based sample selection method for improving replay buffer quality in continual self-supervised learning," <I>in Proc. of the 2025 International Joint Conference on Neural Networks</I> (IJCNN), pp. 1-7, Roma, Italy, June 30 - July 5, 2025.**


## Installation
```
$ pip install -r requirement.txt
```

## Run
### ID Condition
E.g. Running LUMP+CA+ (SimSiam) on CIFAR-10.
```
$ python main_LUMP_clustering.py
```
E.g. Running LUMP+CA+ (BarlowTwins) on CIFAR-10.
```
$ python main_LUMP_clustering.py -c configs/barlow_c10.yaml
```
E.g. Running LUMP+CA+ (SimSiam) on CIFAR-100.
```
$ python main_LUMP_clustering.py -c configs/simsiam_c100.yaml
```

The training configurations can be found in the `./configs` .  
Running LUMP+KM, set `clustering: kmeans` in the configuration file.

The results (accuracy and forgetting) and checkpoints are saved in the `./logs` and `./checkpoints`.

### OOD Condition
After running the ID condition experiment, you can run OOD condition experiments.

E.g.
- Method : LUMP+CA+ (SimSiam)
- Training data : TinyImageNet,
- Test data : MNIST

```
$ python main_LUMP_clustering_OOD.py -c configs/simsiam_tinyimagenet.yaml --trained_model_dir ./checkpoints/lump+caplus_simsiam_seq-tinyimg/ --ood_data_name seq-mnist 
```
Instead of the command, you can set arguments in `arguments.py`.

## Acknowledgement
This code is built upon the publicly available implementation by [divyam3897/UCL](https://github.com/divyam3897/UCL).
We gratefully acknowledge the author for making their code publicly available.
