# Holistic Molecular Representation Leraning via Multi-view Fragmentation (TMLR 2024)

Official Pytorch implementation of ["Holistic Molecular Representation Learning via Multi-view Fragmentation"](https://openreview.net/forum?id=ufDh55J1ML) by [Seojin Kim](https://seojin-kim.github.io), [Jaehyun Nam](https://jaehyun513.github.io/), [Junsu Kim](https://junsu-kim97.github.io), [Hankook Lee](https://hankook.github.io/), [Sungsoo Ahn](https://sites.google.com/view/sungsooahn0215/home), and [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html).

**TL;DR**: We propose a molecular contrastive learning framework that utilizies fragment-wise feature of molecules.

**TODO**: Video description will be uploaded soon.

## 1. Dataset Preparation
```
python GEOM_dataset_preparation.py --n_mol 50000 --data_foler ../datasets/path_to_dataset
```

## 2. Training
```
cd scripts_classification/
bash submit_pretraining_holimol_dihedral2.sh
```

## Citation
```bibtex
@article{
kim2024holistic,
title={Holistic Molecular Representation Learning via Multi-view Fragmentation},
author={Seojin Kim and Jaehyun Nam and Junsu Kim and Hankook Lee and Sungsoo Ahn and Jinwoo Shin},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=ufDh55J1ML},
note={}
}
```
