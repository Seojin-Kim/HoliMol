# Fragment-based Multi-view Molecular Contrastive Learning (ICLR 2023 ML4Materials Workshop)

Official Pytorch implementation of ["Fragment-based Multi-view Molecular Contrastive Learning"](https://openreview.net/forum?id=9lGwd4q8KJc) by [Seojin Kim](https://seojin-kim.github.io), [Jaehyun Nam](https://jaehyun513.github.io/), [Junsu Kim](https://junsu-kim97.github.io), [Hankook Lee](https://hankook.github.io/), [Sungsoo Ahn](https://sites.google.com/view/sungsooahn0215/home), and [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html).

**TL;DR**: We propose a molecular contrastive learning framework that utilizies fragment-wise feature of molecules.

## 1. Dataset Preparation
```
python GEOM_dataset_preparation.py --n_mol 50000
```

## 2. Training
```
cd scripts_classification/
bash submit_pretraining_holimol_dihedral2.sh
```

## Citation
```bibtex
@inproceedings{kim2023fragment,
  title={Fragment-based Multi-view Molecular Contrastive Learning},
  author={Kim, Seojin and Nam, Jaehyun and Kim, Junsu and Lee, Hankook and Ahn, Sungsoo and Shin, Jinwoo},
  booktitle={Workshop on''Machine Learning for Materials''ICLR 2023},
  year={2023}
}
```
