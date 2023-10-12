#!/usr/bin/env bash

source $HOME/.bashrc
#conda activate GraphMVP


echo $@
date
echo "start"
python pretrain_holimol3d_dihedral2.py $@
echo "end"
date
