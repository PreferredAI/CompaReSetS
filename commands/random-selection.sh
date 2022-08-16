#!/usr/bin/bash
#SBATCH -c 2
#SBATCH --mem 4G
source ~/.bashrc
conda activate rs

DATA_DIR=$1
K=$2
SEED=$3

python -u selection.py -i $DATA_DIR/also_bought.txt -s $DATA_DIR/sentiment.txt -k $K -o $DATA_DIR/result/random-$K/$SEED -a random -rs $SEED --skip_exists
