#!/usr/bin/bash
#SBATCH -c 2
#SBATCH --mem 4G
source ~/.bashrc
conda activate rs

DATA_DIR=$1
K=$2
ALGORITHM=$3
BEGIN=$4
END=$5

python -u selection.py -i $DATA_DIR/also_bought.txt -s $DATA_DIR/sentiment.txt -k $K -o $DATA_DIR/result/$ALGORITHM-$K -a $ALGORITHM -t $BEGIN-$END #--skip_exists
