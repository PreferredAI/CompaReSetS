#!/usr/bin/bash
#SBATCH -c 2
#SBATCH --mem 4G
source ~/.bashrc
conda activate rs

DATA_DIR=$1
K=$2
INIT=$3
MAX_ITER=$4
BEGIN=$5
END=$6

python -u selection.py -i $DATA_DIR/also_bought.txt -s $DATA_DIR/sentiment.txt -k $K -o $DATA_DIR/result/unified-integer-regression-rs1-$K-init-$INIT-$K-e-$MAX_ITER -a unified-integer-regression-rs1 -is $DATA_DIR/result/$INIT-$K -mi $MAX_ITER -t $BEGIN-$END --skip_exists
