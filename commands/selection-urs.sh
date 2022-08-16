#!/usr/bin/bash
#SBATCH -c 4
#SBATCH --mem 4G
source ~/.bashrc
conda activate rs

DATA_DIR=$1
K=$2
INIT=$3
LD=$4
MU=$5
BEGIN=$6
END=$7

python -u selection.py -i $DATA_DIR/also_bought.txt -s $DATA_DIR/sentiment.txt -k $K -o $DATA_DIR/result/unified-integer-regression-rs-$K-init-$INIT-$K-ld-$LD-mu-$MU -a unified-integer-regression-rs -is $DATA_DIR/result/$INIT-$K -ld $LD -mu $MU -t $BEGIN-$END # --skip_exists
