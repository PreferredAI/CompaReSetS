#!/usr/bin/bash
#SBATCH -c 4
#SBATCH --mem 2G
source ~/.bashrc
conda activate rs

INPUT_DIR=$1
DATA_DIR=$2
K=$3
LD=$4
MU=$5
BEGIN=$6
END=$7

python -u prepare_HkS_data.py -i $INPUT_DIR -d $DATA_DIR/also_bought.txt -s $DATA_DIR/sentiment.txt -k $K -ld $LD -mu $MU -t $BEGIN-$END
