#!/usr/bin/bash
#SBATCH -c 2
#SBATCH --mem 8G
source ~/.bashrc
conda activate rs

INPUT_DIR=$1
DATA_DIR=$2
BEGIN=$3
END=$4

python -u evaluate_text_similarity.py -i $INPUT_DIR -d $DATA_DIR/also_bought.txt -r $DATA_DIR/review.txt -t $BEGIN-$END #--skip_exists
