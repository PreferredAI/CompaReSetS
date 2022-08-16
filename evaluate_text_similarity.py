import os
from collections import OrderedDict
from selection import read_data, read_review, read_sentiment
from rouge import rouge

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="data/cellphone/also_bought.txt",
        help="Input item sequences, each line contains 'target_item,item1,item2,...'",
    )
    parser.add_argument(
        "-r", "--review_path", type=str, default=None, help="Review path"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="Target indices of data to perform selection",
    )
    parser.add_argument(
        "-mi",
        "--max_iter",
        type=int,
        default=1,
        help="Maximum number of iteration for alternating integer regression approach",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/cellphone/result/greedy-rs",
        help="Ouput path",
    )
    parser.add_argument(
        "-rs", "--random_seed", type=int, default=123, help="Random seed value"
    )
    parser.add_argument("--skip_exists", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(args)
    return args


def evaluate_and_export(review, target_item, input_dir, skip_exists=False):
    result_dir = os.path.join(input_dir, '/'.join(str(target_item)))
    result_path = os.path.join(result_dir, '{}.txt'.format(target_item))
    output_path = os.path.join(result_dir, 'evaluation.csv')
    if skip_exists and os.path.exists(output_path):
        return
    results = [] 
    with open(result_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            results.append(tokens)

    rouge_scores = OrderedDict()
    for i in range(len(results) - 1):
        item_i = results[i][0]
        item_i_reviewers = results[i][1:]
        item_i_reviews = [review.get(item_i, {}).get(reviewer, '') for reviewer in item_i_reviewers]
        for j in range(i + 1, len(results)):
            item_j = results[j][0]
            item_j_reviewers = results[j][1:]
            item_j_reviews = [review.get(item_j, {}).get(reviewer, '') for reviewer in item_j_reviewers]
            batch_i_reviews = [
                i_review
                for i_review in item_i_reviews
                for _ in range(len(item_j_reviews))
            ]
            batch_j_reviews = [
                j_review
                for _ in range(len(item_i_reviews))
                for j_review in item_j_reviews
            ]
            rouge_score = rouge(batch_i_reviews, batch_j_reviews)
            rouge_scores[(item_i, item_j)] = rouge_score
    with open(output_path, 'w') as f:
        f.write('item_i,item_j,rouge_1,rouge_2,rouge_l\n')
        for (item_i, item_j), rouge_score in rouge_scores.items():
            f.write('{},{},{},{},{}\n'.format(
                item_i,
                item_j,
                rouge_score['rouge_1/f_score'],
                rouge_score['rouge_2/f_score'],
                rouge_score['rouge_l/f_score'],
            ))

if __name__ == '__main__':
    from joblib import Parallel, delayed
    args = parse_arguments()
    data = read_data(args.data_path)
    review = read_review(args.review_path)
    begin_index = 0
    end_index = len(data)
    if args.target is not None:
        tokens = args.target.split("-")
        begin_index = int(tokens[0])
        end_index = int(tokens[1]) if int(tokens[1]) < end_index else end_index
    
    Parallel(n_jobs=(-1), prefer="threads", verbose=100)(
        (
            delayed(evaluate_and_export)(
                review,
                target_item,
                args.input_dir,
                args.skip_exists
            )
            for inc, (target_item, _) in enumerate(data.items())
            if inc >= begin_index and inc <= end_index
        )
    )