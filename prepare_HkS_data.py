import pandas as pd
import os
from collections import OrderedDict
from selection import read_data, read_sentiment, get_item_aspect_sentiments, get_aspect_id_map, get_aspect_opinion_vectors, get_aspect_sentiments, distance
EPS = 1e-9


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/cellphone/result/unified-integer-regression-rs-3-init-integer-regression-rs-3-ld-1-mu-0.1",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="data/cellphone/also_bought.txt"
    )
    parser.add_argument(
        "-s", "--sentiment_path", type=str, default=None,
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
        help="Maximum number of reviews for in selection set",
    )
    parser.add_argument(
        "-ld",
        "--ld",
        type=float,
        default=1.0,
        help="Traceoff factor of aspect distance and opinion distance",
    )
    parser.add_argument("-mu", "--mu", type=float, default=1.0)
    parser.add_argument(
        "-wt",
        "--weight_type",
        type=str,
        default=""
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="Target indices of data to perform selection",
    )
    args = parser.parse_args()
    print(args)
    return args


def export_weighted_edges(sentiment, input_dir, target_item, k, ld=1.0, mu=1.0, weight_type='aspect_only'):
    result_dir = os.path.join(input_dir, "/".join(target_item))
    result_path = os.path.join(result_dir, '{}.txt'.format(target_item))
    result = OrderedDict()
    with open(result_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            result[tokens[0]] = tokens[1:]
    aspect_id_map = get_aspect_id_map(sentiment, list(result.keys()))
    target_aspect_vector, _ = get_aspect_opinion_vectors(
        aspect_id_map, get_item_aspect_sentiments(sentiment, target_item)
    )
    aspect_vectors = []
    opinion_vectors = []
    target_opinion_vectors = []
    for inc, (item, selected_reviewers) in enumerate(result.items()):
        _, target_opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map, get_item_aspect_sentiments(sentiment, item)
        )
        target_opinion_vectors.append(target_opinion_vector)
        aspect_vector, opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map, get_aspect_sentiments(sentiment, item, selected_reviewers)
        )
        aspect_vectors.append(aspect_vector)
        opinion_vectors.append(opinion_vector)
    # compute distance
    pairs = []
    distances = []
    all_items = list(result.keys())
    d_opinions = [
        distance(target_opinion_vector, opinion_vectors[i])
        for i in range(len(all_items))
    ]
    d_aspects = [
        ld * ld * distance(target_aspect_vector, aspect_vectors[i])
        for i in range(len(all_items))
    ]
    for i in range(len(all_items) - 1):
        for j in range(i+1, len(all_items)):
            pairs.append((i, j))
            d_ij = distance(aspect_vectors[i], aspect_vectors[j]) if weight_type =='aspect_only' else (
                d_opinions[i]
                + d_opinions[j]
                + d_aspects[i]
                + d_aspects[j]
                + mu * mu * distance(aspect_vectors[i], aspect_vectors[j])
            )
            distances.append(d_ij)
    max_distance = max(distances) + EPS
    weights = [max_distance - d for d in distances]
    ext = '_aspect_only' if weight_type == 'aspect_only' else ''
    with open(os.path.join(result_dir, 'edges{}.txt'.format(ext)), 'w') as f:
        for (i, j), w in zip(pairs, weights):
            f.write('{} {} {}\n'.format(i, j, w))


if __name__ == '__main__':
    from joblib import Parallel, delayed
    args = parse_arguments()
    data = read_data(args.data_path)
    sentiment = read_sentiment(args.sentiment_path)
    begin_index = 0
    end_index = len(data)
    if args.target is not None:
        tokens = args.target.split('-')
        begin_index = int(tokens[0])
        if int(tokens[1]) < end_index:
            end_index = int(tokens[1])
    Parallel(n_jobs=(-1), prefer="threads", backend="multiprocessing", verbose=100)(
        (
            delayed(export_weighted_edges)(
                sentiment,
                args.input_dir,
                target_item,
                args.k,
                args.ld,
                args.mu,
                args.weight_type
            )
            for inc, (target_item, comparison_items) in enumerate(data.items())
            if inc >= begin_index and inc <= end_index
        )
    )
