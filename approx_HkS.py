import pandas as pd
import os
from collections import OrderedDict
from selection import read_data, read_sentiment, get_item_aspect_sentiments, get_aspect_id_map, get_aspect_opinion_vectors, get_aspect_sentiments, distance, rs_random, rs_iterative_random
from subprocess import call

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
        "-k",
        "--k",
        type=int,
        default=3,
        help="Number of items",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=[
            "rank",
            "iterative-rank",
            "random",
            "iterative-random"
        ],
        default="rank",
    )
    parser.add_argument(
        "-wt",
        "--weight_type",
        type=str,
        default="",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="Target indices of data to perform selection",
    )
    parser.add_argument(
        "-rs", "--random_seed", type=int, default=123, help="Random seed value"
    )
    args = parser.parse_args()
    print(args)
    return args

def read_edges(path):
    edges = []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            edges.append((int(tokens[0]), int(tokens[1]), float(tokens[2])))
    return edges

def execute(input_dir, target_item, k, algorithm="rank", weight_type="aspect_only", seed=None):
    result_dir = os.path.join(input_dir, '/'.join(target_item))
    result_path = os.path.join(result_dir, '{}.txt'.format(target_item))
    ext = "_aspect_only" if weight_type == "aspect_only" else ""
    edges_path = os.path.join(result_dir, 'edges{}.txt'.format(ext))
    output_path = os.path.join(result_dir, 'nodes_top_{}_{}{}.txt'.format(k, algorithm, ext))
    edges = read_edges(edges_path)
    if algorithm == "rank":
        # get weights
        t_edges = []
        for e in edges:
            # get edges including target item only
            if e[0] == 0:
                t_edges.append(e)
        # sort descending
        t_edges.sort(key=lambda y: y[2], reverse=True)
        with open(output_path, 'w') as f:
            f.write('0 {}'.format(' '.join(str(s[1]) for s in t_edges[:k-1])))
    elif algorithm == "iterative-rank":
        weight_by_node = {}
        for e in edges:
            adjacent_nodes = weight_by_node.setdefault(e[0], {})
            adjacent_nodes1 = weight_by_node.setdefault(e[1], {})
            adjacent_nodes[e[1]] = e[2]
            adjacent_nodes1[e[0]] = e[2]
        solution = [0]
        nodes = [node for node in weight_by_node.keys() if node != 0]
        for _ in range(min(k-1, len(nodes))):
            best_node = None
            best_weight = 0
            for node in nodes:
                if node in solution:
                    continue
                weight = 0
                for selected_node in solution:
                    weight += weight_by_node[node][selected_node]
                if weight >= best_weight:
                    best_weight = weight
                    best_node = node
            solution.append(best_node)
        with open(output_path, 'w') as f:
            f.write(' '.join(str(s) for s in solution))
    elif algorithm == "random":
        nodes = []
        for e in edges:
            if e[1] not in nodes:
                nodes.append(e[1])
        solution = [0] + rs_random(nodes, k-1, seed)
        with open(output_path, 'w') as f:
            f.write(' '.join(str(s) for s in solution))
    elif algorithm == "iterative-random":
        nodes = []
        for e in edges:
            if e[1] not in nodes:
                nodes.append(e[1])
        solution = [0] + rs_iterative_random(nodes, k-1)
        with open(output_path, 'w') as f:
            f.write(' '.join(str(s) for s in solution))

if __name__ == '__main__':
    from joblib import Parallel, delayed
    args = parse_arguments()
    data = read_data(args.data_path)
    begin_index = 0
    end_index = len(data)
    if args.target is not None:
        tokens = args.target.split('-')
        begin_index = int(tokens[0])
        if int(tokens[1]) < end_index:
            end_index = int(tokens[1])
    Parallel(n_jobs=(-1), prefer="threads", backend="multiprocessing", verbose=100)(
        (
            delayed(execute)(
                args.input_dir,
                target_item,
                args.k,
                args.algorithm,
                args.weight_type,
                seed=args.random_seed,
            )
            for inc, (target_item, comparison_items) in enumerate(data.items())
            if inc >= begin_index and inc <= end_index
        )
    )
