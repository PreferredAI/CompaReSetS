import os
import gurobipy as gp
from gurobipy import GRB
from selection import read_data
import numpy as np

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/cellphone/result/unified-integer-regression-rs-3-init-integer-regression-rs-3-ld-1-ld-1-mu-0.1",
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
        "-tl",
        "--time_limit",
        type=int,
        default=300,
        help="Time limit for solver to solve a problem instance",
    )
    parser.add_argument(
        "-nt",
        "--num_threads",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="Target indices of data to perform selection",
    )
    parser.add_argument("--skip_optimal", action="store_true")
    args = parser.parse_args()
    print(args)
    return args

def read_edges(edges_path):
    edges = {}
    with open(edges_path, 'r') as f:
        for l in f:
            tokens = l.strip().split()
            i = int(tokens[0])
            j = int(tokens[1])
            w = float(tokens[2])
            edges[(i, j)] = w
    return edges

def solve_hks_target(edges, k, time_limit=300):
    n_nodes = max([j for (_, j) in edges.keys()]) + 1
    # maximize: indicator_i * indicator_j * w_ij, st. sum(indicator) == k, indicator_0 = 1
    m = gp.Model("hks_target")
    m.setParam('TimeLimit', time_limit)
    # create variables
    # v_nodes = m.addVars([i for i in range(n_nodes)], vtype=GRB.BINARY)
    v_nodes = [m.addVar(vtype=GRB.BINARY, name="v_{}".format(i)) for i in range(n_nodes)]

    obj = gp.LinExpr()
    for i in range(n_nodes - 1):
        for j in range(i+1, n_nodes):
            obj += v_nodes[i] * v_nodes[j] * edges[(i, j)]
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(v_nodes[0] == 1, "c0")
    expr = gp.LinExpr()
    for i in range(n_nodes):
        expr += v_nodes[i]
    m.addConstr(expr == min(k, n_nodes), "c1")
    m.optimize()
    selected = []
    for i in range(n_nodes):
        if np.round(v_nodes[i].X, 0) == 1.0:
            selected.append(i)
    return selected, m.status

def execute(input_dir, target_item, k, time_limit=300, skip_optimal=True):
    result_dir = os.path.join(input_dir, '/'.join(target_item))
    result_path = os.path.join(result_dir, '{}.txt'.format(target_item))
    edges_path = os.path.join(result_dir, 'edges.txt')
    output_path = os.path.join(result_dir, 'nodes_top_{}_gurobi.txt'.format(k))
    edges = read_edges(edges_path)
    if os.path.exists(output_path) and skip_optimal:
        n_nodes = max(j for (i, j), w in edges.items()) + 1
        with open(output_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                if int(lines[1]) == 2 and len(lines[0].strip().split()) == min(n_nodes, k):
                    return

    selected, status = solve_hks_target(edges, k, time_limit)
    with open(output_path, 'w') as f:
        f.write(' '.join([str(i) for i in selected]))
        f.write('\n{}'.format(status))

if __name__ == '__main__':
    from tqdm import tqdm
    from joblib import Parallel, delayed
    args = parse_arguments()
    data = read_data(args.data_path)
    begin_index = 0
    end_index = len(data)
    if args.target is not None:
        tokens = args.target.split('-')
        begin_index = int(tokens[0])
        end_index = int(tokens[1])

    for inc, (target_item, comparison_items) in enumerate(data.items()):
        if inc >= begin_index and inc <= end_index:
            execute(args.input_dir, target_item, args.k)
    # Parallel(n_jobs=(args.num_threads), prefer="threads", backend="multiprocessing", verbose=100)(
    #     (
    #         delayed(execute)(
    #             args.input_dir,
    #             target_item,
    #             args.k,
    #             args.time_limit,
    #             args.skip_optimal
    #         )
    #         for inc, (target_item, comparison_items) in enumerate(data.items())
    #         if inc >= begin_index and inc <= end_index
    #     )
    # )
