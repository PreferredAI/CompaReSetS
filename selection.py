from collections import OrderedDict, defaultdict
import numpy as np, os

np.seterr(all="raise")
from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls
from omp import omp


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="data/cellphone/also_bought.txt",
        help="Input item sequences, each line contains 'target_item,item1,item2,...'",
    )
    parser.add_argument(
        "-s", "--sentiment_path", type=str, default=None, help="Sentiment path"
    )
    parser.add_argument(
        "-is",
        "--initial_selection_dir",
        type=str,
        default=None,
        help="Initial selection file directory, solution from other algorithm",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
        help="Maximum number of reviews for in selection set",
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
        "-o",
        "--out",
        type=str,
        default="data/cellphone/result/greedy-rs",
        help="Ouput path",
    )
    parser.add_argument(
        "-ot",
        "--opinion_type",
        choices=[
            "binary",
            "polarity",
            "scale",
        ],
        default="binary",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=[
            "greedy-crs",
            "integer-regression-crs",
            "random",
            "iterative-random",
            "greedy-rs",
            "integer-regression-rs",
            "unified-greedy-rs",
            "unified-integer-regression-rs",
            "unified-integer-regression-rs1",
        ],
        default="greedy-rs",
    )
    parser.add_argument(
        "-ld",
        "--ld",
        type=float,
        default=1.0,
        help="Tradeoff factor of aspect distance and opinion distance",
    )
    parser.add_argument("-mu", "--mu", type=float, default=0.1)
    parser.add_argument(
        "-rs", "--random_seed", type=int, default=123, help="Random seed value"
    )
    parser.add_argument("--skip_exists", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print(args)
    return args


def read_data(data_path):
    data = OrderedDict()
    with open(data_path, "r") as f:
        for line in f:
            tokens = line.strip().split(",")
            target_item, comparison_items = tokens[0], tokens[1:]
            data[str(target_item)] = comparison_items

    return data


def read_review(review_path):
    review = {}
    with open(review_path, "r") as f:
        for l in f:
            tokens = l.strip().split("\t")
            if len(tokens) > 2:
                reviewer, item, reviewText = tokens[0], tokens[1], tokens[2]
                item_reviews = review.setdefault(item, {})
                item_reviews[reviewer] = reviewText

    return review


def get_review(review, item, reviewer):
    return review.get(item, {}).get(reviewer, "")


def read_sentiment(sentiment_path):
    sentiment = {}
    with open(sentiment_path, "r") as f:
        for l in f:
            tokens = l.strip().split(",")
            reviewer, item, sentiment_tup = tokens[0], tokens[1], tokens[2:]
            item_sentiments = sentiment.setdefault(item, {})
            item_sentiments[reviewer] = [
                (tup.split(":")[0], float(tup.split(":")[2])) for tup in sentiment_tup
            ]

    return sentiment


def read_selection(selection_path):
    res = {}
    with open(selection_path, "r") as f:
        for l in f:
            tokens = l.strip().split(",")
            res[tokens[0]] = tokens[1:]

    return res


def get_item_aspects(sentiment, item):
    return [
        aspect
        for _, aspect_sentiments in sentiment.get(item, {}).items()
        for aspect, _ in iter(aspect_sentiments)
    ]


def get_item_aspect_sentiments(sentiment, item):
    return [
        (aspect, score)
        for _, aspect_sentiments in sentiment.get(item, {}).items()
        for aspect, score in iter(aspect_sentiments)
    ]


def get_aspects(sentiment, item, user, aspects=[]):
    return aspects + [aspect for aspect, _ in sentiment.get(item, {}).get(user, [])]


def get_aspect_sentiments(sentiment, item, users=[], aspect_sentiments=[]):
    res = aspect_sentiments.copy()
    for user in users:
        _aspect_sentiments = sentiment.get(item, {}).get(user, [])
        res = res + _aspect_sentiments

    return res


def get_aspect_id_map(sentiment, items):
    aspect_id_map = OrderedDict()
    for item in items:
        for _, aspect_sentiments in sentiment.get(item, {}).items():
            for aspect, _ in aspect_sentiments:
                aspect_id_map.setdefault(aspect, len(aspect_id_map))

    return aspect_id_map


def get_aspect_opinion_vectors(aspect_id_map, aspect_sentiments, opinion_type="binary"):
    aspect_vector = np.zeros((len(aspect_id_map)), dtype=(np.float64))
    if opinion_type == "binary":  # positive, negative
        opinion_vector = np.zeros((2 * len(aspect_id_map)), dtype=(np.float64))
        for aspect, score in aspect_sentiments:
            if score > 0:
                opinion_vector[2 * aspect_id_map[aspect]] += score
            elif score < 0:
                opinion_vector[2 * aspect_id_map[aspect] + 1] += score
            aspect_vector[aspect_id_map[aspect]] += 1
        scale_factor = max(np.max(aspect_vector), 1)
        opinion_vector = opinion_vector / scale_factor
        aspect_vector = aspect_vector / scale_factor
    elif opinion_type == "polarity":  # positive, negative, neutral
        opinion_vector = np.zeros((3 * len(aspect_id_map)), dtype=(np.float64))
        for aspect, score in aspect_sentiments:
            if score > 0:
                opinion_vector[3 * aspect_id_map[aspect]] += score
            elif score < 0:
                opinion_vector[3 * aspect_id_map[aspect] + 1] += score
            else:
                opinion_vector[3 * aspect_id_map[aspect] + 2] += score
            aspect_vector[aspect_id_map[aspect]] += 1
        scale_factor = max(np.max(aspect_vector), 1)
        opinion_vector = opinion_vector / scale_factor
        aspect_vector = aspect_vector / scale_factor
    elif opinion_type == "scale":  # opinion = sigmoid(total_sentiment)
        opinion_vector = np.zeros((len(aspect_id_map)), dtype=(np.float64))
        aggregated_sentiments = {}
        for aspect, score in aspect_sentiments:
            aggregated_sentiments.setdefault(aspect, 0)
            aggregated_sentiments[aspect] += score
            aspect_vector[aspect_id_map[aspect]] += 1
        for aspect, total_sentiment in aggregated_sentiments.items():
            opinion_vector[aspect_id_map[aspect]] = 1.0 / (1 + np.exp(-total_sentiment))
        scale_factor = max(np.max(aspect_vector), 1)
        aspect_vector = aspect_vector / scale_factor
    return (aspect_vector, opinion_vector)


def distance(a, b):
    return (
        (np.array(a, dtype=(np.float64)) - np.array(b, dtype=(np.float64))) ** 2
    ).sum()


def rs_random(choices, k, seed=123):
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(choices, min(k, len(choices)), replace=False).tolist())


def rs_iterative_random(reviewers, k):
    selected = []
    remaining_reviewers = reviewers.copy()
    for _ in range(min(k, len(reviewers))):
        selected_reviewer = np.random.choice(remaining_reviewers)
        selected.append(selected_reviewer)
        remaining_reviewers.remove(selected_reviewer)

    return selected


def crs_greedy(
    sentiment,
    item,
    reviewers,
    k,
    opinion_type="binary",
):
    """
    item: item_id
    reviewers: user ids that wrote review on item, for retrieve review/sentiment purpose
    """
    aspect_id_map = get_aspect_id_map(sentiment, [item])
    _, target_opinion_vector = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, item),
        opinion_type=opinion_type,
    )
    selected = []
    current_aspect_sentiments = []
    remaining_reviewers = reviewers.copy()
    for _ in range(k):
        selected_reviewer = None
        min_cost = 1e9
        for reviewer in remaining_reviewers:
            t_aspect_sentiments = get_aspect_sentiments(
                sentiment, item, [reviewer], current_aspect_sentiments
            )
            _, t_opinion_vector = get_aspect_opinion_vectors(
                aspect_id_map,
                t_aspect_sentiments,
                opinion_type=opinion_type,
            )
            d_opinion = distance(target_opinion_vector, t_opinion_vector)
            if d_opinion < min_cost:
                selected_reviewer = reviewer
                min_cost = d_opinion
                current_aspect_sentiments = t_aspect_sentiments

        if selected_reviewer:
            selected.append(selected_reviewer)
            remaining_reviewers.remove(selected_reviewer)

    return selected


def rs_greedy(
    sentiment,
    target_item,
    item,
    reviewers,
    k,
    ld=1.0,
    opinion_type="binary",
):
    """
    item: item_id
    reviewers: user ids that wrote review on item, for retrieve review/sentiment purpose
    """
    aspect_id_map = get_aspect_id_map(sentiment, [target_item, item])
    target_aspect_vector, _ = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, target_item),
        opinion_type=opinion_type,
    )
    _, target_opinion_vector = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, item),
        opinion_type=opinion_type,
    )
    selected = []
    current_aspect_sentiments = []
    remaining_reviewers = reviewers.copy()
    for _ in range(k):
        selected_reviewer = None
        min_cost = 1e9
        for user in remaining_reviewers:
            t_aspect_sentiments = get_aspect_sentiments(
                sentiment, item, [user], current_aspect_sentiments
            )
            t_aspect_vector, t_opinion_vector = get_aspect_opinion_vectors(
                aspect_id_map,
                t_aspect_sentiments,
                opinion_type=opinion_type,
            )
            d_aspect = distance(target_aspect_vector, t_aspect_vector)
            d_opinion = distance(target_opinion_vector, t_opinion_vector)
            total_cost = ld * ld * d_aspect + d_opinion
            if total_cost < min_cost:
                selected_reviewer = user
                min_cost = total_cost
                current_aspect_sentiments = t_aspect_sentiments

        if selected_reviewer:
            selected.append(selected_reviewer)
            remaining_reviewers.remove(selected_reviewer)

    return selected


def crs_integer_regression(
    sentiment,
    item,
    reviewers,
    k,
    opinion_type="binary",
):
    aspect_id_map = get_aspect_id_map(sentiment, [item])
    _, target_opinion_vector = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, item),
        opinion_type=opinion_type,
    )
    vector_review_id = defaultdict()
    for reviewer in reviewers:
        _, opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map,
            get_aspect_sentiments(sentiment, item, [reviewer]),
            opinion_type=opinion_type,
        )
        reviewer_ids = vector_review_id.setdefault(
            tuple(opinion_vector.nonzero()[0]), []
        )
        reviewer_ids.append(reviewer)

    R = np.zeros(
        (len(vector_review_id), target_opinion_vector.shape[0]), dtype=(np.float64)
    )
    id_review_ids_map = []
    id_review_ids_count = []
    for inc, (tup, reviewer_ids) in enumerate(vector_review_id.items()):
        R[(inc, list(tup))] = 1
        id_review_ids_map.append(reviewer_ids)
        id_review_ids_count.append(len(reviewer_ids))

    id_review_ids_count = np.array(id_review_ids_count)
    min_cost = 1e9
    selected = []
    result = omp(R.T, target_opinion_vector)
    x = result.coef
    t_indices = x.nonzero()[0]
    C = sum((id_review_ids_count[idx] for idx in t_indices))
    for l in range(1, k + 1):
        possible_set = []
        for N in range(1, C + 1):
            s = np.zeros_like(x)
            Nv = N * x[t_indices]
            U = np.ceil(Nv).sum()
            L = np.floor(Nv).sum()
            if N <= L:
                s[t_indices] = np.floor(Nv)
            elif N >= U:
                s[t_indices] = np.ceil(Nv)
            else:
                X = int(N - L)
                (Nv - np.floor(Nv)).argsort()[:]
                X_largest_indices, X_other_indices = ((-Nv + np.floor(Nv)).argsort())[
                    :X
                ], ((-Nv + np.floor(Nv)).argsort())[X:]
                for idx in X_largest_indices:
                    s[t_indices[idx]] = np.ceil(Nv[idx])

                for idx in X_other_indices:
                    s[t_indices[idx]] = np.floor(Nv[idx])

            if s.sum() > 0 and s.sum() <= k:
                possible_set.append(s)

        if len(possible_set) > 0:
            S = np.array(possible_set)
            S_distances = (S / S.sum(axis=1).reshape(-1, 1) - x / x.sum()).sum(axis=1)
            s_ = S[S_distances.argmin()].astype(np.int64)
            selected_reviewers = []
            for idx in s_.nonzero()[0]:
                selected_reviewers.extend(id_review_ids_map[idx][: s_[idx]])

            _, opinion_vector = get_aspect_opinion_vectors(
                aspect_id_map,
                get_aspect_sentiments(sentiment, item, selected_reviewers),
                opinion_type=opinion_type,
            )
            d_opinion = distance(target_opinion_vector, opinion_vector)
            if d_opinion < min_cost:
                min_cost = d_opinion
                selected = selected_reviewers

    return selected


def rs_integer_regression(
    sentiment,
    target_item,
    item,
    reviewers,
    k,
    ld=1.0,
    opinion_type="binary",
):
    aspect_id_map = get_aspect_id_map(sentiment, [target_item, item])
    target_aspect_vector, _ = get_aspect_opinion_vectors(
        aspect_id_map, get_item_aspect_sentiments(sentiment, target_item)
    )
    _, target_opinion_vector = get_aspect_opinion_vectors(
        aspect_id_map, get_item_aspect_sentiments(sentiment, item)
    )
    vector_review_id = defaultdict()
    for reviewer in reviewers:
        aspect_vector, opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map, get_aspect_sentiments(sentiment, item, [reviewer])
        )
        joined_vector = np.concatenate([aspect_vector, opinion_vector])
        reviewer_ids = vector_review_id.setdefault(
            tuple(joined_vector.nonzero()[0]), []
        )
        reviewer_ids.append(reviewer)

    R = np.zeros(
        (
            len(vector_review_id),
            target_aspect_vector.shape[0] + target_opinion_vector.shape[0],
        ),
        dtype=(np.float64),
    )
    id_review_ids_map = []
    id_review_ids_count = []
    for inc, (tup, reviewer_ids) in enumerate(vector_review_id.items()):
        R[(inc, list(tup))] = 1
        id_review_ids_map.append(reviewer_ids)
        id_review_ids_count.append(len(reviewer_ids))

    R[:, target_aspect_vector.shape[0] :] *= ld
    min_cost = 1e9
    selected = []
    result = omp(R.T, np.concatenate([target_aspect_vector, target_opinion_vector]))
    x = result.coef
    t_indices = x.nonzero()[0]
    C = sum((id_review_ids_count[idx] for idx in t_indices))
    for l in range(1, k + 1):
        possible_set = []
        for N in range(1, C + 1):
            s = np.zeros_like(x)
            Nv = N * x[t_indices]
            U = np.ceil(Nv).sum()
            L = np.floor(Nv).sum()
            if N <= L:
                s[t_indices] = np.floor(Nv)
            elif N >= U:
                s[t_indices] = np.ceil(Nv)
            else:
                X = int(N - L)
                (Nv - np.floor(Nv)).argsort()[:]
                X_largest_indices, X_other_indices = ((-Nv + np.floor(Nv)).argsort())[
                    :X
                ], ((-Nv + np.floor(Nv)).argsort())[X:]
                for idx in X_largest_indices:
                    s[t_indices[idx]] = np.ceil(Nv[idx])

                for idx in X_other_indices:
                    s[t_indices[idx]] = np.floor(Nv[idx])

            if s.sum() > 0 and s.sum() <= k:
                possible_set.append(s)

        if len(possible_set) > 0:
            S = np.array(possible_set)
            S_denominator = S.sum(axis=1).reshape(-1, 1)
            S_denominator[np.where(S_denominator == 0)[0]] = 1.0
            S_distances = (S / S_denominator - x / x.sum()).sum(axis=1)
            s_ = S[S_distances.argmin()].astype(np.int64)
            selected_reviewers = []
            for idx in s_.nonzero()[0]:
                selected_reviewers.extend(id_review_ids_map[idx][: s_[idx]])

            aspect_vector, opinion_vector = get_aspect_opinion_vectors(
                aspect_id_map,
                get_aspect_sentiments(sentiment, item, selected_reviewers),
            )
            d_aspect = distance(target_aspect_vector, aspect_vector)
            d_opinion = distance(target_opinion_vector, opinion_vector)
            total_cost = ld * ld * d_aspect + d_opinion
            if total_cost < min_cost:
                min_cost = total_cost
                selected = selected_reviewers
    return selected


def select(
    sentiment,
    item,
    target_item=None,
    reviewers=[],
    algorithm="greedy-rs",
    k=1,
    ld=1.0,
    seed=None,
    opinion_type="binary",
):
    from time import time

    start_time = time()
    selected = []
    if len(reviewers) > 0:
        if algorithm == "random":
            selected = rs_random(reviewers, k, seed)
        elif algorithm == "iterative-random":
            selected = rs_iterative_random(reviewers, k)
        elif algorithm == "greedy-crs":
            selected = crs_greedy(
                sentiment, item, reviewers, k, opinion_type=opinion_type
            )
        elif algorithm == "integer-regression-crs":
            selected = crs_integer_regression(
                sentiment,
                item,
                reviewers,
                k,
                opinion_type=opinion_type,
            )
        elif algorithm == "greedy-rs":
            selected = rs_greedy(
                sentiment,
                target_item,
                item,
                reviewers,
                k,
                ld,
                opinion_type=opinion_type,
            )
        elif algorithm == "integer-regression-rs":
            selected = rs_integer_regression(
                sentiment,
                target_item,
                item,
                reviewers,
                k,
                ld,
                opinion_type=opinion_type,
            )
    end_time = time()
    return {"selected": selected, "time": end_time - start_time}


def urs_greedy(
    sentiment,
    target_item,
    items,
    k=1,
    ld=1.0,
    mu=1.0,
    opinion_type="binary",
):
    all_items = [target_item] + items
    aspect_id_map = get_aspect_id_map(sentiment, all_items)
    target_aspect_vector, _ = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, target_item),
        opinion_type=opinion_type,
    )
    all_selected = []
    for inc, item in enumerate(all_items):
        _, target_opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map,
            get_item_aspect_sentiments(sentiment, item),
            opinion_type=opinion_type,
        )
        selected = []
        current_aspect_sentiments = []
        reviewers = list(sentiment.get(item, {}).keys())
        remaining_reviewers = reviewers.copy()
        for _ in range(k):
            selected_reviewer = None
            min_cost = 1e9
            for user in remaining_reviewers:
                t_aspect_sentiments = get_aspect_sentiments(
                    sentiment, item, [user], current_aspect_sentiments
                )
                t_aspect_vector, t_opinion_vector = get_aspect_opinion_vectors(
                    aspect_id_map,
                    t_aspect_sentiments,
                    opinion_type=opinion_type,
                )
                d_aspect = distance(target_aspect_vector, t_aspect_vector)
                d_opinion = distance(target_opinion_vector, t_opinion_vector)
                total_cost = ld * d_aspect + d_opinion
                prev_items = all_items[:inc]
                for p_item in prev_items:
                    p_aspect_sentiments = get_aspect_sentiments(
                        sentiment, p_item, [user], current_aspect_sentiments
                    )
                    p_aspect_vector, _ = get_aspect_opinion_vectors(
                        aspect_id_map,
                        p_aspect_sentiments,
                        opinion_type=opinion_type,
                    )
                    total_cost += mu * distance(t_aspect_vector, p_aspect_vector)

                if total_cost < min_cost:
                    selected_reviewer = user
                    min_cost = total_cost
                    current_aspect_sentiments = t_aspect_sentiments

            if selected_reviewer:
                selected.append(selected_reviewer)
                remaining_reviewers.remove(selected_reviewer)

        all_selected.append(selected)

    return all_selected


def urs_integer_regression(
    sentiment,
    target_item,
    items,
    k=1,
    ld=1.0,
    mu=1.0,
    max_iter=1,
    initial_selection=None,
    seed=None,
    shuffle=False,
    opinion_type="binary",
):
    all_selected = []
    if initial_selection is None:
        reviewers = list(sentiment.get(target_item, {}).keys())
        selected = rs_integer_regression(
            sentiment,
            target_item,
            target_item,
            reviewers,
            k,
            ld,
            opinion_type=opinion_type,
        )
        initial_selection = {}
        initial_selection[target_item] = selected
        for inc, comparison_item in enumerate(items):
            reviewers = list(sentiment.get(comparison_item, {}).keys())
            selected = rs_integer_regression(
                sentiment,
                target_item,
                comparison_item,
                reviewers,
                k,
                ld,
                opinion_type=opinion_type,
            )
            initial_selection[comparison_item] = selected

    assert (
        len(initial_selection) == len(items) + 1
    ), f"{len(initial_selection)} != {len(items) + 1} for {target_item}"
    all_items = [target_item] + items
    all_items = [str(item) for item in all_items]
    aspect_id_map = get_aspect_id_map(sentiment, all_items)
    target_aspect_vector, _ = get_aspect_opinion_vectors(
        aspect_id_map,
        get_item_aspect_sentiments(sentiment, target_item),
        opinion_type=opinion_type,
    )
    aspect_vectors = []
    target_opinion_vectors = []
    for item in all_items:
        _, target_opinion_vector = get_aspect_opinion_vectors(
            aspect_id_map,
            get_item_aspect_sentiments(sentiment, item),
            opinion_type=opinion_type,
        )
        target_opinion_vectors.append(target_opinion_vector)
        selected_reviewers = initial_selection[item]
        all_selected.append(selected_reviewers)
        aspect_vector, _ = get_aspect_opinion_vectors(
            aspect_id_map,
            get_aspect_sentiments(sentiment, item, selected_reviewers),
            opinion_type=opinion_type,
        )
        aspect_vectors.append(aspect_vector)

    shuffled_ids = list(range(len(all_items)))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(shuffled_ids)
    for _ in range(max_iter):
        for sidx in shuffled_ids:
            item = all_items[sidx]
            target_opinion_vector = target_opinion_vectors[sidx]
            vector_review_id = {}
            reviewers = list(sentiment.get(item, {}).keys())
            for reviewer in reviewers:
                aspect_vector, opinion_vector = get_aspect_opinion_vectors(
                    aspect_id_map,
                    get_aspect_sentiments(sentiment, item, [reviewer]),
                    opinion_type=opinion_type,
                )
                joined_vector = np.concatenate(
                    [aspect_vector, opinion_vector]
                    + [aspect_vector for _ in range(len(all_items) - 1)]
                )
                reviewer_ids = vector_review_id.setdefault(
                    tuple(joined_vector.nonzero()[0]), []
                )
                reviewer_ids.append(reviewer)

            R = np.zeros(
                (
                    len(vector_review_id),
                    target_aspect_vector.shape[0]
                    + target_opinion_vector.shape[0]
                    + target_aspect_vector.shape[0] * (len(all_items) - 1),
                ),
                dtype=(np.float64),
            )
            id_review_ids_map = []
            id_review_ids_count = []
            for inc, (tup, reviewer_ids) in enumerate(vector_review_id.items()):
                R[(inc, list(tup))] = 1
                id_review_ids_map.append(reviewer_ids)
                id_review_ids_count.append(len(reviewer_ids))

            R[
                :,
                target_aspect_vector.shape[0] : target_aspect_vector.shape[0]
                + target_opinion_vector.shape[0],
            ] *= ld
            R[:, target_aspect_vector.shape[0] + target_opinion_vector.shape[0] :] *= mu
            min_cost = 1e9
            selected = []
            target_vector = np.concatenate(
                [target_aspect_vector, target_opinion_vector]
                + [av for inc, av in enumerate(aspect_vectors) if inc != sidx]
            )
            result = omp(R.T, target_vector)
            x = result.coef
            t_indices = x.nonzero()[0]
            C = sum((id_review_ids_count[idx] for idx in t_indices))
            for l in range(1, k + 1):
                possible_set = []
                for N in range(1, C + 1):
                    s = np.zeros_like(x)
                    Nv = N * x[t_indices]
                    U = np.ceil(Nv).sum()
                    L = np.floor(Nv).sum()
                    if N <= L:
                        s[t_indices] = np.floor(Nv)
                    elif N >= U:
                        s[t_indices] = np.ceil(Nv)
                    else:
                        X = int(N - L)
                        (Nv - np.floor(Nv)).argsort()[:]
                        X_largest_indices, X_other_indices = (
                            (-Nv + np.floor(Nv)).argsort()
                        )[:X], ((-Nv + np.floor(Nv)).argsort())[X:]
                        for idx in X_largest_indices:
                            s[t_indices[idx]] = np.ceil(Nv[idx])

                        for idx in X_other_indices:
                            s[t_indices[idx]] = np.floor(Nv[idx])

                    if s.sum() > 0 and s.sum() <= k:
                        possible_set.append(s)

                if len(possible_set) > 0:
                    S = np.array(possible_set)
                    S_denominator = S.sum(axis=1).reshape(-1, 1)
                    S_denominator[np.where(S_denominator == 0)[0]] = 1.0
                    S_distances = (S / S_denominator - x / x.sum()).sum(axis=1)
                    s_ = S[S_distances.argmin()].astype(np.int64)
                    selected_reviewers = []
                    for idx in s_.nonzero()[0]:
                        selected_reviewers.extend(id_review_ids_map[idx][: s_[idx]])

                    aspect_vector, opinion_vector = get_aspect_opinion_vectors(
                        aspect_id_map,
                        get_aspect_sentiments(sentiment, item, selected_reviewers),
                        opinion_type=opinion_type,
                    )
                    d_aspect = distance(target_aspect_vector, aspect_vector)
                    d_opinion = distance(target_opinion_vector, opinion_vector)
                    total_cost = ld * ld * d_aspect + d_opinion
                    if total_cost < min_cost:
                        min_cost = total_cost
                        selected = selected_reviewers
                all_selected[sidx] = selected

    return all_selected


def unified_select(
    sentiment,
    target_item,
    items,
    algorithm,
    k=1,
    ld=1.0,
    mu=1.0,
    initial_selection=None,
    seed=None,
    shuffle=False,
    max_iter=1,
    opinion_type="binary",
):
    from time import time

    start_time = time()
    all_selected = []
    if algorithm == "unified-greedy-rs":
        all_selected = urs_greedy(
            sentiment,
            target_item,
            items,
            k=k,
            ld=ld,
            mu=mu,
            opinion_type=opinion_type,
        )
    elif algorithm == "unified-integer-regression-rs":
        all_selected = urs_integer_regression(
            sentiment,
            target_item,
            items,
            k=k,
            ld=ld,
            mu=mu,
            max_iter=max_iter,
            initial_selection=initial_selection,
            seed=seed,
            shuffle=shuffle,
            opinion_type=opinion_type,
        )
    end_time = time()
    return {"all_selected": all_selected, "time": end_time - start_time}


def solve_and_export(
    sentiment,
    target_item,
    comparison_items,
    algorithm,
    k,
    ld=1.0,
    mu=1.0,
    save_dir="dist",
    initial_dir=None,
    seed=None,
    shuffle=False,
    max_iter=1,
    skip_exists=False,
    opinion_type="binary",
):
    output_dir = os.path.join(save_dir, "/".join(target_item))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "{}.txt".format(target_item))
    running_time_path = os.path.join(output_dir, "{}-time.txt".format(target_item))
    total_running_time = 0
    if skip_exists:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                lines = f.readlines()
                if len(lines) == len(comparison_items) + 1:
                    return
    if "unified" in algorithm:
        initial_selection = None
        if initial_dir is not None:
            initial_selection_path = os.path.join(
                initial_dir, "/".join(target_item), "{}.txt".format(target_item)
            )
            if os.path.exists(initial_selection_path):
                initial_selection = read_selection(initial_selection_path)
        res = unified_select(
            sentiment,
            target_item,
            comparison_items,
            algorithm,
            k=k,
            ld=ld,
            mu=mu,
            initial_selection=initial_selection,
            seed=seed,
            shuffle=shuffle,
            max_iter=max_iter,
            opinion_type=opinion_type,
        )
        all_items = [target_item] + comparison_items
        all_selected = res["all_selected"]
        with open(output_path, "w") as f:
            for inc, selected in enumerate(all_selected):
                f.write("{},{}\n".format(all_items[inc], ",".join(selected)))
        total_running_time += res["time"]
    else:
        reviewers = list(sentiment.get(target_item, {}).keys())
        with open(output_path, "w") as f:
            res = select(
                sentiment,
                target_item,
                target_item,
                reviewers,
                algorithm,
                k,
                ld,
                seed,
                opinion_type=opinion_type,
            )
            total_running_time += res["time"]
            f.write("{},{}\n".format(target_item, ",".join(res["selected"])))
            for inc, comparison_item in enumerate(comparison_items):
                reviewers = list(sentiment.get(comparison_item, {}).keys())
                c_res = select(
                    sentiment,
                    comparison_item,
                    target_item,
                    reviewers,
                    algorithm,
                    k,
                    ld,
                    seed,
                    opinion_type=opinion_type,
                )
                total_running_time += c_res["time"]
                f.write("{},{}\n".format(comparison_item, ",".join(c_res["selected"])))

    with open(running_time_path, "w") as f:
        f.write("{}\n{}\n".format(total_running_time, len(comparison_items) + 1))


if __name__ == "__main__":
    from tqdm import tqdm
    from joblib import Parallel, delayed

    args = parse_arguments()
    os.makedirs((args.out), exist_ok=True)
    sentiment = read_sentiment(args.sentiment_path)
    data = read_data(args.input_path)
    begin_index = 0
    end_index = len(data)
    if args.target is not None:
        tokens = args.target.split("-")
        begin_index = int(tokens[0])
        end_index = int(tokens[1])

    for inc, (target_item, comparison_items) in enumerate(data.items()):
        if inc >= begin_index and inc <= end_index:
            solve_and_export(
                sentiment,
                target_item,
                comparison_items,
                args.algorithm,
                args.k,
                args.ld,
                args.mu,
                args.out,
                args.initial_selection_dir,
                args.random_seed,
                args.shuffle,
                args.max_iter,
                args.skip_exists,
                args.opinion_type,
            )
