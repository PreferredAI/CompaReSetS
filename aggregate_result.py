import pandas as pd
import os
from selection import read_data


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/cellphone/result/greedy-rs-3",
    )
    parser.add_argument(
        "-d", "--data_path", type=str, default="data/cellphone/also_bought.txt"
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


if __name__ == "__main__":
    from tqdm import tqdm

    args = parse_arguments()
    data = read_data(args.data_path)
    begin_index = 0
    end_index = len(data)
    output_path = os.path.join(args.input_dir, "result.csv")
    output_running_time_path = os.path.join(args.input_dir, "running_time.csv")
    if args.target is not None:
        tokens = args.target.split("-")
        begin_index = int(tokens[0])
        end_index = int(tokens[1])
        output_path = os.path.join(
            args.input_dir, "result_{}_{}.csv".format(begin_index, end_index)
        )
        output_running_time_path = os.path.join(
            args.input_dir, "running_time_{}_{}.csv".format(begin_index, end_index)
        )
    results = []
    running_time_results = []
    for inc, target_item in tqdm(enumerate(data.keys())):
        if inc >= begin_index and inc <= end_index:
            i_result = pd.read_csv(
                os.path.join(
                    args.input_dir, "/".join(str(target_item)), "evaluation.csv"
                )
            )
            i_result["item_i"] = i_result["item_i"].map(str)
            results.append(
                {
                    "item": target_item,
                    "rouge_1": i_result["rouge_1"].mean(),
                    "rouge_2": i_result["rouge_2"].mean(),
                    "rouge_l": i_result["rouge_l"].mean(),
                    "rouge_1_target": i_result[i_result["item_i"] == str(target_item)][
                        "rouge_1"
                    ].mean(),
                    "rouge_2_target": i_result[i_result["item_i"] == str(target_item)][
                        "rouge_2"
                    ].mean(),
                    "rouge_l_target": i_result[i_result["item_i"] == str(target_item)][
                        "rouge_l"
                    ].mean(),
                }
            )
            running_time_path = os.path.join(
                args.input_dir,
                "/".join(str(target_item)),
                "{}-time.txt".format(target_item),
            )
            if os.path.exists(running_time_path):
                res = open(
                    running_time_path,
                    "r",
                ).readlines()
                running_time_results.append({"time": float(res[0].strip()), "n_items": int(res[1].strip())})
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    pd.DataFrame(running_time_results).to_csv(output_running_time_path, index=False)
    print(",".join([str(x) for x in df.describe().loc["mean"].tolist()]))
