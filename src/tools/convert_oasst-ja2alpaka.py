import argparse
import os
import json

import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser(
            description="generate chat dataset using Narou API",
            add_help=True,
            )
parser.add_argument("--input_dir",
                    default="/home/user0/work/data/datasets/oasst1-89k-ja/oasst1_ja_89k",
                    default="/home/user0/work/data/datasets/alpaca_ja",
                    help="oasst format jsons dir path")
parser.add_argument("--output_dir",
                    default="/home/user0/work/data/datasets/alpaca_ja",
                    help="output dir path")
args = parser.parse_args()


def prepare_default_dataset(dataset_path:str) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame()
    for f in os.listdir(dataset_path):
        ds = load_dataset("json", data_files=os.path.join(dataset_path, f))
        df = pd.concat([df, ds["train"].to_pandas()], axis=0).reset_index(drop=True)

    df = df.set_index("message_id")

    inputs = []
    outputs = []
    parent_ids = []
    message_ids = []
    for index, row in df.iterrows():
        if row.parent_id == row.message_tree_id:
            continue

        input = df.loc[row.parent_id]
        inputs.append(input.text_ja)
        outputs.append(row.text_ja)
        parent_ids.append(input.parent_id)
        message_ids.append(index)

    df_output: pd.DataFrame = pd.DataFrame()
    df_output["instruction"] = inputs
    df_output["output"] = outputs
    df_output["message_id"] = message_ids
    df_output["parent_id"] = parent_ids

    df_output = df_output[
        ["instruction", "output", "message_id", "parent_id"]
    ].rename(columns={"message_id": "id"})

    return df_output


def main():
    df: pd.DataFrame = prepare_default_dataset(args.input_dir)

    corpus: list = []
    for _, row in df.iterrows():
        record:dict = {}
        record["instruction"] = row.instruction
        record["input"] = ""
        record["output"] = row.output
        corpus.append(record)

    with open(os.path.join(args.output_dir, "./output.json"), encoding="utf-8", mode="w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
