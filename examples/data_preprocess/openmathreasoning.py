# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Openmathreasoning dataset to parquet format with train/test split
"""

import argparse
import os
import re
from sklearn.model_selection import train_test_split

import datasets

from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str: str) -> str:
    """
    '#### ' 以降に書かれた最終解答を取り出して正規化して返す。

    * 数値         : カンマ除去 ('1,234.5' → '1234.5')
    * LaTeX 分数   : '\\frac{3}{4}' → '3/4'
    * それ以外     : 記号を軽く除去した文字列をそのまま返す
                     例) '\\( a \\in (-1, 1) \\)' → 'a in (-1, 1)'
    """
    # 1. '####' の右側 1 行を取得
    m = re.search(r"####\s*([^\n]+)", solution_str)
    assert m is not None, f"'####' が見つかりません: {solution_str}"
    raw = m.group(1).strip()

    # 2. LaTeX の囲み (\( \) や $ $) を除去し、\in → in へ簡易変換
    raw = raw.replace(r"\in", "in")
    raw = re.sub(r"[\\\(\)\$]", "", raw).strip()

    # 3. 数値（整数／小数）を試す
    num_match = re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", raw)
    if num_match:
        return raw.replace(",", "")

    # 4. LaTeX 分数を試す
    frac_match = re.fullmatch(r"(-?)frac\{([^}]+)\}\{([^}]+)\}", raw)
    if frac_match:
        sign, num, den = frac_match.groups()
        return f"{'-' if sign else ''}{num.strip()}/{den.strip()}"

    # 5. 上記以外 → クリーンアップ済み文字列をそのまま返す
    return raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/miromind_m1_sft_719k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--test_size", type=float, default=0.001, help="Proportion of data to use for test set")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for train/test split")

    args = parser.parse_args()
    data_source = "LLMcompe-Team-Watanabe/OpenMathReasoning-filtered-transformed"
    
    try:
        dataset = datasets.load_dataset(data_source)
    except Exception as e:
        print(f"Error loading dataset {data_source}: {e}")
        print("Trying alternative loading methods...")
        # Try loading with specific configurations if available
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # Check dataset structure
    if "train" in dataset:
        # Dataset already has train/test split
        train_dataset = dataset["train"]
        test_dataset = dataset.get("test", dataset.get("validation"))
        if test_dataset is None:
            # Create test split from train data
            all_data = train_dataset
            train_indices, test_indices = train_test_split(
                range(len(all_data)), 
                test_size=args.test_size, 
                random_state=args.random_seed
            )
            train_dataset = all_data.select(train_indices)
            test_dataset = all_data.select(test_indices)
    else:
        # Dataset doesn't have splits, create them
        all_data = dataset
        if isinstance(all_data, dict):
            # Get the first split available
            all_data = list(all_data.values())[0]
        
        # Create train/test split
        train_indices, test_indices = train_test_split(
            range(len(all_data)), 
            test_size=args.test_size, 
            random_state=args.random_seed
        )
        train_dataset = all_data.select(train_indices)
        test_dataset = all_data.select(test_indices)

    # Process function to format data
    def make_map_fn(split):
        def process_fn(example, idx):
            # MiroMind-M1-SFT-719K has specific structure: id, question, response
            question_raw = example.get("question", "")
            answer_raw = example.get("answer", "")

            question = question_raw
            
            # Extract final answer if possible
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution if solution else answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    # Apply processing to datasets
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet format
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    
    print("Dataset split completed:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Files saved to: {local_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"  Files copied to HDFS: {args.hdfs_dir}")
