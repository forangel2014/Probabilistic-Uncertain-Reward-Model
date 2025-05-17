"""
数据集处理脚本：将不同来源的数据集统一处理成 prompt, chosen, reject 格式
"""

import datasets
from tqdm import tqdm
from typing import Dict, List, Tuple
from huggingface_hub import login
from transformers import AutoTokenizer
import random

def load_and_process_helpsteer2() -> List[Dict]:
    """处理 HelpSteer2 数据集"""
    dataset = datasets.load_dataset("nvidia/HelpSteer2")
    processed_data = []
    for split in ["train", "validation"]:
        split_data = dataset[split]
        for i in tqdm(range(len(split_data)//2)):
            sample_a = split_data[2*i]
            sample_b = split_data[2*i+1]
            score_a = sample_a["helpfulness"] + sample_a["correctness"] + sample_a["coherence"] + sample_a["complexity"] + sample_a["verbosity"]
            score_b = sample_b["helpfulness"] + sample_b["correctness"] + sample_b["coherence"] + sample_b["complexity"] + sample_b["verbosity"]
            processed_data.append({
                "prompt": sample_a["prompt"],
                "chosen": sample_a["response"] if score_a > score_b else sample_b["response"],
                "reject": sample_b["response"] if score_a > score_b else sample_a["response"]
            })
    return processed_data

def load_and_process_chatarena() -> List[Dict]:
    """处理 ChatArena 数据集"""
    dataset = datasets.load_dataset("lmsys/chatbot_arena_conversations")
    processed_data = []
    for split in ["train"]:
        split_data = dataset[split]
        for i in tqdm(range(len(split_data))):
            sample = split_data[i]
            if sample["winner"] != "tie" and sample["turn"] == 1:
                conversation_a = sample["conversation_a"]
                conversation_b = sample["conversation_b"]
                processed_data.append({
                    "prompt": conversation_a[0]["content"],
                    "chosen": conversation_a[1]["content"] if sample["winner"] == "model_a" else conversation_b[1]["content"],
                    "reject": conversation_b[1]["content"] if sample["winner"] == "model_a" else conversation_a[1]["content"]
                })
    return processed_data

def load_and_process_alpacafarm() -> List[Dict]:
    """处理 AlpacaFarm-HumanPref 数据集"""
    dataset = datasets.load_dataset("allenai/tulu-2.5-preference-data")
    #读取dataset下所有split
    processed_data = []
    for split in dataset.keys():
        split_data = dataset[split]
        for i in tqdm(range(len(split_data))):
            sample = split_data[i]
            processed_data.append({
                "prompt": sample["chosen"][0]["content"],
                "chosen": sample["chosen"][1]["content"],
                "reject": sample["rejected"][1]["content"]
            })
    return processed_data

def load_and_process_pku_saferlhf() -> List[Dict]:
    """处理 PKU-SafeRLHF 数据集"""
    dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF")
    processed_data = []
    for split in dataset.keys():
        split_data = dataset[split]
        for i in tqdm(range(len(split_data))):
            sample = split_data[i]
            processed_data.append({
                "prompt": sample["prompt"],
                "chosen": sample["response_0"] if sample["better_response_id"] == 0 else sample["response_1"],
                "reject": sample["response_1"] if sample["better_response_id"] == 0 else sample["response_0"]
            })
    return processed_data

def load_all_datasets() -> List[Dict]:
    all_data = []
    all_data.extend(load_and_process_helpsteer2())
    all_data.extend(load_and_process_chatarena())
    all_data.extend(load_and_process_alpacafarm())
    all_data.extend(load_and_process_pku_saferlhf())
    return all_data

def split_dataset(dataset, name, ratio=[0.8, 0.1, 0.1]):
    train_dataset = dataset.select(range(int(len(dataset) * ratio[0])))
    valid_dataset = dataset.select(range(int(len(dataset) * ratio[0]), int(len(dataset) * (ratio[0] + ratio[1]))))
    test_dataset = dataset.select(range(int(len(dataset) * (ratio[0] + ratio[1])), len(dataset)))
    train_dataset.save_to_disk(f"{name}-train")
    valid_dataset.save_to_disk(f"{name}-valid")
    test_dataset.save_to_disk(f"{name}-test")
    mini_test_dataset = dataset.select(range(10))
    mini_test_dataset.save_to_disk(f"{name}-mini")

def preprocess_function(examples, tokenizer):
    input_ids_chosen = []
    attention_mask_chosen = []
    input_ids_rejected = []
    attention_mask_rejected = []
    
    chosen_text = f"Question: {examples['prompt']}\nAnswer: {examples['chosen']}"
    rejected_text = f"Question: {examples['prompt']}\nAnswer: {examples['reject']}"

    chosen_ids = tokenizer(chosen_text, truncation=True, max_length=2048)
    rejected_ids = tokenizer(rejected_text, truncation=True, max_length=2048)

    input_ids_chosen.append(chosen_ids["input_ids"])
    attention_mask_chosen.append(chosen_ids["attention_mask"])
    input_ids_rejected.append(rejected_ids["input_ids"])
    attention_mask_rejected.append(rejected_ids["attention_mask"])
    
    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
    }

def preprocess_function_noisy(examples, tokenizer):
    input_ids_chosen = []
    attention_mask_chosen = []
    input_ids_rejected = []
    attention_mask_rejected = []
    
    chosen_text = f"Question: {examples['prompt']}\nAnswer: {examples['chosen']}"
    rejected_text = f"Question: {examples['prompt']}\nAnswer: {examples['reject']}"

    chosen_ids = tokenizer(chosen_text, truncation=True, max_length=2048)
    rejected_ids = tokenizer(rejected_text, truncation=True, max_length=2048)

    input_ids_chosen.append(chosen_ids["input_ids"])
    attention_mask_chosen.append(chosen_ids["attention_mask"])
    input_ids_rejected.append(rejected_ids["input_ids"])
    attention_mask_rejected.append(rejected_ids["attention_mask"])
    
    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
    }

def preprocess_function_reverse(examples, tokenizer, reverse_ratio=0.1):
    input_ids_chosen = []
    attention_mask_chosen = []
    input_ids_rejected = []
    attention_mask_rejected = []
    
    if random.random() < 1-reverse_ratio:
        chosen_text = f"Question: {examples['prompt']}\nAnswer: {examples['chosen']}"
        rejected_text = f"Question: {examples['prompt']}\nAnswer: {examples['reject']}"
    else:
        chosen_text = f"Question: {examples['prompt']}\nAnswer: {examples['reject']}"
        rejected_text = f"Question: {examples['prompt']}\nAnswer: {examples['chosen']}"        

    chosen_ids = tokenizer(chosen_text, truncation=True, max_length=2048)
    rejected_ids = tokenizer(rejected_text, truncation=True, max_length=2048)

    input_ids_chosen.append(chosen_ids["input_ids"])
    attention_mask_chosen.append(chosen_ids["attention_mask"])
    input_ids_rejected.append(rejected_ids["input_ids"])
    attention_mask_rejected.append(rejected_ids["attention_mask"])
    
    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
    }


if __name__ == "__main__":
    processed_data = load_all_datasets()
    dataset = datasets.Dataset.from_list(processed_data)
    dataset = dataset.shuffle(seed=73)
    
    tokenizer = AutoTokenizer.from_pretrained("<your_reward_model_path>")

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=1,
        num_proc=100
    )

    indices = list(range(len(tokenized_dataset)))
    random.Random(37).shuffle(indices)
    tokenized_dataset = tokenized_dataset.select(indices)

    split_dataset(tokenized_dataset, 
        f"./dataset/preference_ppo",
        ratio=[0.49, 0.49, 0.02]
    )

    # for reverse_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #     # 生成随机排列
    #     n = int(len(indices) * reverse_ratio)
    #     reverse_indices = set(indices[:n])

    #     # 定义交换函数
    #     def swap_fields(example, idx):
    #         if idx in reverse_indices:
    #             return {
    #                 "input_ids_chosen": example["input_ids_rejected"],
    #                 "attention_mask_chosen": example["attention_mask_rejected"],
    #                 "input_ids_rejected": example["input_ids_chosen"],
    #                 "attention_mask_rejected": example["attention_mask_chosen"]
    #             }
    #         return example

    #     # 应用交换
    #     swapped_dataset = tokenized_dataset.map(
    #         swap_fields,
    #         with_indices=True,
    #         batched=False,
    #         num_proc=40
    #     )

    #     split_dataset(swapped_dataset, 
    #         f"/cpfs/user/sunwangtao/dataset/uncertain/reverse/preference-reverse-{reverse_ratio}",
    #         ratio=[0.99, 0.005, 0.005]
    #     )