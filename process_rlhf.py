import json
from datasets import load_dataset
import random

def process_hh_rlhf():
    # 加载数据集
    dataset = load_dataset("Anthropic/hh-rlhf")  # 官方数据集
    samples = []
    # 处理并保存为jsonl
    for split in ['train', 'test']:
        with open(f'./dataset/hh_rlhf_{split}.jsonl', 'w') as f:
            for example in dataset[split]:
                # 构建对话上下文
                prompt = example["chosen"].split("\nAssistant:")[0].replace("\n\nHuman: ", "")
                chosen_answer = example["chosen"].split("\nAssistant: ")[1].split("\n\nHuman:")[0]
                # 构建最终样本
                sample = {
                    'sys_prompt': "You are a helpful assistant.",
                    'input': prompt,#example['input'].strip(),
                    'answer': chosen_answer,
                    'context_messages': [{"role": "user", "content": prompt}]
                }
                samples.append(sample)
                f.write(json.dumps(sample) + '\n')
    return samples

def process_ultra_feedback():
    dataset = load_dataset("openbmb/UltraFeedback")
    samples = []
    for split in ['train']:
        for example in dataset[split]:
            prompt = example["instruction"]
            chosen_answer = example["completions"][0]["response"] if len(example["completions"]) > 0 else "sorry, I don't know."
            # 构建最终样本
            sample = {
                'sys_prompt': "You are a helpful assistant.",
                'input': prompt,#example['input'].strip(),
                'answer': chosen_answer,
                'context_messages': [{"role": "user", "content": prompt}]
            }
            samples.append(sample)
            #f.write(json.dumps(sample) + '\n')
    return samples

if __name__ == "__main__":
    prompts = process_hh_rlhf() #+ process_ultra_feedback()
    # random.shuffle(prompts)
    # with open('./.jsonl', 'w') as f:
    #     for sample in prompts:
    #         f.write(json.dumps(sample) + '\n')
