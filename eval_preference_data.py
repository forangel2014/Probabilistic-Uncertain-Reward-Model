from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
import numpy as np
import pandas as pd
import random
import os

tqdm.pandas()

def subset2domain(subset):
    for domain, subsets in SUBSET_MAPPING.items():
        if subset in subsets:
            return domain
    return None

def calculate_average_overlap_degree(mus, sigmas):

    n = len(mus)

    mus = torch.stack(mus)  # shape: [n]
    sigmas = torch.stack(sigmas)  # shape: [n]
    # 创建网格以进行批量计算

    mu_i = mus.unsqueeze(1)  # shape: [n, 1]
    mu_j = mus.unsqueeze(0)  # shape: [1, n]
    sigma_i = sigmas.unsqueeze(1)  # shape: [n, 1]
    sigma_j = sigmas.unsqueeze(0)  # shape: [1, n]

    # 批量计算 Bhattacharyya 系数
    sqrt_term = torch.sqrt(2 * sigma_i * sigma_j / (sigma_i**2 + sigma_j**2))
    exp_term = torch.exp(-(mu_i - mu_j)**2 / (4 * (sigma_i**2 + sigma_j**2)))
    bc_matrix = sqrt_term * exp_term

    #让bc_matrix减去自身的对角线矩阵
    bc_matrix = bc_matrix - torch.diag(torch.diag(bc_matrix))
    bc = torch.sum(bc_matrix, dim=1) / (n-1)
    
    A = torch.sum(bc) / n
    print(f"Average overlap degree: {A}")
    return bc, A

def calculate_ece(ece_stat):
    ece_stat_sorted = sorted(ece_stat, key=lambda x: x[0])
    # 把0.5到1的概率范围划分成100个区间
    interval = 0.005
    ece_interval = [0.5+interval*i for i in range(1,101)]
    ece_samples = []
    i = 0
    current_samples = []
    while i < len(ece_interval) and len(ece_stat_sorted) > 0:
        if ece_stat_sorted[0][0] < ece_interval[i]:
            current_samples.append(ece_stat_sorted[0])
            ece_stat_sorted.pop(0)
        else:
            ece_samples.append(current_samples)
            current_samples = []
            i += 1
    ece_acc = []
    ece_probs = []
    for samples in ece_samples:
        if len(samples) == 0:
            ece_acc.append(0)
            ece_probs.append(0)
        else:
            ece_acc.append((torch.tensor(samples)[:,1]).mean().item())
            ece_probs.append((torch.tensor(samples)[:,0]).mean().item())
    ece_acc = torch.tensor(ece_acc)
    ece_probs = torch.tensor(ece_probs)
    sample_nums = torch.tensor([len(samples) for samples in ece_samples])
    ece = torch.sum(torch.abs(ece_acc - ece_probs) * sample_nums).item() / len(ece_stat)
    print(f"ECE: {ece}")
    return ece

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    data_set_name: Optional[str] = field(
        default='allenai/reward-bench',
        metadata={"help": "the location of the dataset name or path"},
    )
    record_dir: Optional[str] = field(
        default="./exp/reward_bench.txt",
        metadata={"help": "the location of the output file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="./exp/purm/ckpt/checkpoint-2000",
        metadata={"help": "the name of the gold reward model"},
    )
    model_type: Optional[str] = field(
        default="purm",
        metadata={"help": "the type of the reward model"},
    )
    ood_type: Optional[str] = field(
        default="no",
        metadata={"help": "the type of the ood"},
    )
    prompt_template: Optional[str] = field(
        default="qa",
        metadata={"help": "the type of the prompt template"},
    )
    device: Optional[int] = field(
        default=6,
        metadata={"help": "the device to use"},
    )

    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ds_dir = script_args.data_set_name
record_dir = script_args.record_dir 

rm_name = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(rm_name)
device = script_args.device
max_length = 1024

if script_args.model_type == "purm":
    num_labels = 2
elif script_args.model_type == "odin":
    num_labels = 2
elif script_args.model_type == "brme":
    num_labels = 10
else:
    num_labels = 1

if script_args.data_set_name == "argilla/distilabel-math-preference-dpo":
    ds = load_dataset(ds_dir, split='train', keep_in_memory=True)
    ds = ds.rename_column("instruction", "prompt")
    ds = ds.rename_column("chosen_response", "chosen") 
    ds = ds.rename_column("rejected_response", "rejected")
    ds = ds.add_column("subset", ["math"] * len(ds))
    ds = ds.add_column("id", [i for i in range(len(ds))])

    categories = {
        "math": ["math"],
    }
    EXAMPLE_COUNTS = {
        "math": len(ds),  # actual length 447, upweighting to be equal to code
    }
    SUBSET_MAPPING = {
        "math": ["math"],
    }

elif script_args.data_set_name == "sdiazlor/math-preference-dataset":
    ds = load_dataset(ds_dir, 'format_dpo', split='train', keep_in_memory=True)
    ds = ds.add_column("subset", ["math"] * len(ds))
    ds = ds.add_column("id", [i for i in range(len(ds))])

    # 处理prompt列
    def clean_prompt(text):
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith('1. "') and text.endswith('"'):
            text = text[4:-1]
        return text
    
    ds = ds.map(lambda x: {'prompt': clean_prompt(x['prompt'])})
    
    # 处理chosen和rejected列
    ds = ds.map(lambda x: {
        'chosen': x['chosen'][1]['content'],
        'rejected': x['rejected'][1]['content']
    })
    
    categories = {
        "math": ["math"],
    }
    EXAMPLE_COUNTS = {
        "math": len(ds),
    }
    SUBSET_MAPPING = {
        "math": ["math"],
    }

elif script_args.data_set_name == "Hello-SimpleAI/HC3-Chinese":
    ds = load_dataset(ds_dir, name="all", split='train', keep_in_memory=True)
    ds = ds.rename_column("question", "prompt")
    ds = ds.rename_column("human_answers", "chosen") 
    ds = ds.rename_column("chatgpt_answers", "rejected")
    ds = ds.add_column("subset", ["chinese"] * len(ds))

    categories = {
        "chinese": ["chinese"],
    }
    EXAMPLE_COUNTS = {
        "chinese": len(ds),
    }
    SUBSET_MAPPING = {
        "chinese": ["chinese"],
    }

elif script_args.data_set_name == "Aratako/magpie-sft-v1.0-dpo-judged":
    ds = load_dataset(ds_dir, split='train', keep_in_memory=True).select(range(1000))
    ds = ds.add_column("subset", ["japanese"] * len(ds))

    categories = {
        "japanese": ["japanese"],
    }
    EXAMPLE_COUNTS = {
        "japanese": len(ds),
    }
    SUBSET_MAPPING = {
        "japanese": ["japanese"],
    }

elif script_args.data_set_name == "dzunggg/legal-qa-v1":
    ds = load_dataset(ds_dir, split='train', keep_in_memory=True)
    ds = ds.rename_column("question", "prompt")
    ds = ds.rename_column("answer", "chosen") 
    ds = ds.add_column("rejected", ds["chosen"])
    ds = ds.add_column("subset", ["legal"] * len(ds))
    ds = ds.add_column("id", [i for i in range(len(ds))])

    categories = {
        "legal": ["legal"],
    }
    EXAMPLE_COUNTS = {
        "legal": len(ds),
    }
    SUBSET_MAPPING = {
        "legal": ["legal"],
    }

elif script_args.data_set_name == "allenai/reward-bench":
    ds = load_dataset(ds_dir, split='filtered', keep_in_memory=True)#.select(range(100))
    categories = {
        "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                    'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
                'donotanswer'],
        "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
    }
    EXAMPLE_COUNTS = {
        "alpacaeval-easy": 100,
        "alpacaeval-length": 95,
        "alpacaeval-hard": 95,
        "mt-bench-easy": 28,
        "mt-bench-med": 40,
        "mt-bench-hard": 37,
        "math-prm": 984,  # actual length 447, upweighting to be equal to code
        "refusals-dangerous": 100,
        "refusals-offensive": 100,
        "llmbar-natural": 100,
        "llmbar-adver-neighbor": 134,
        "llmbar-adver-GPTInst": 92,
        "llmbar-adver-GPTOut": 47,
        "llmbar-adver-manual": 46,
        "xstest-should-refuse": 250,
        "xstest-should-respond": 154,
        "donotanswer": 136,
        "hep-cpp": 164,
        "hep-go": 164,
        "hep-java": 164,
        "hep-js": 164,
        "hep-python": 164,
        "hep-rust": 164,
    }
    SUBSET_MAPPING = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Chat Hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "Safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "Reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

if script_args.model_type == "bte":
    rm_names = [rm_name] + [rm_name.replace("btrm", f"btrm{i}") for i in [1, 2, 3, 4]]
    devices = [0, 1, 2, 3, 4]
    rm_pipes = [pipeline(
        "sentiment-analysis",
        model=rm_name,
        device=device,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16, "num_labels": num_labels},  # 添加num_labels参数
        truncation=True,
        max_length=max_length#2048
    ) for rm_name, device in zip(rm_names, devices)]

else:
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=rm_name,
        device=device,  
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16, "num_labels": num_labels},  # 添加num_labels参数
        truncation=True,
        max_length=max_length#2048
    )

pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}

if script_args.model_type == "dropout":
    rm_pipe.model.train()  # 设置模型为训练模式以启用dropout
    #pipe_kwargs["num_samples"] = 10  # 添加采样次数参数

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = round(100 * total_weighted_score / total_examples, 2)
        else:
            section_scores[section] = 0
    return section_scores

def change_of_format(prompt, resp):

    #将resp的词序随机打乱
    if script_args.ood_type == "words":
        resp = " ".join(random.sample(resp.split(), len(resp.split())))

    elif script_args.ood_type == "responses":
        resp = domains_responses[subset2domain(example['subset'])].pop(0)

    if script_args.prompt_template == "chat":
        message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]

        return rm_tokenizer.apply_chat_template(message, tokenize=False)#.replace(rm_tokenizer.bos_token, "")
    else:
        return f"Question: {prompt}\nAnswer: {resp}"

def get_reward(test_texts):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards

if script_args.ood_type == "responses":
    domains_responses = dict(zip(SUBSET_MAPPING.keys(), [[] for _ in range(len(SUBSET_MAPPING.keys()))]))
    for example in ds:
        domains_responses[subset2domain(example['subset'])].append(example['chosen'])
        domains_responses[subset2domain(example['subset'])].append(example['rejected'])

    for domain in domains_responses.keys():
        np.random.shuffle(domains_responses[domain])

df = pd.DataFrame(columns=['id', 'subset', 'correct', 'prob', 'sigma_chosen', 'sigma_rejected', 'mu_chosen', 'mu_rejected', 'bc_chosen', 'bc_rejected', 'nll'])

gmm_params = {
    "weights": [],
    "means": [],
    "sigmas": [],
}
ece_stat = []
ece_2d_stat = []

for i, example in enumerate(tqdm(ds)):

    if script_args.model_type == "dropout":

        rewards_chosen = []
        rewards_rejected = []
        for i in range(10):
            output_chosen, output_rejected = rm_pipe(
                [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
            )
            rewards_chosen.append(torch.tensor(output_chosen[0]["score"]))
            rewards_rejected.append(torch.tensor(output_rejected[0]["score"]))
        #prob = torch.sigmoid(mean_chosen - mean_rejected)
        rewards_chosen = torch.tensor(rewards_chosen)
        rewards_rejected = torch.tensor(rewards_rejected)
        mean_chosen = rewards_chosen.mean()
        sigma_chosen = rewards_chosen.std()
        mean_rejected = rewards_rejected.mean()
        sigma_rejected = rewards_rejected.std()
        prob = torch.sigmoid(mean_chosen - mean_rejected)

    elif script_args.model_type == "purm":

        output_chosen, output_rejected = rm_pipe(
            [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
        )

        mean_chosen = torch.tensor(output_chosen[0]["score"])
        log_sigma_chosen = torch.tensor(output_chosen[1]["score"])
        mean_rejected = torch.tensor(output_rejected[0]["score"])
        log_sigma_rejected = torch.tensor(output_rejected[1]["score"])
        temperature = 10.0
        sigma_chosen = torch.exp(log_sigma_chosen/temperature)
        sigma_rejected = torch.exp(log_sigma_rejected/temperature)
        mean_z = mean_chosen - mean_rejected
        sigma_z = torch.sqrt(sigma_chosen**2 + sigma_rejected**2)
        num_sample = 1000
        # 重参数化采样num_sample个z
        z_samples = torch.randn(num_sample).to(sigma_z.device).to(torch.float16) * sigma_z + mean_z
        prob = torch.sigmoid(z_samples).mean()

        #ece_2d_stat.append((sigma_z, sigma_z, correct))

    elif script_args.model_type == "odin":

        output_chosen, output_rejected = rm_pipe(
            [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
        )

        reward_q_chosen = torch.tensor(output_chosen[0]["score"])
        reward_l_chosen = torch.tensor(output_chosen[1]["score"])
        reward_q_rejected = torch.tensor(output_rejected[0]["score"])
        reward_l_rejected = torch.tensor(output_rejected[1]["score"])
        mean_chosen = reward_q_chosen + reward_l_chosen
        mean_rejected = reward_q_rejected + reward_l_rejected
        prob = torch.sigmoid(mean_chosen - mean_rejected)

    elif script_args.model_type == "brme":

        output_chosen, output_rejected = rm_pipe(
            [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
        )
        temperature = 10.0
        mus_chosen = []
        mus_rejected = []
        sigmas_chosen = []
        sigmas_rejected = []
        for i in range(5):
            mean_chosen = torch.tensor(output_chosen[i]["score"])
            mean_rejected = torch.tensor(output_rejected[i]["score"])
            log_sigma_chosen = torch.tensor(output_chosen[i+5]["score"])
            log_sigma_rejected = torch.tensor(output_rejected[i+5]["score"])
            sigma_chosen = torch.exp(log_sigma_chosen/temperature)
            sigma_rejected = torch.exp(log_sigma_rejected/temperature)
            mus_chosen.append(mean_chosen)
            mus_rejected.append(mean_rejected)
            sigmas_chosen.append(sigma_chosen)
            sigmas_rejected.append(sigma_rejected)
        mus_chosen = torch.stack(mus_chosen)
        mus_rejected = torch.stack(mus_rejected)
        sigmas_chosen = torch.stack(sigmas_chosen)
        sigmas_rejected = torch.stack(sigmas_rejected)
        
        min_sigma_idx_chosen = torch.argmin(sigmas_chosen)
        nominal_chosen = mus_chosen[min_sigma_idx_chosen]
        min_sigma_idx_rejected = torch.argmin(sigmas_rejected)
        nominal_rejected = mus_rejected[min_sigma_idx_rejected]
        prob = torch.sigmoid(nominal_chosen - nominal_rejected)
        
    elif script_args.model_type == "btrm":
        
        output_chosen, output_rejected = rm_pipe(
            [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
        )
        
        mean_chosen = torch.tensor(output_chosen[0]["score"])
        mean_rejected = torch.tensor(output_rejected[0]["score"])

        prob = torch.sigmoid(mean_chosen - mean_rejected)
    
    elif script_args.model_type == "bte":
        rewards_chosen = []
        rewards_rejected = []
        for rm_pipe in rm_pipes:
            output_chosen, output_rejected = rm_pipe(
                [change_of_format(example['prompt'], example['chosen']), change_of_format(example['prompt'], example['rejected'])], **pipe_kwargs
            )
            rewards_chosen.append(torch.tensor(output_chosen[0]["score"]))
            rewards_rejected.append(torch.tensor(output_rejected[0]["score"]))
        rewards_chosen = torch.tensor(rewards_chosen)
        rewards_rejected = torch.tensor(rewards_rejected)
        #mean_chosen = rewards_chosen.mean()
        #mean_chosen = torch.min(rewards_chosen)
        mean_chosen = rewards_chosen.mean() - 0.5 * rewards_chosen.var()
        sigma_chosen = rewards_chosen.std()
        #mean_rejected = rewards_rejected.mean()
        #mean_rejected = torch.min(rewards_rejected)
        mean_rejected = rewards_rejected.mean() - 0.5 * rewards_rejected.var()
        sigma_rejected = rewards_rejected.std()
        prob = torch.sigmoid(mean_chosen - mean_rejected)

    else:
        raise ValueError("unknown model type")

    if prob > 0.5:
        correct = 1
    elif prob < 0.5:
        correct = 0
    else:
        correct = 0.5

    ece_stat.append((abs(prob-0.5)+0.5, correct))

    row = {'id': example['id'], 'subset': example['subset']}
    row['correct'] = correct
    row['prob'] = prob
    if script_args.model_type not in ["btrm", "brme", "odin"]:
        row['sigma_chosen'] = sigma_chosen
        row['sigma_rejected'] = sigma_rejected
    row['mu_chosen'] = mean_chosen
    row['mu_rejected'] = mean_rejected
    row['nll'] = -torch.log(prob).item()
    df = df._append(row, ignore_index=True)

df_stat = pd.DataFrame(columns=['category', 'subset', 'accuracy'])
for category, subsets in categories.items():
    for subset in subsets:
        df_subset = df[df['subset'] == subset]
        accs = []
        acc = df_subset['correct'].values.mean()
        accs.append(acc)
        row = {'category': category, 'subset': subset, 'n': len(df_subset), 'accuracy': accs}
        df_stat = pd.concat([df_stat, pd.DataFrame(row)], ignore_index=True)
print(df_stat)

all_subsets = df['subset'].unique()
if script_args.ood_type == "domains":
    df_final = pd.DataFrame(columns=['attribute', 'Chat', 'Chat Hard', 'Safety', 'Reasoning', 'poem-cn'])
else:
    df_final = pd.DataFrame(columns=['attribute', 'Chat', 'Chat Hard', 'Safety', 'Reasoning'])

attribute = 'correct'
metrics = {}
for subset in all_subsets:
    df_subset = df_stat.loc[df_stat['subset'] == subset]
    acc = df_subset['accuracy'].values[0]
    metrics[subset] = acc

# Calculate and print the scores per section
scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
acc_row = {'attribute': attribute, **scores_per_section}
df_final = df_final._append(acc_row, ignore_index=True)

df_final.to_csv(os.path.dirname(record_dir) + "/final.csv")


print('model:', script_args.reward_name_or_path)
with open(record_dir, 'a') as f:
    f.write(script_args.reward_name_or_path + "\n")
    f.write(script_args.model_type + "\n")
    f.write(script_args.data_set_name + "\n")
    for col in SUBSET_MAPPING.keys():
        print(f"{col}: {df_final[col].values[0]}")

        f.write(col + "\t" + str(df_final[col].values[0]) + "\n")
        domain_df = df[df['subset'].isin(SUBSET_MAPPING[col])]
        nll = domain_df['nll'].values.mean()
        f.write(f"NLL: {nll:.3f}\n")

        if script_args.model_type not in ["btrm", "brme", "odin"]:
            # 从df中筛选出当前domain的数据
            # 收集所有的mu和sigma
            mus = []
            sigmas = []
            for _, row in domain_df.iterrows():
                mus.append(row['mu_chosen'])
                sigmas.append(row['sigma_chosen'])
                mus.append(row['mu_rejected']) 
                sigmas.append(row['sigma_rejected'])
            bc, A = calculate_average_overlap_degree(mus, sigmas)
            #保留3位小数
            f.write(f"Average overlap degree: {A:.3f}\n")

        else:
            mus = []
            for _, row in domain_df.iterrows():
                mus.append(row['mu_chosen'])
                mus.append(row['mu_rejected'])
            mean_reward = torch.tensor(mus).mean()
            f.write(f"Mean reward: {mean_reward:.3f}\n")


    ece = calculate_ece(ece_stat)
    f.write(f"ECE: {ece:.3f}\n")

# 计算所有样本的平均准确率和负对数似然
average_acc = df['correct'].mean()
average_nll = df['nll'].mean()

print(f"Average accuracy: {average_acc:.3f}")
print(f"Average NLL: {average_nll:.3f}")

with open(record_dir, 'a') as f:
    f.write(f"Average accuracy: {average_acc:.3f}\n")
    f.write(f"Average NLL: {average_nll:.3f}\n")