import os
import re
import csv
import json
import math
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# from scipy.spatial.distance import cosine
# from sentence_transformers import SentenceTransformer

# model_name = "/cpfs/user/bupo/OpenRLHF/examples/checkpoint/reinforce_human_model_rl_0210/merged/global_step1152"
model_name = "/cpfs/user/bupo/backbones/Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# cos_sim_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


def cosine_similarity(str1, str2):
    """
    根据字符串频率计算余弦相似度(0-1)。
    """
    vec1 = Counter(str1)
    vec2 = Counter(str2)
    
    # 计算点积
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    # 计算向量模长
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

    # embedding1 = cos_sim_model.encode(str1)
    # embedding2 = cos_sim_model.encode(str2)
    # similarity = 1 - cosine(embedding1, embedding2)
    
    # del embedding1, embedding2
    # similarity = float(similarity)
    # torch.cuda.empty_cache()
    # return similarity


def get_rewards(pred_response_list, target_response):
    """
    计算奖励。
    """
    rewards = []
    for pred_response in pred_response_list:
        rewards.append(cosine_similarity(pred_response, target_response))
    return rewards


def get_response(prompt: str):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [text], 
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    """
    Test results for test results.
    """
    max_test_num = 1000
    test_file = '/cpfs/user/linke/human_model/test_data/hm_24-10_test_500.jsonl'
    # result_save_file = '/cpfs/user/bupo/OpenRLHF/examples/checkpoint/reinforce_human_model_rl_0210/test_restuls/rl_step_1152.csv'
    result_save_file = '/cpfs/user/bupo/OpenRLHF/examples/checkpoint/reinforce_human_model_rl_0210/test_restuls//baseline.csv'
    
    if not os.path.exists(os.path.dirname(result_save_file)):
        os.makedirs(os.path.dirname(result_save_file))

    with open(test_file, 'r') as f, \
        open(result_save_file, 'w') as f2:
        csv_writer = csv.writer(f2)
        csv_writer.writerow(['prompt', 'response', 'predict_label', 'target_label', 'rewards', 'bon_reward'])
        
        all_lines = f.readlines()[:max_test_num]
        
        for line in tqdm(all_lines):
            data = json.loads(line)
            try:
                target = data['answer']
                response = get_response(
                    data['context_messages'][0]['content']
                )

                response = response.replace('<|im_end|>', '').replace('\n', '')
                predict_response = re.findall(
                    r'<answer>(.*?)</answer>', 
                    response
                )[0]
                 
                predict_response_list = eval(predict_response)
                rewards = []
                for predict_response in predict_response_list:
                    reward = cosine_similarity(predict_response, target)
                    rewards.append(reward)

                bon_reward = max(rewards)
                csv_writer.writerow([
                    data['context_messages'][0]['content'],
                    response,
                    predict_response_list,
                    target,
                    rewards,
                    bon_reward
                ])
            except:
                csv_writer.writerow([
                    data['context_messages'][0]['content'],
                    response,
                    [],
                    target,
                    [],
                    "-"
                ])
    

if __name__ == '__main__':
    main()