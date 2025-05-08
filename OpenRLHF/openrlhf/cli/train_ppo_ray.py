import re
import json, os, shutil, re, random, io
import math
import argparse
from typing import List
from datetime import datetime
from collections import Counter

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ray
import torch
import requests
#from xpinyin import Pinyin
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import get_strategy

from scipy.spatial.distance import cosine
#from sentence_transformers import SentenceTransformer


# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    print(f"enter into reward_fn, rewards: {rewards}")
    return torch.stack(rewards).sum(dim=0)


def cosine_similarity(str1, str2):
    print(f"enter into cosine_similarity, str1: {str1}, str2: {str2}")

    """
    根据字符串频率计算余弦相似度(0-1)。
    """
    # 1. No model embedding, rule based.
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
    
    # 2. Use SentenceTransformer model, local.
    # cos_sim_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    # embedding1 = cos_sim_model.encode(str1)
    # embedding2 = cos_sim_model.encode(str2)
    # similarity = 1 - cosine(embedding1, embedding2)
    
    # del cos_sim_model, embedding1, embedding2
    # similarity = float(similarity)
    # torch.cuda.empty_cache()
    # return similarity
    
    # 3. Use SentenceTransformer model, remote.
    # url = "http://10.39.127.251:8000/similarity"
    
    # payload = {
    #     "str1": str1,
    #     "str2": str2
    # }
    
    # try:
    #     response = requests.post(url, json=payload)
    #     response.raise_for_status() 
    #     result = response.json()
    #     return result["similarity"]
    # except requests.exceptions.RequestException as e:
    #     print(f"请求失败: {e}")
    #     return 0.


def get_diversity_score(predict_response_list):
    """
    计算字符串的多样性分数，多样性越差，扣分越多。
    """
    score = 0
    for i in range(len(predict_response_list)):
        for j in range(i + 1, len(predict_response_list)):
            temp_score = cosine_similarity(predict_response_list[i], predict_response_list[j])
            score += temp_score
            # print(f"{temp_score=}")
    try:
        return -score / (len(predict_response_list) * (len(predict_response_list) - 1) / 2)
    except:
        return 0.
    

def cot_length_reward(
    response: str
):
    """
    提取出 cot 的部分，计算长度，长度越长，给分越高（鼓励思考）。
    """
    cot = re.findall(
        r'<think>(.*?)</think>', 
        response
    )
    
    if not cot:
        return 0.
    
    cot = cot[0]
    cot_r = len(cot) / 1024 * 0.05
    return cot_r
    

def sug_score(
    predict_response_list, 
    prompt: str
):
    """
    判断生成答案是否满足用户之前输入的拼音（或汉字）。
    """
    user_input = re.findall(
        r'\<用户当前输入部分\>(.*?)\-',
        prompt,
        re.DOTALL
    )
    
    if not user_input:
        return 0.
    
    p = Pinyin()
    user_input = user_input[0].strip()
    user_input_pinyin = p.get_pinyin(user_input, splitter="")
    # print(f"{user_input_pinyin=}")
    
    penalty = 0.
    for predict_response in predict_response_list:
        # if any(c.isascii() and c.isalpha() for c in predict_response):
        #     penalty += -1.
        # else:
        predict_respponse_pinyin = p.get_pinyin(predict_response, splitter="")
        # print(f"{predict_respponse_pinyin=}")
        if not predict_respponse_pinyin.startswith(user_input_pinyin):
            penalty += -1.
    return penalty
    
'''
def custom_reward_model(
    sequences_str_list: list,
    tokenizer,
    meta_info: dict = {}
) -> torch.Tensor:
    print(f"enter into custom_reward_model, sequences_str_list: {sequences_str_list}")

    """
    自定义 RM，可以是基于规则，或其他方式的 signal 获取方式。

    Args:
        sequences_tensor (_type_): _description_
        tokenizer (_type_): _description_
    """
    # print(f"=" * 20 + " RM FUNC " + "=" * 20)
    # print(f'{sequences_str_list=}')
    # print(f'{meta_info=}')
    # return torch.tensor([0.] * len(sequences_str_list))

    rewards, bon_rewards, diversity_rewards, sug_format_penaltys, average_rewards, cot_length_rs = [], [], [], [], [], []
    pred_right, pred_wrong, format_wrong, sequences, responses, gt = 0, 0, 0, [], [], []
    for sequence, gt_answer in zip(sequences_str_list, meta_info['answer']):
        elements = sequence.split('<|im_start|>assistant')                  # split the response from total sequence
        
        if len(elements) < 2:
            rewards.append(-5.)
            format_wrong += 1
            continue
        
        prompt = elements[0].strip()
        sequence = elements[-1].strip()
        # sequences.append(sequence)
        
        predict_response = re.findall(
            r'<answer>(.*?)</answer>', 
            sequence
        )
        # responses.append(predict_response)
        # gt.append(gt_answer)

        if len(predict_response) != 1:
            rewards.append(-5.)
            format_wrong += 1
            continue
        
        response = predict_response[0]
        required_tags = ['<think>', '</think>', '<answer>', '</answer>']
        valid_format = True

        for tag in required_tags:
            count = sequence.count(tag)
            if count != 1:
                rewards.append(-5.)
                format_wrong += 1
                valid_format = False
                break

        if not valid_format:
            continue
        
        try:
            response_list = eval(response)
            assert isinstance(response_list, list)
            assert len(response_list) == 3
        except:
            rewards.append(-5.)
            format_wrong += 1
            continue
        
        all_rewards = []
        for r in response_list:
            similarity = cosine_similarity(r, gt_answer)
            all_rewards.append(similarity)
        
        bon_reward = max(all_rewards)
        average_reward = sum(all_rewards) / len(all_rewards)
        
        try:
            diversity_score = get_diversity_score(response_list)
        except:
            print("Diversity score error, use 0 instead.")
            diversity_score = 0.
            
        try:
            sug_format_penalty = sug_score(response_list, prompt)
        except:
            print("Sug score error, use 0 instead.")
            sug_format_penalty = 0.
            
        try:
            cot_length_r = cot_length_reward(sequence)
        except:
            print("Cot length reward error, use 0 instead.")
            cot_length_r = 0.
        
        rewards.append(average_reward * 2 + diversity_score + sug_format_penalty + cot_length_r)
        bon_rewards.append(bon_reward)
        average_rewards.append(average_reward)
        diversity_rewards.append(diversity_score)
        sug_format_penaltys.append(sug_format_penalty)
        cot_length_rs.append(cot_length_r)

    if torch.distributed.get_rank() == 0:
        # print(f"{pred_right=}")
        # print(f"{pred_wrong=}")
        # print(f"{format_wrong=}")
        print(f"{rewards=}")
        print(f"{bon_rewards=}")
        print(f"{average_rewards=}")
        print(f"{diversity_rewards=}")
        print(f"{sug_format_penaltys=}")
        print(f"{cot_length_rs=}")
    #     print(f"{rewards=}")
    #     print(f"{sequences=}")
    #     print(f"{responses=}")
    #     print(f"{gt=}")

    # exit()
    return torch.tensor(rewards)
'''

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b))

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def get_rewards_from_rm(q, a, tokenizer):
    try:
        inputs = tokenizer(f"Question: {q}\nAnswer: {a}", return_tensors="pt", padding=True, add_special_tokens=False).input_ids
        xdata = make_bytes_list([json.dumps({}).encode(), tensor_to_bytes(inputs)])
        # 直接从response中获取结果
        reward_server = "http://localhost:59876"
        response = requests.post(f"{reward_server}/upload", data=xdata).content
        dd = bytes_list_to_list(response)
        rewards = bytes_to_tensor(dd[2])
        return rewards
    except Exception as e:
        print(f"获取reward时发生错误: {str(e)}")
        return None

def custom_reward_model(
    sequences_str_list: list,
    tokenizer,
    meta_info: dict = {}
) -> torch.Tensor:
    rewards = []
    tokenizer = AutoTokenizer.from_pretrained("<your_reward_model_path>")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for sequence, gt_answer in zip(sequences_str_list, meta_info['answer']):
        rewards.append(get_rewards_from_rm(sequence, gt_answer, tokenizer))
        
    return torch.tensor(rewards)

def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        args.rollout_batch_size % actor_world_size == 0
    ), f"rollout_bach_size must be divisible by actor_world_size, got {args.rollout_batch_size} and {actor_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

    if args.vllm_num_engines > 0:
        assert (
            actor_world_size % args.vllm_num_engines == 0
        ), f"actor_world_size must be divisible by vllm_num_engines, got {actor_world_size} and {args.vllm_num_engines}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"


def train(args):
    _validate_args(args)

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [
            {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
            for _ in range(args.actor_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
    # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
    # If we allocate 0.5 gpu for both actor and ref model, then gpu allocation is
    #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
    #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
    #
    # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
    #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
    #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.25 if pg else 1,
    )

    # if colocated, create placement group for critic and reward model explicitly.
    pg = None
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [
            {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node}
            for _ in range(args.critic_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = PPORayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.75 if pg else 1,
        )
    else:
        critic_model = None

    # multiple reward models
    if (
        not args.remote_rm_url
        and
        args.reward_pretrain
    ):
        reward_pretrains = args.reward_pretrain.split(",")
        reward_models = []
        for _ in reward_pretrains:
            reward_models.append(
                PPORayActorGroup(
                    args.reward_num_nodes,
                    args.reward_num_gpus_per_node,
                    RewardModelRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                )
            )
    else:
        reward_models = None

    # init reference/reward/actor model
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    if (
        args.reward_pretrain
        and
        not args.remote_rm_url
    ):
        for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
            refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
        )

    ray.get(refs)

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # train actor and critic mdoel
    refs = actor_model.async_fit_actor_model(
        critic_model, 
        ref_model, 
        reward_models, 
        args.remote_rm_url, 
        reward_fn=reward_fn, 
        vllm_engines=vllm_engines,
        custom_reward_model=custom_reward_model
    )
    ray.get(refs)

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic_pretrain and args.save_value_network:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo",
    )

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--meta_keys", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # RL Logging Board parameters
    parser.add_argument("--use_rl_logging_board", type=str, default=None, help="Rollout data saved path for RL Logging Board")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.vllm_num_engines >= 1 and args.enable_prefix_caching:
        args.enable_prefix_caching = False
        print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache.")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    train(args)
