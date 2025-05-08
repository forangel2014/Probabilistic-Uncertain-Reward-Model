export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 
export NCCL_P2P_DISABLE=1

set -x 

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/cpfs/user/bupo/TempPython/OpenRLHF/"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --reward_pretrain Qwen/Qwen2.5-7B-Instruct \
   --save_steps 128 \
   --ckpt_path /cpfs/user/bupo/OpenRLHF/examples/checkpoint/aime_1983_2024 \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --num_episodes 100 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /cpfs/user/bupo/TempPython/AIMERL/datasets/AIME_Dataset_1983_2024.jsonl \
   --input_key context_messages \
   --meta_keys answer \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_tensorboard /cpfs/user/bupo/OpenRLHFLogs/tensorboard/aime_1983_2024 \
   --use_rl_logging_board /cpfs/user/bupo/OpenRLHFLogs/rl_logging_board/aime_1983_2024 \
   --ref_reward_offload \
   --vllm_sync_backend gloo

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)