set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/cpfs/user/bupo/TempPython/OpenRLHF/"}' \
   -- RAY_BACKEND_WORKER_TIMEOUT_S=300 RAY_timeout_ms=10000 python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 4 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 4 \
   --pretrain /cpfs/user/bupo/backbones/Qwen/Qwen2.5-7B-Instruct \
   --save_steps 64 \
   --micro_train_batch_size 8 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 10000 \
   --prompt_max_len 2048 \
   --generate_max_len 1024 \
   --max_samples 50000 \
   --advantage_estimator reinforce \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.0001 \
   --prompt_data /cpfs/user/bupo/data/intent_rm_dataset/human_model/hm_data_0214_train.jsonl \
   --input_key context_messages \
   --meta_keys answer \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --ckpt_path /cpfs/user/bupo/OpenRLHF/examples/checkpoint/reinforce_hm_rl_0214_add_small_cot_len_r \
   --use_tensorboard /cpfs/user/bupo/OpenRLHFLogs/tensorboard/reinforce_hm_rl_0214_add_small_cot_len_r \
   --use_rl_logging_board /cpfs/user/bupo/OpenRLHFLogs/rl_logging_board/reinforce_hm_rl_0214_add_small_cot_len_r \
   --vllm_sync_backend gloo

# also supports --advantage_estimator rloo