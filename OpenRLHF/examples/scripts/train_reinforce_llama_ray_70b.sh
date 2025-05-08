set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/cpfs/user/bupo/TempPython/OpenRLHF/"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --colocate_actor_ref \
   --pretrain /cpfs/user/bupo/backbones/Qwen/Qwen2.5-72B-Instruct \
   --save_steps 128 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --num_episodes 10000 \
   --prompt_max_len 256 \
   --generate_max_len 1792 \
   --max_samples 100000 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.001 \
   --prompt_data /cpfs/user/bupo/TempPython/AIMERL/datasets/AIME_Dataset_1983_2024.jsonl \
   --input_key context_messages \
   --meta_keys answer \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --ckpt_path /cpfs/user/bupo/OpenRLHF/examples/checkpoint/72b_reinforce_aime_1983_2024 \
   --use_tensorboard /cpfs/user/bupo/OpenRLHFLogs/tensorboard/72b_reinforce_aime_1983_2024 \
   --use_rl_logging_board /cpfs/user/bupo/OpenRLHFLogs/rl_logging_board/72b_reinforce_aime_1983_2024 \
   --vllm_sync_backend gloo

# also supports --advantage_estimator rloo