exp_dir=./ppo_exp
exp_name=hh_rlhf_purm
prompt_data=./dataset/hh_rlhf_train.jsonl
pretrain_model=<your_policy_model_path>
reward_model=<your_policy_model_path> #this does not matter

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1 
export NCCL_P2P_DISABLE=1
set -x 

mkdir -p ${exp_dir}/${exp_name}
cp train.sh ${exp_dir}/${exp_name}/

# train_reinforce_llama_ray.sh 修改以下参数
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./OpenRLHF/"}' \
   -- RAY_BACKEND_WORKER_TIMEOUT_S=300 RAY_timeout_ms=10000 python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${pretrain_model} \
   --reward_pretrain ${reward_model} \
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
   --prompt_data ${prompt_data} \
   --input_key context_messages \
   --meta_keys answer \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --ckpt_path ${exp_dir}/${exp_name}/ckpt/ \
   --use_tensorboard ${exp_dir}/${exp_name}/tensorboard/ \
   --use_rl_logging_board ${exp_dir}/${exp_name}/rl_logging_board/ \
   --vllm_sync_backend gloo \
   --max_ckpt_num 10000
# also supports --advantage_estimator rloo

