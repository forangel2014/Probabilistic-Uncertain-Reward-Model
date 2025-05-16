exp_dir=./exp/purm
base_model=<your_reward_model_path>

if [ ! -d ${exp_dir} ];then
    mkdir -p ${exp_dir}
fi

train_files=./dataset/preference_ppo-train
valid_files=./dataset/preference_ppo-valid
cp ./run_purm.sh ${exp_dir}

deepspeed \
    --include localhost:4,5,6,7 \
    --master_port 10170 \
    purm.py \
    --model_name_or_path ${base_model} \
    --train_file_path ${train_files} \
    --valid_file_path  ${valid_files} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${exp_dir}/ckpt \
    --evaluation_strategy  steps \
    --max_eval_samples 8000 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${exp_dir}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 1000 \
    --eval_steps 1000000000000 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ./ds_config.json \
    --ignore_data_skip true \
    --fp16 \
    --fp16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${exp_dir}/train.log