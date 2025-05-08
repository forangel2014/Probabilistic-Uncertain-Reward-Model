step=2000
# 设置默认参数
DATASETS=(
    "allenai/reward-bench"
    # "dzunggg/legal-qa-v1"
    # "Aratako/magpie-sft-v1.0-dpo-judged"
    # "Hello-SimpleAI/HC3-Chinese"
    # "sdiazlor/math-preference-dataset"
    # "argilla/distilabel-math-preference-dpo"
)
# for noise_rate in 0.6 0.7 0.8 0.9 1.0
# do
EXP_DIR="./exp/purm"
REWARD_NAME_OR_PATH="${EXP_DIR}/ckpt/checkpoint-${step}"
MODEL_TYPE="purm"
PROMPT_TEMPLATE="qa"
OOD_TYPE="no"
DEVICE=0
RECORD_DIR="${EXP_DIR}/reward_bench_${MODEL_TYPE}.txt"

# 解析命令行参数
while getopts d:r:n:m:p:o: flag
do
    case "${flag}" in
        d) DATA_SET_NAME=${OPTARG};;
        r) RECORD_DIR=${OPTARG};;
        n) REWARD_NAME_OR_PATH=${OPTARG};;
        m) MODEL_TYPE=${OPTARG};;
        p) PROMPT_TEMPLATE=${OPTARG};;
        o) OOD_TYPE=${OPTARG};;
        d) DEVICE=${OPTARG};;
    esac
done

for DATA_SET_NAME in "${DATASETS[@]}"
do
    # 运行Python脚本
    python eval_preference_data.py \
        --data_set_name $DATA_SET_NAME \
        --record_dir $RECORD_DIR \
        --reward_name_or_path $REWARD_NAME_OR_PATH \
        --model_type $MODEL_TYPE \
        --prompt_template $PROMPT_TEMPLATE \
        --ood_type $OOD_TYPE \
        --device $DEVICE
done
# done