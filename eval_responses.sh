#!/bin/bash

#   bash eval_responses.sh hh_rlhf_bte_uwo 64 128 192 256 320 384 448 512 576 640 704 768 832 896 960 1024 1088 1152 1216 1280 1344

# 参数解析
if [ $# -lt 2 ]; then
    echo "用法: $0 <实验名称> <步骤1> [<步骤2> ...]"
    echo "示例: $0 hh_rlhf_btrm 32 64 96 128"
    exit 1
fi

exp_name=$1
shift  # 移除第一个参数，剩余的都是步骤

# 设置最大进程数为步骤数量
max_processes=$#
echo "共有 $max_processes 个步骤，将启动 $max_processes 个并行进程"

# 添加信号处理函数
cleanup() {
    echo "捕获中断信号，终止所有子进程..."
    kill -TERM 0  # 终止整个进程组
    wait
    rm -f $temp_script
    exit 1
}
trap cleanup SIGINT

# 创建临时脚本来运行单个评估
temp_script=$(mktemp)
cat > $temp_script << 'EOF'
#!/bin/bash
# 添加子进程的信号处理
trap "kill -TERM \$pid 2>/dev/null" TERM
exp_name=$1
step=$2
echo "开始评估 $exp_name 的步骤 $step"
python eval_responses.py --exp_name $exp_name --step $step &
pid=$!
wait $pid
echo "完成评估 $exp_name 的步骤 $step"
EOF
chmod +x $temp_script

# 并行执行所有步骤
active_processes=0
for step in "$@"; do
    # 如果达到最大进程数，则等待任意子进程完成
    if [ $active_processes -ge $max_processes ]; then
        wait -n
        active_processes=$((active_processes - 1))
    fi
    
    # 修改进程启动方式，使用进程组
    (
        $temp_script $exp_name $step &
        wait
    ) &
    active_processes=$((active_processes + 1))
    echo "启动进程评估 $exp_name 的步骤 $step (当前活跃进程: $active_processes)"
done

# 等待所有剩余进程完成
wait
echo "所有评估完成！"

# 修改清理部分为EXIT信号处理
trap 'rm -f $temp_script' EXIT
