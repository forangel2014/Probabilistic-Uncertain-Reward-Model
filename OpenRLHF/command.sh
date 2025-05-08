ray stop
# 主节点
ray start --head

# 所有其他的节点
ray start --address 10.39.6.0:6379

CUDA_VISIBLE_DEVICES=7 python reward_server.py

tensorboard --logdir=./tensorboard --port=6006