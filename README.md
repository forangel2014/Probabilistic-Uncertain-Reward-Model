### preparation
```
pip install -r requirement.txt
python setting.py --reward_model_path <your_reward_model_path> --policy_model_path <your_policy_model_path>
python process_preference.py
python process_rlhf.py
```

### reward model training
```
bash run_purm.sh
```

### RLHF with PURM
```
cd Open_RLHF
CUDA_VISIBLE_DEVICES=7 python reward_server.py
bash run.sh
```