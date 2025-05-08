import json, os, shutil, re, random, io
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# 复用现有的辅助函数
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

def calculate_average_overlap_degree(mus, vars):

    n = len(mus)

    mus = torch.stack(mus)  # shape: [n]
    vars = torch.stack(vars)  # shape: [n]
    # 创建网格以进行批量计算

    mu_i = mus.unsqueeze(1)  # shape: [n, 1]
    mu_j = mus.unsqueeze(0)  # shape: [1, n]
    var_i = vars.unsqueeze(1)  # shape: [n, 1]
    var_j = vars.unsqueeze(0)  # shape: [1, n]

    # 批量计算 Bhattacharyya 系数
    sqrt_term = torch.sqrt(2 * var_i * var_j / (var_i**2 + var_j**2))
    exp_term = torch.exp(-(mu_i - mu_j)**2 / (4 * (var_i**2 + var_j**2)))
    bc_matrix = sqrt_term * exp_term

    #让bc_matrix减去自身的对角线矩阵
    bc_matrix = bc_matrix - torch.diag(torch.diag(bc_matrix))
    bc = torch.sum(bc_matrix, dim=1) / (n-1)
    
    # # 只取上三角矩阵的值(不包括对角线)
    # mask = torch.triu(torch.ones_like(bc_matrix), diagonal=1)
    # total_bc = torch.sum(bc_matrix * mask)

    # # 计算平均重叠度
    # A = 2 * total_bc / (n * (n-1))

    A = torch.sum(bc) / n
    return bc, A

def calculate_last_overlap(mus, vars):
    """计算最后一个mu/sigma与之前所有记录的平均重叠度"""
    if len(mus) < 2:
        return torch.tensor(0.0)  # 至少需要两个样本

    mus = torch.stack(mus)  # shape: [n]
    vars = torch.stack(vars)  # shape: [n]

    # 分离最后一个和之前的记录
    prev_mus = mus[:-1]
    prev_vars = vars[:-1]
    last_mu = mus[-1]
    last_var = vars[-1]
    
    # 广播计算
    mu_diff = prev_mus - last_mu
    var_sum_sq = prev_vars**2 + last_var**2
    sqrt_term = torch.sqrt(2 * prev_vars * last_var / var_sum_sq)
    exp_term = torch.exp(-mu_diff**2 / (4 * var_sum_sq))
    
    # 计算单个样本的重叠度并取平均
    bc = torch.mean(sqrt_term * exp_term)
    return bc

if __name__ == '__main__':
    from bottle import request
    import bottle, threading, queue
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_type = "btrm"
    file_name = './exp/brme/rewards_log'
    reward_model_path = "<your_reward_model_path>"
    policy_model_path = "<your_policy_model_path>"

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    f = open(file_name, 'a')

    reward_ls = []
    reward_variance_ls = []

    # 加载reward model
    if model_type == "purm":
        model_path = "./exp/purm/ckpt/checkpoint-2000" 
        num_labels = 2
    elif model_type == "brme":
        model_path = "./exp/brme/ckpt/checkpoint-2000" 
        num_labels = 10
    elif model_type in ["btrm", "bte", "par"]:
        model_path = "./exp/btrm/ckpt/checkpoint-2000" 
        num_labels = 1
    elif model_type == "rrm":
        model_path = "./exp/rrm/ckpt/checkpoint-6000" 
        num_labels = 1
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    if model_type in ["bte"]:
        rm_names = [model_path] + [model_path.replace("btrm", f"btrm{i}") for i in [1,2,3,4]]
        devices = [0, 1, 2, 3, 3]
        reward_models = [AutoModelForSequenceClassification.from_pretrained(
            rm_name,
            torch_dtype=torch.bfloat16,
            num_labels=num_labels
        ).to(device) for rm_name, device in zip(rm_names, devices)]
        reward_model = reward_models[0]
        for rm in reward_models:
            rm.eval()
            rm.requires_grad_(False)
    elif model_type in ["par"]:
        tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            num_labels=num_labels
        ).to('cuda')
        reward_model.eval()
        reward_model.requires_grad_(False)
        ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        ref_model.eval()
        ref_model.requires_grad_(False)
        de_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        if de_tokenizer.pad_token is None:
            de_tokenizer.pad_token = de_tokenizer.eos_token
            de_tokenizer.pad_token_id = de_tokenizer.eos_token_id

        def generate_ref_response(input_ids):
            input_text = de_tokenizer.decode(input_ids[0])
            input_text = input_text.replace("\nanswer: ", "\nAnswer: ").replace("\nquestion: ", "\nQuestion: ")
            try:
                question = input_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
                #question = question.replace("Question:", "")
            except Exception as e:
                print(f"error: {e}")
                print(f"input_text: {input_text}")
            
            # 使用聊天模板包装问题
            messages = [{"role": "user", "content": question}]
            chat_template = tokenizer.apply_chat_template(messages, tokenize=False)
            chat_input_ids = tokenizer(chat_template, return_tensors="pt").input_ids.to(ref_model.device)
            
            # 一次性生成5个不同的回答
            outputs = ref_model.generate(
                chat_input_ids, 
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=5
            )
            
            # 处理所有生成的回答
            ref_responses = []
            for i in range(5):
                # 解码生成的回答
                response_text = tokenizer.decode(outputs[i][chat_input_ids.shape[1]:], skip_special_tokens=True)
                # 重新编码为QA格式
                qa_text = f"Question: {question}\nAnswer: {response_text}"
                # if i == 0:
                #     print(f"**************\n{qa_text}\n**************")
                qa_ids = de_tokenizer(qa_text, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.to('cuda')
                ref_responses.append(qa_ids)

            return ref_responses
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            num_labels=num_labels
        ).to('cuda')
        reward_model.eval()
        reward_model.requires_grad_(False)

    valid_model_path = "./exp/btrm/ckpt/checkpoint-2000" 
    valid_reward_model = AutoModelForSequenceClassification.from_pretrained(
        valid_model_path,
        torch_dtype=torch.bfloat16,
        num_labels=1#num_labels
    ).to('cuda')
    valid_reward_model.eval()
    valid_reward_model.requires_grad_(False)
    def get_valid_reward(input_ids):
        return valid_reward_model(input_ids).logits.squeeze(-1) 

    def get_rewards(input_ids):

        with torch.inference_mode():

            valid_rewards = get_valid_reward(input_ids)
            #print(f"valid_rewards: {valid_rewards}")

            max_length = 1024
            
            # 添加输入验证
            if torch.max(input_ids) >= reward_model.config.vocab_size:
                raise ValueError(f"输入包含无效的token ID: 最大值 {torch.max(input_ids).item()} 超过词表大小 {reward_model.config.vocab_size}")
            
            if torch.min(input_ids) < 0:
                raise ValueError(f"输入包含负数token ID: 最小值 {torch.min(input_ids).item()}")
            
            original_length = input_ids.shape[1]
            if original_length > max_length:
                #print(f"警告：输入长度 {original_length} 超过最大长度 {max_length}，进行截断")
                input_ids = input_ids[:, :max_length]

            if model_type == "purm":
                mu = reward_model(input_ids).logits[:,0]  # (B,)
                log_sigma = reward_model(input_ids).logits[:,1]  # (B,)
                # valid_mu = valid_reward_model(input_ids).logits[:,0]  # (B,)
                # valid_log_sigma = valid_reward_model(input_ids).logits[:,1]  # (B,)
                temperature = 10.0
                sigma = torch.exp(log_sigma/temperature)
                #valid_sigma = torch.exp(valid_log_sigma/temperature)
                return mu, sigma, valid_rewards#valid_mu, valid_sigma

            if model_type == "odin":
                reward_q = reward_model(input_ids).logits[:,0]  # (B,)
                reward_l = reward_model(input_ids).logits[:,1]  # (B,)
                # valid_mu = valid_reward_model(input_ids).logits[:,0]  # (B,)
                # valid_log_sigma = valid_reward_model(input_ids).logits[:,1]  # (B,)
                return reward_q, valid_rewards

            if model_type == "brme":
                logits = reward_model(input_ids).logits
                temperature = 10.0
                mus = []
                sigmas = []
                for i in range(5):
                    mu = logits[:, i]
                    log_sigma = logits[:, i+5]
                    sigma = torch.exp(log_sigma/temperature)
                    mus.append(mu)
                    sigmas.append(sigma)
                mus = torch.stack(mus)
                sigmas = torch.stack(sigmas)
                return mus, sigmas, valid_rewards#valid_mu, valid_sigma

            elif model_type in ["btrm", "rrm"]:
                rewards = reward_model(input_ids).logits.squeeze(-1)  # (B,)
                return rewards, valid_rewards
            
            elif model_type == "data_ensemble":
                rewards = []
                for _ in range(10):
                    reward = reward_model(input_ids).logits.squeeze(-1)  # (B,)
                    rewards.append(reward)
                rewards = torch.stack(rewards)
                print(f"rewards: {rewards.shape}")
                mu = rewards.mean(dim=0)
                sigma = rewards.std(dim=0)
                print(f"mu: {mu.item()}, sigma: {sigma.item()}")
                return mu, sigma, valid_rewards
            
            elif model_type == "bte":
                rewards = []
                for rm in reward_models:
                    reward = rm(input_ids.to(rm.device)).logits.squeeze(-1).to(reward_models[0].device)  # (B,)
                    rewards.append(reward)
                rewards = torch.stack(rewards)
                mu = rewards.mean(dim=0)
                sigma = rewards.std(dim=0)
                return mu, sigma, valid_rewards, rewards

            elif model_type == "par":
                ref_ids = generate_ref_response(input_ids)
                rewards = reward_model(input_ids).logits.squeeze(-1)  # (B,)
                ref_rewards = []
                for ref_id in ref_ids:
                    ref_reward = reward_model(ref_id).logits.squeeze(-1)  # (B,)
                    ref_rewards.append(ref_reward)
                ref_rewards = torch.stack(ref_rewards)
                reward = torch.sigmoid(rewards - ref_rewards).mean(dim=0)
                return reward, valid_rewards

    raw_queue = queue.Queue()
    result_queue = queue.Queue()

    app = bottle.Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        try:
            dd = request.body.read()
            dd = bytes_list_to_list(dd)
            data = {'base': json.loads(dd[0])}
            data['inputs'] = bytes_to_tensor(dd[1])

            if model_type == "purm":
                mu, sigma, valid_mu = get_rewards(data['inputs'].to(reward_model.device))
                rewards = mu
                f.writelines(f"reward: {mu.item()}\n")
                f.writelines(f"valid_reward: {valid_mu.item()}\n")
                # f.writelines(f"sigma: {sigma.item()}\n")
                # f.writelines(f"valid_sigma: {valid_sigma.item()}\n")

                reward_ls.append(mu[0].detach().cpu())
                reward_variance_ls.append(sigma[0].detach().cpu())
                f.writelines(f"len(reward_ls): {len(reward_ls)}\n")
                f.writelines(f"sigma: {reward_variance_ls[-1]}\n")
                
                if len(reward_ls) >= 100:
                    # bc, A = calculate_average_overlap_degree(reward_ls[-10000:], reward_variance_ls[-10000:])
                    # uncertainty = bc[-1]

                    uncertainty = calculate_last_overlap(reward_ls[-1000000:], reward_variance_ls[-1000000:])

                    f.writelines(f"uncertainty: {uncertainty}\n")

                    calibrated_reward = rewards - uncertainty * 30

                    f.writelines(f"calibrated_reward: {calibrated_reward.item()}\n")

                    rewards = calibrated_reward

                response_data = make_bytes_list([
                    json.dumps(data['base']).encode(),
                    tensor_to_bytes(data['inputs']),
                    tensor_to_bytes(rewards)
                ])
                return response_data

            elif model_type == "brme":
                mus, sigmas, valid_rewards = get_rewards(data['inputs'].to(reward_model.device))
                min_sigma_idx = torch.argmin(sigmas)
                nominal = mus[min_sigma_idx]
                robust = torch.min(mus)
                lambda_ = 0.4
                rewards = lambda_ * nominal + (1 - lambda_) * robust

                f.writelines(f"reward: {rewards.item()}\n")
                f.writelines(f"valid_reward: {valid_rewards.item()}\n")
                response_data = make_bytes_list([
                    json.dumps(data['base']).encode(),
                    tensor_to_bytes(data['inputs']),
                    tensor_to_bytes(rewards)
                ])
                return response_data

            elif model_type == "bte":
                mu, sigma, valid_rewards, rewards = get_rewards(data['inputs'].to(reward_model.device))
                # mean
                #rewards = mu
                # WCO
                rewards = torch.min(rewards)
                # UWO
                #rewards = mu - 0.5 * rewards.var()
                f.writelines(f"reward: {mu.item()}\n")
                f.writelines(f"valid_reward: {valid_rewards.item()}\n")
                response_data = make_bytes_list([
                    json.dumps(data['base']).encode(),
                    tensor_to_bytes(data['inputs']),
                    tensor_to_bytes(rewards)
                ])
                return response_data

            elif model_type in ["btrm", "rrm", "odin", "par"]:
                reward, valid_reward = get_rewards(data['inputs'].to(reward_model.device))
                rewards = reward
                f.writelines(f"reward: {reward.item()}\n")
                f.writelines(f"valid_reward: {valid_reward.item()}\n")
                response_data = make_bytes_list([
                    json.dumps(data['base']).encode(),
                    tensor_to_bytes(data['inputs']),
                    tensor_to_bytes(rewards)
                ])
                return response_data

            else:
                raise ValueError(f"unknown model_type: {model_type}")

        except Exception as e:
            print(f"处理请求时发生错误: {str(e)}")
            return bottle.HTTPError(500, str(e))
        #print(data['inputs'])
        raw_queue.put(data)
        print('receive', data['inputs'].shape)

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()

    def run_server(): 
        bottle.run(app, host='0.0.0.0', port=59876, server='tornado')
    
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get()
        rewards = get_rewards(d['inputs'].to(reward_model.device))
        print(f"rewards: {rewards}")
        xdata = make_bytes_list([
            json.dumps(d['base']).encode(),
            tensor_to_bytes(d['inputs']),
            tensor_to_bytes(rewards)
        ])
        result_queue.put(xdata)