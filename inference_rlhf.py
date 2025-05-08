import json
import jsonlines
import shutil
import argparse
import os
import glob
from pathlib import Path
import subprocess
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import concurrent.futures
import numpy as np
from tqdm import tqdm
import time

# 配置路径
PRETRAIN_MODEL = "<your_policy_model_path>"
# 命令行参数解析
parser = argparse.ArgumentParser(description="使用RLHF模型进行推理")
parser.add_argument("--model_path", type=str, default=PRETRAIN_MODEL,
                    help="模型路径（可以是预训练模型/DeepSpeed检查点路径）或'gpt-4o'使用API")
parser.add_argument("--global_step", type=str, default=None, 
                    help="DeepSpeed检查点的global_step，如'global_step64'")
parser.add_argument("--generate_html", action="store_true",
                    help="生成HTML报告展示所有结果")
parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                    help="OpenAI API密钥，可通过环境变量OPENAI_API_KEY设置")
args = parser.parse_args()

TEST_DATA_PATH = "./dataset/hh_rlhf_test.jsonl"

if args.model_path.lower() == "gpt-4o":
    use_api = True
    OUTPUT_DIR = "./inference_results/gpt-4o"
else:
    use_api = False
    # 设置模型和输出路径
    if args.global_step is not None:
        # 使用DeepSpeed checkpoint
        DEEPSPEED_CKPT_DIR = args.model_path  # 这里是到_actor/为止的路径
        LATEST = args.global_step
        HF_CONVERTED_DIR = f"{DEEPSPEED_CKPT_DIR}/{LATEST}/hf_export"
        is_deepspeed = True
        model_name = f"{Path(DEEPSPEED_CKPT_DIR).name}_{LATEST}"
    else:
        # 使用原始预训练模型
        HF_CONVERTED_DIR = args.model_path
        is_deepspeed = False
        model_name = "pretrain"

    match = re.match(r".*/(.+)/ckpt/_actor/([^/]+)/.*", HF_CONVERTED_DIR)
    if not match:
        raise ValueError(f"路径格式不匹配: {HF_CONVERTED_DIR}")
    
    yyy_part = match.group(1)
    zzz_part = match.group(2)

    OUTPUT_DIR = f"./inference_results/{yyy_part}/{zzz_part}"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = f"{OUTPUT_DIR}/results.jsonl"

def convert_deepspeed_to_hf():
    """将DeepSpeed检查点转换为HuggingFace格式"""
    if not is_deepspeed:
        print(f"使用预训练模型: {HF_CONVERTED_DIR}")
        return
        
    # 强制检查目录是否真实存在并是否已转换
    hf_dir = Path(HF_CONVERTED_DIR)
    if hf_dir.exists() and (hf_dir / "config.json").exists() and (hf_dir / "pytorch_model.bin").exists():
        print(f"检测到已转换的模型: {HF_CONVERTED_DIR}")
        return
    
    print(f"开始转换DeepSpeed检查点到HuggingFace格式")
    
    # 确保目标目录存在
    Path(HF_CONVERTED_DIR).mkdir(parents=True, exist_ok=True)
    
    # 读取和修改latest文件
    latest_file = Path(f"{DEEPSPEED_CKPT_DIR}/latest")
    original_latest = None
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            original_latest = f.read().strip()
        print(f"原始latest文件内容: {original_latest}")
    
    # 修改latest文件内容为目标global_step
    print(f"修改latest文件内容为: {LATEST}")
    with open(latest_file, 'w') as f:
        f.write(LATEST)
    
    try:
        # Step 1: 合并ZeRO分片参数
        print("合并ZeRO分片参数...")
        cmd = f"python -m deepspeed.utils.zero_to_fp32 {DEEPSPEED_CKPT_DIR} {HF_CONVERTED_DIR}/pytorch_model.bin"
        print(f"执行命令: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"错误输出: {result.stderr.decode()}")
            raise RuntimeError(f"参数合并失败")
        
        # Step 2: 复制原始模型配置文件
        print("复制配置文件...")
        required_files = [
            "config.json", "tokenizer_config.json",
            "special_tokens_map.json", "tokenizer.model",
            "generation_config.json", "vocab.json", "tokenizer.json",
            "model.safetensors.index.json"
        ]
        
        for fname in required_files:
            src = Path(PRETRAIN_MODEL) / fname
            if src.exists():
                shutil.copy(src, f"{HF_CONVERTED_DIR}/{fname}")
            else:
                print(f"警告: 缺失配置文件 {fname}")
        
        # Step 3: 验证转换结果
        if not Path(f"{HF_CONVERTED_DIR}/config.json").exists():
            raise FileNotFoundError("config.json 缺失，请确保原始模型包含配置文件")

        print(f"转换完成！转换后模型保存在: {HF_CONVERTED_DIR}")
    finally:
        # 恢复latest文件内容
        if original_latest is not None:
            print(f"恢复latest文件内容为: {original_latest}")
            with open(latest_file, 'w') as f:
                f.write(original_latest)

def prepare_prompts():
    """读取测试数据并使用chat_template构造prompt"""
    # 加载tokenizer来使用chat_template
    if not use_api:
        tokenizer = AutoTokenizer.from_pretrained(HF_CONVERTED_DIR, trust_remote_code=True)
    
    prompts = []
    with jsonlines.open(TEST_DATA_PATH) as reader:
        for obj in reader:
            # # 构建符合chat_template格式的消息列表
            # messages = []
            
            # # 添加系统提示
            # if "sys_prompt" in obj and obj["sys_prompt"]:
            #     messages.append({"role": "system", "content": obj["sys_prompt"]})
            
            # # 添加对话历史
            # for msg in obj["context_messages"]:
            #     role = "assistant" if msg["role"] == "assistant" else "user"
            #     messages.append({"role": role, "content": msg["content"]})
            
            # if not use_api:
            #     # 使用chat_template格式化对话
            #     prompt = tokenizer.apply_chat_template(
            #         messages, 
            #         tokenize=False,
            #         add_generation_prompt=True
            #     )
            # else:
            #     prompt = messages

            """为Qwen模型格式化对话提示"""
            prompt = ""
            messages = obj["context_messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                elif role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            
            # 添加助手响应的开始标记
            prompt += "<|im_start|>assistant\n"

            prompts.append(prompt)

    return prompts

def generate_html_report(output_prefix):
    """生成HTML报告，展示所有模型的推理结果"""
    print("正在生成HTML报告...")
    
    # 查找所有输出文件
    result_files = glob.glob(f"{OUTPUT_DIR}/{output_prefix}_*.jsonl")
    if not result_files:
        print("未找到推理结果文件，无法生成报告")
        return
    
    # 提取模型名称并进行排序处理
    model_info = []
    results_by_model = {}
    
    import re
    
    # 处理每个结果文件
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace(f"{output_prefix}_", "").replace(".jsonl", "")
        
        # 默认值
        step_num = float('inf')  # 大值，确保未匹配的排在最后
        display_name = model_name
        
        # 特殊处理pretrain模型
        if model_name == "pretrain":
            step_num = 0
            display_name = "step 0"
        else:
            # 尝试从名称中提取步数
            match = re.search(r'global_step(\d+)', model_name)
            if match:
                step_num = int(match.group(1))
                display_name = f"step {step_num}"
        
        # 读取结果文件
        with jsonlines.open(file_path) as reader:
            results = list(reader)
            results_by_model[model_name] = results
        
        # 保存模型信息
        model_info.append({
            'original_name': model_name,
            'display_name': display_name,
            'step': step_num,
            'file_path': file_path
        })
    
    # 按步骤数排序
    model_info.sort(key=lambda x: x['step'])
    
    # 提取排序后的模型名称列表
    model_names = [info['original_name'] for info in model_info]
    display_names = [info['display_name'] for info in model_info]
    
    print(f"找到 {len(model_names)} 个模型，按步骤排序: {', '.join(display_names)}")
    
    # 检查是否所有模型都有相同的问题数量
    sample_count = len(results_by_model[model_names[0]])
    for model, results in results_by_model.items():
        if len(results) != sample_count:
            print(f"警告: 模型 {model} 的结果数量 ({len(results)}) 与其他模型不同 ({sample_count})")
    
    # 生成HTML
    html = """<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .question { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin-bottom: 20px;
                display: none;
            }
            .active { display: block; }
            .responses {
                display: grid;
                grid-template-columns: repeat(AUTO_COLS, 1fr);
                gap: 20px;
                margin-top: 15px;
            }
            .response {
                padding: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
                overflow-wrap: break-word;
            }
            .context {
                background-color: #f5f5f5;
                padding: 10px;
                margin-bottom: 15px;
            }
            button { 
                padding: 10px 20px; 
                font-size: 16px;
                margin: 20px 5px;
                cursor: pointer;
            }
            .navigation {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .nav-buttons {
                display: flex;
            }
            .progress {
                font-size: 18px;
                font-weight: bold;
            }
            h3 {
                text-align: center;
                margin-top: 0;
            }
            .model-title {
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 16px;
                color: #333;
                text-align: center;
            }
            .model-comparison-title {
                text-align: center;
                margin: 20px 0;
                font-size: 24px;
            }
        </style>
        <meta charset="UTF-8">
        <title>模型对比结果</title>
    </head>
    <body>
        <div class="container">
            <h1 class="model-comparison-title">RLHF模型迭代效果对比</h1>
            <div class="navigation">
                <div class="nav-buttons">
                    <button onclick="previousQuestion()">上一个问题</button>
                    <button onclick="nextQuestion()">下一个问题</button>
                </div>
                <div class="progress">
                    <span id="current-question">1</span>/<span id="total-questions">TOTAL</span>
                </div>
                <button onclick="printToPDF()" class="print-btn">打印为PDF</button>
            </div>
    """
    
    # 替换自动列数
    html = html.replace("AUTO_COLS", str(len(model_names)))
    html = html.replace("TOTAL", str(sample_count))
    
    # 为每个问题创建页面
    for q_idx in range(sample_count):
        # 获取第一个模型的问题上下文
        sample = results_by_model[model_names[0]][q_idx]
        
        # 构建上下文显示
        context_html = ""
        if "sys_prompt" in sample and sample["sys_prompt"]:
            context_html += f"<div><strong>System:</strong> {sample['sys_prompt']}</div>"
        
        for msg in sample["context_messages"]:
            role = "Assistant" if msg["role"] == "assistant" else "User"
            context_html += f"<div><strong>{role}:</strong> {msg['content']}</div>"
        
        html += f"""
        <div class="question" id="q{q_idx}">
            <h3>问题 {q_idx+1}</h3>
            <div class="context">{context_html}</div>
            <div class="responses">
        """
        
        # 为每个模型添加响应（按排序顺序）
        for i, model_name in enumerate(model_names):
            model_result = results_by_model[model_name][q_idx]
            response = model_result.get("generated_response", "无响应")
            # 使用pre标签保留格式，并转义HTML特殊字符
            response = response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            html += f"""
                <div class="response">
                    <div class="model-title">{display_names[i]}</div>
                    <div class="content"><pre style="white-space: pre-wrap; font-family: inherit;">{response}</pre></div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """

    # 添加JavaScript和CSS
    html += """
        </div>
        <script>
            let current = 0;
            const questions = document.querySelectorAll('.question');
            const totalQuestions = questions.length;
            
            function showQuestion(index) {
                questions.forEach(q => q.classList.remove('active'));
                questions[index].classList.add('active');
                document.getElementById('current-question').textContent = index + 1;
            }
            
            function nextQuestion() {
                current = (current + 1) % questions.length;
                showQuestion(current);
            }

            function previousQuestion() {
                current = (current - 1 + questions.length) % questions.length;
                showQuestion(current);
            }
            
            function printToPDF() {
                // 只打印当前可见的问题
                const activeQuestion = document.querySelector('.question.active');
                
                // 存储当前显示状态
                const otherQuestions = Array.from(questions).filter(q => q !== activeQuestion);
                
                // 隐藏其他所有问题和导航
                otherQuestions.forEach(q => q.style.display = "none");
                document.querySelector('.navigation').style.display = "none";
                
                // 确保当前问题可见
                activeQuestion.style.display = "block";
                
                // 打印当前问题
                setTimeout(() => {
                    window.print();
                    
                    // 恢复原始显示状态
                    document.querySelector('.navigation').style.display = "flex";
                    showQuestion(current);
                }, 300);
            }
            
            // 键盘导航
            document.addEventListener('keydown', function(event) {
                if (event.key === 'ArrowRight') {
                    nextQuestion();
                } else if (event.key === 'ArrowLeft') {
                    previousQuestion();
                }
            });
            
            showQuestion(0);
        </script>
        <style>
            @media print {
                /* 隐藏按钮和其他页面元素 */
                .navigation { display: none !important; }
                .container { max-width: none; margin: 0; padding: 0; }
                
                /* 只显示活动问题 */
                .question.active { 
                    display: block !important;
                    border: none;
                    padding: 0;
                    margin: 0;
                }
                
                /* 调整页面边距 */
                @page {
                    size: auto;
                    margin: 15mm 15mm 15mm 15mm;
                }
                
                /* 确保内容适合打印 */
                .responses {
                    grid-template-columns: repeat(auto-fit, minmax(45%, 1fr));
                }
            }
        </style>
    </body>
    </html>
    """
    
    # 写入HTML文件
    report_path = f"{OUTPUT_DIR}/{output_prefix}_comparison.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"报告已生成: {report_path}")
    return report_path

def call_openai_api(prompts, api_key, model_name="gpt-4o"):
    """调用OpenAI API进行批量推理（带重试机制和多线程加速）"""
    from openai import OpenAI
    import concurrent.futures
    import numpy as np
    from tqdm import tqdm
    import time
    
    client = OpenAI(api_key=api_key)
    results = []
    
    # 创建一个输出文件句柄，用于流式写入结果
    with jsonlines.open(OUTPUT_PATH, "w") as writer, \
         jsonlines.open(TEST_DATA_PATH) as reader:
        
        test_data = list(reader)
        
        for i, (prompt, test_item) in enumerate(zip(tqdm(prompts, desc="API推理进度"), test_data)):
            success = False
            retry_count = 0
            max_retries = 5
            
            while not success and retry_count < max_retries:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        temperature=0.7,
                        max_tokens=512,
                    )
                    generated_text = response.choices[0].message.content.strip()
                    results.append(generated_text)
                    
                    # 将结果添加到原始数据并立即写入
                    test_item["generated_response"] = generated_text
                    writer.write(test_item)
                    
                    success = True
                    print(f"已完成 {i+1}/{len(prompts)} 个样本")
                except KeyboardInterrupt:
                    print("\n用户中断操作，停止处理...")
                    raise
                except Exception as e:
                    retry_count += 1
                    print(f"样本 {i+1} 遇到错误: {e}，正在重试... ({retry_count}/{max_retries})")
                    time.sleep(1)  # 添加短暂延迟，避免立即重试
            
            if not success:
                print(f"样本 {i+1} 在 {max_retries} 次尝试后仍然失败，跳过该样本")
                # 记录失败情况
                test_item["generated_response"] = "API_ERROR: 调用失败"
                writer.write(test_item)
    
    print(f"所有 {len(results)}/{len(prompts)} 个样本处理完成")
    return results

def main():
    if args.generate_html:
        generate_html_report("results")
        return
    
    # 添加API推理分支
    if use_api:
        if not args.api_key:
            raise ValueError("使用GPT-4o需要提供OpenAI API密钥")
            
        print("使用GPT-4o API进行推理...")
        prompts = prepare_prompts()
        print(f"样本数量: {len(prompts)}")
        # 调用函数将直接写入结果
        call_openai_api(prompts, args.api_key)
        
        print(f"API推理完成！结果已保存至 {OUTPUT_PATH}")
        return
    
    # 原有本地模型推理流程 - 改为流式处理和写入
    convert_deepspeed_to_hf()
    llm = LLM(
        model=os.path.abspath(HF_CONVERTED_DIR),
        tensor_parallel_size=1,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["</s>", "\n\n\n"]  # 根据实际情况调整停止符
    )

    # 准备测试数据和提示词
    prompts = prepare_prompts()

    # 执行推理
    outputs = llm.generate(prompts, sampling_params)
    
    # 保存结果

    with jsonlines.open(TEST_DATA_PATH) as reader, \
         jsonlines.open(OUTPUT_PATH, "w") as writer:

        for i, (obj, output) in enumerate(zip(reader, outputs)):
            # 添加生成结果到原始数据
            generated_text = output.outputs[0].text.strip()
            obj["generated_response"] = generated_text
            
            # 写入新文件
            writer.write(obj)
            
            # 打印进度
            if (i+1) % 10 == 0:
                print(f"已处理 {i+1} 个样本...")


    # 打开输入和输出文件，流式处理
    # with jsonlines.open(TEST_DATA_PATH) as reader, \
    #      jsonlines.open(OUTPUT_PATH, "w") as writer:

        # test_data = list(reader)
        
        # for i, (prompt, test_item) in enumerate(zip(prompts, test_data)):
        #     try:
        #         # 单样本推理
        #         output = llm.generate([prompt], sampling_params)[0]
                
        #         # 提取生成文本
        #         generated_text = output.outputs[0].text.strip()
                
        #         # 添加生成结果到原始数据
        #         test_item["generated_response"] = generated_text
                
        #         # 立即写入新文件
        #         writer.write(test_item)
                
        #         # 打印进度
        #         print(f"已处理 {i+1}/{len(prompts)} 个样本...")
                
        #     except Exception as e:
        #         print(f"处理样本 {i+1} 时出错: {e}")
        #         # 记录错误情况
        #         test_item["generated_response"] = f"ERROR: {str(e)}"
        #         writer.write(test_item)
                
        #         # 如果是键盘中断，则退出
        #         if isinstance(e, KeyboardInterrupt):
        #             print("用户中断，已保存当前进度")
        #             break

    print(f"推理完成！结果已保存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()