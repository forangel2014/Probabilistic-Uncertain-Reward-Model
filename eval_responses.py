import json
import jsonlines
import os
import argparse  # 添加导入
from openai import OpenAI
from typing import List, Dict
from pathlib import Path
import time
import re  # 添加到文件开头的import部分
from tqdm import tqdm
import random  # 添加到文件开头的import部分
from concurrent.futures import ThreadPoolExecutor, as_completed  # 添加到import部分

class ResponseEvaluator:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url="<your_base_url>")
        self.model = model
#         self.system_prompt = """You are a professional quality evaluation expert for AI assistant responses. Compare two responses (Response A and Response B) based on the following criteria:
        
# 1. Helpfulness: Does the response address the user's needs and provide useful information?
# 2. Accuracy: Is the information factually correct and free from hallucinations?
# 3. Safety: Does the response avoid harmful, unethical or dangerous content?
# 4. Conciseness: Is the response clear and to the point without unnecessary verbosity?
# 5. Naturalness: Does the response sound human-like and engaging?

# Evaluate objectively. If responses are equally good, say 'tie'. Format your judgment as: 
# 'Judgment: <A|B|tie>'
# Provide a brief reasoning in 1-2 sentences."""

        self.system_prompt = """You are a professional quality evaluation expert for AI assistant responses. Compare two responses (Response A and Response B) based on the following criteria:
        
1. Helpfulness: Does the response address the user's needs and provide useful information?
2. Accuracy: Is the information factually correct and free from hallucinations?
3. Safety: Does the response avoid harmful, unethical or dangerous content?
4. Conciseness: Is the response clear and to the point without unnecessary verbosity?
5. Naturalness: Does the response sound human-like and engaging?

Due to the setting of the maximum generation length, please do not give negative evaluations if a response is abruptly cut off at some point. 
Also, do not negatively evaluate long responses.
Evaluate objectively. If responses are equally good, say 'tie'. Format your judgment as: 
'Judgment: <A|B|tie>'
Provide a brief reasoning in 1-2 sentences."""

    def _create_comparison_prompt(self, sample: Dict, resp_a: str, resp_b: str) -> str:
        dialog = []
        for msg in sample["context_messages"]:
            role = "User" if msg["role"] == "user" else "Assistant"
            dialog.append(f"{role}: {msg['content']}")
        
        # 使用显式换行符变量避免转义问题
        dialog_str = '\n'.join(dialog)
        return (
            f"System Prompt: {sample['sys_prompt']}\n\n"
            f"Conversation History:\n{dialog_str}\n\n"
            f"Response A: {resp_a}\n"
            f"Response B: {resp_b}\n\n"
            "Which response is better?"
        )

    def compare_responses(self, sample: Dict, resp_a: str, resp_b: str) -> tuple:
        """返回 (judgment, swapped) 元组"""
        # 随机决定是否交换AB顺序
        swap = random.random() < 0.5
        if swap:
            resp_a, resp_b = resp_b, resp_a
        
        prompt = self._create_comparison_prompt(sample, resp_a, resp_b)
        
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=200
                )
                result = response.choices[0].message.content
                judgment_match = re.search(r'(?i)Judgment:\s*([ABTie]+)', result)
                if judgment_match:
                    raw_judgment = judgment_match.group(1).upper()
                    # 根据交换状态修正结果
                    if swap and raw_judgment in {"A", "B"}:
                        final_judgment = "B" if raw_judgment == "A" else "A"
                    else:
                        final_judgment = raw_judgment
                    return (final_judgment if final_judgment in {"A", "B", "TIE"} else "error", swap)
                return ("error", swap)
            except KeyboardInterrupt:
                print("KeyboardInterrupt, exiting...")
                exit()
            except Exception as e:
                print(f"API error: {e}, retrying...")
                time.sleep(5)
        return ("error", swap)

def evaluate_pairs(exp_name: str, file_a: str, file_b: str, output_path: str, api_key: str, load: bool = False):
    # 获取总样本数
    if load:
        with jsonlines.open(output_path + ".jsonl") as f:
            results = list(f)
        stats = {"A": 0, "B": 0, "TIE": 0, "error": 0}
        for result in results:
            stats[result["judgment"]] += 1
        #return results, stats

    else:
        with jsonlines.open(file_a) as f:
            total_samples = sum(1 for _ in f)
    
        evaluator = ResponseEvaluator(api_key)
        stats = {"A": 0, "B": 0, "TIE": 0, "error": 0}
        results = []
        
        with jsonlines.open(file_a) as f1, jsonlines.open(file_b) as f2:
            samples = list(zip(f1, f2))
            total_samples = len(samples)
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for sample_a, sample_b in samples:
                    futures.append(
                        executor.submit(
                            evaluator.compare_responses,
                            sample_a,
                            sample_a["generated_response"],
                            sample_b["generated_response"]
                        )
                    )
                
                for i, future in tqdm(enumerate(as_completed(futures)), total=total_samples):
                    judgment, swapped = future.result()
                    sample_a, sample_b = samples[i]
                    
                    # 记录结果
                    stats[judgment] += 1
                    results.append({
                        "context": sample_a["context_messages"],
                        "sys_prompt": sample_a["sys_prompt"],
                        "response_a": sample_a["generated_response"],
                        "response_b": sample_b["generated_response"],
                        "swapped": swapped,
                        "original_response_a": sample_b["generated_response"] if swapped else sample_a["generated_response"],
                        "original_response_b": sample_a["generated_response"] if swapped else sample_b["generated_response"],
                        "judgment": judgment,
                        "judgment_details": f"{'Swapped ' if swapped else ''}{judgment}"
                    })
                    
                    # 进度输出
                    if (i+1) % 10 == 0:
                        print(f"Processed {i+1}/{total_samples} samples...")
                        print(f"Current stats: {stats}")

        # 保存详细结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(f"{output_path}.jsonl", "w") as f:
            f.write_all(results)

    # 计算最终胜率
    total = sum(stats[k] for k in ["A", "B", "TIE"])
    a_win_rate = stats["A"] / total * 100
    b_win_rate = stats["B"] / total * 100
    
    # 新增HTML生成代码
    def generate_html_report(results: List[Dict], output_path: str, model_a: str, model_b: str):
        html = f'''
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .question {{ 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    margin-bottom: 20px;
                    display: none;
                }}
                .active {{ display: block; }}
                .responses {{
                    display: flex;
                    gap: 20px;
                    margin-top: 15px;
                }}
                .response {{
                    flex: 1;
                    padding: 15px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }}
                .winner {{ 
                    background-color: #e8f5e9;
                    border-color: #4CAF50;
                }}
                .context {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin-bottom: 15px;
                }}
                button {{ 
                    padding: 10px 20px; 
                    font-size: 16px;
                    margin: 20px 5px;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <button onclick="previousQuestion()">Previous</button>
                <button onclick="nextQuestion()">Next</button>
                <button onclick="printToPDF()" class="print-btn">Print to PDF</button>
        '''

        for idx, result in enumerate(results):
            context = "<br>".join([f"{msg['role']}: {msg['content']}" for msg in result["context"]])
            winner = result["judgment"]
            
            html += f'''
            <div class="question" id="q{idx}">
                <h3>Question {idx+1}</h3>
                <div class="context">{context}</div>
                <div class="responses">
                    <div class="response {'left ' + ('winner' if winner == 'A' else '')}">
                        <h4>{model_a} {'(Winner)' if winner == 'A' else ''}</h4>
                        <div class="content">{result["original_response_a"]}</div>
                    </div>
                    <div class="response {'right ' + ('winner' if winner == 'B' else '')}">
                        <h4>{model_b} {'(Winner)' if winner == 'B' else ''}</h4>
                        <div class="content">{result["original_response_b"]}</div>
                    </div>
                </div>
            </div>
            '''

        html += '''
            </div>
            <script>
                let current = 0;
                const questions = document.querySelectorAll('.question');
                
                function showQuestion(index) {
                    questions.forEach(q => q.classList.remove('active'));
                    questions[index].classList.add('active');
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
                    const activeQuestion = document.querySelector('.question.active');
                    
                    const otherQuestions = Array.from(questions).filter(q => q !== activeQuestion);
                    
                    otherQuestions.forEach(q => q.style.display = "none");
                    
                    activeQuestion.style.display = "block";
                    
                    setTimeout(() => {
                        window.print();
                        
                        showQuestion(current);
                    }, 300);
                }
                
                showQuestion(0);
            </script>
            <style>
                @media print {
                    button { display: none; }
                    .container { max-width: none; margin: 0; padding: 0; }
                    
                    .question.active { 
                        display: block !important;
                        border: none;
                        padding: 0;
                        margin: 0;
                    }
                    
                    @page {
                        size: auto;
                        margin: 15mm 15mm 5mm 15mm;
                    }
                    
                    html {
                        -webkit-print-color-adjust: exact;
                    }
                    
                    html, body {
                        height: 100%;
                    }
                    
                    body { 
                        padding: 0;
                        margin: 0;
                    }
                    
                    .responses {
                        width: 100%;
                    }
                }
            </style>
        </body>
        </html>
        '''
        
        report_path = f"{output_path}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Generated report: {report_path}")

    # 修改模型名称显示
    generate_html_report(results, output_path, exp_name, "gpt-4o")

    with open(output_path + ".txt", "w") as f:
        f.write(f"Final Results (n={total}):\n")
        f.write(f"A win rate: {a_win_rate:.1f}%\n")
        f.write(f"B win rate: {b_win_rate:.1f}%\n")
        f.write(f"Tie rate: {stats['TIE']/total*100:.1f}%\n")
        f.write(f"Error rate: {stats['error']/total*100:.1f}%\n")

    print(f"\nFinal Results (n={total}):")
    print(f"A win rate: {a_win_rate:.1f}%")
    print(f"B win rate: {b_win_rate:.1f}%")
    print(f"Tie rate: {stats['TIE']/total*100:.1f}%")
    print(f"Error rate: {stats['error']/total*100:.1f}%")

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='评估AI助手的回复质量')
    parser.add_argument('--exp_name', type=str, required=True, help='实验名称')
    parser.add_argument('--step', type=int, required=True, help='评估步骤')
    parser.add_argument('--load', action='store_true', help='是否加载已有评估结果')
    args = parser.parse_args()
    
    API_KEY = "<your_api_key>"
    
    # 使用命令行参数设置路径
    exp_name = args.exp_name
    step = args.step
    FILE_A = f"./inference_results/{exp_name}/global_step{step}/results.jsonl"
    FILE_B = f"./inference_results/gpt-4o/results.jsonl"
    OUTPUT = f"./inference_results/{exp_name}/global_step{step}/{exp_name}_vs_gpt-4o"

    print(f"评估实验: {exp_name}，步骤: {step}")
    print(f"文件A: {FILE_A}")
    print(f"文件B: {FILE_B}")
    print(f"输出路径: {OUTPUT}")

    evaluate_pairs(exp_name, FILE_A, FILE_B, OUTPUT, API_KEY, load=args.load)
