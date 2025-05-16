import os
import re
import glob
import argparse

def replace_placeholders(reward_model_path, policy_model_path, reverse=False):
    """
    在当前目录下的所有.py和.sh文件中替换路径占位符。
    
    参数:
        reward_model_path: 奖励模型路径
        policy_model_path: 策略模型路径
        reverse: 是否执行逆向替换操作（将实际路径替换回占位符）
    """
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    
    # 获取当前目录下所有.py和.sh文件
    file_paths = glob.glob("**/*.py", recursive=True) + glob.glob("**/*.sh", recursive=True)
    
    # 定义替换映射
    if not reverse:
        replacements = {
            "<your_reward_model_path>": reward_model_path,
            "<your_policy_model_path>": policy_model_path
        }
    else:
        replacements = {
            reward_model_path: "<your_reward_model_path>",
            policy_model_path: "<your_policy_model_path>"
        }
    
    # 遍历文件并进行替换
    for file_path in file_paths:
        # 跳过当前脚本自身
        if os.path.abspath(file_path) == current_script_path:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 检查文件中是否有任何需要替换的内容
            modified = False
            new_content = content
            for old_text, new_text in replacements.items():
                if old_text in content:
                    new_content = new_content.replace(old_text, new_text)
                    modified = True
            
            # 如果有修改，写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"已更新文件: {file_path}")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="替换文件中的模型路径占位符")
    parser.add_argument("--reward_model_path", required=True, help="奖励模型的路径")
    parser.add_argument("--policy_model_path", required=True, help="策略模型的路径")
    parser.add_argument("--reverse", action="store_true", help="是否执行逆向替换（将实际路径替换回占位符）")
    
    args = parser.parse_args()
    
    replace_placeholders(args.reward_model_path, args.policy_model_path, args.reverse)
    
    if args.reverse:
        print("已将模型路径替换回占位符")
    else:
        print(f"已将占位符替换为以下路径:")
        print(f"奖励模型路径: {args.reward_model_path}")
        print(f"策略模型路径: {args.policy_model_path}")

if __name__ == "__main__":
    main()
