import matplotlib.pyplot as plt
import re
import sys
import os
import json
import numpy as np

def parse_file(file_path):
    subsets = {}
    current_subset = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'Average overlap degree' in line:
                if current_subset:
                    value = float(line.split(': ')[1])
                    subsets[current_subset] = value
                    current_subset = None
            elif line.startswith('NLL:'):
                continue
            else:
                parts = line.split('\t')
                if parts:
                    current_subset = parts[0].strip()
    return subsets

if __name__ == "__main__":
    reverse_ratios = [i/10 for i in range(0, 11)]
    data = {
        'Chat': [],
        'Chat Hard': [],
        'Safety': [],
        'Reasoning': []
    }
    
    for ratio in reverse_ratios:
        file_path = f'./exp/purm_reverse_{ratio:.1f}/reward_bench_purm.txt'
        if os.path.exists(file_path):
            subsets = parse_file(file_path)
            for subset in data:
                data[subset].append(subsets.get(subset, 0))
    
    plt.figure(figsize=(10, 6), facecolor='#f8f8f8')
    ax = plt.gca()
    #ax.set_facecolor('#ebebf2')
    for subset, values in data.items():
        plt.plot(reverse_ratios, values, marker='o', label=subset)
    
    plt.xlabel('Reversal Ratio')
    plt.ylabel('Uncertainty')
    #plt.title('Uncertainty vs Reverse Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('./exp/purm_reverse_0.0/uncertainty_vs_reverse_ratio.pdf', bbox_inches='tight')
    plt.show()
