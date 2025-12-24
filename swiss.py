import numpy as np
import random
import pandas as pd 
from copy import deepcopy
import time

overall_model_scores = {}
all_models = [
        'qwen-plus-0728', 'qwen3-next-80b-a3b-thinking', 'DeepSeek-V3.1-Terminus-nothinking', 'Claude-Opus-4.1-nothinking',
        'Gemini-2.5-Pro', 'Gemini-2.5-Flash-Lite', 'Claude-4-Sonnet-nothinking', 'GPT-5-chat', 'Gemini-2.5-Flash',
        'Claude-Sonnet-4.5-nothinking', 'DeepSeek-V3.2-Exp-nothinking', 'Kimi-K2-0711', 'GLM-4.5',
        'DeepSeek-V3.1-Terminus-thinking', 'DeepSeek-V3.2-Exp-thinking', 'qwen3-235b-a22b-instruct-2507', 'qwen3-next-80b-a3b-instruct',
        'Kimi-K2-0905',
        'Gemini-2.5-Flash-Lite-Preview-2509', 'Claude-Sonnet-4.5-thinking', 'GPT-5-high', 'GPT-5-medium', 'Gemini-2.5-Flash.1',
        'Gemini-2.5-Flash-Preview-2509', 'GLM-4.6', 'qwen3-max-0923', "Gemini-3-pro", "GPT-5.1", "DeepSeek-V3.2-thinking"
    ]
for model_name in all_models:
    overall_model_scores[model_name] = []

datasets_selection = [
        [0,1,2,3,4],
        [5,6,7],
        [25,26],
        [28,29,30,32,33],
        [19,20,21,22,23,24],
        [11,12,13,14],
        [15,16,17,18],
        [8,9,10],
        [34,35,36],
        [37,38,39,40],
        [41],
        [49,50,51],
]

for ii in range(3):
    score = pd.read_csv('obm4_new.csv') 
    all_models = [
        'qwen-plus-0728', 'qwen3-next-80b-a3b-thinking', 'DeepSeek-V3.1-Terminus-nothinking', 'Claude-Opus-4.1-nothinking',
        'Gemini-2.5-Pro', 'Gemini-2.5-Flash-Lite', 'Claude-4-Sonnet-nothinking', 'GPT-5-chat', 'Gemini-2.5-Flash',
        'Claude-Sonnet-4.5-nothinking', 'DeepSeek-V3.2-Exp-nothinking', 'Kimi-K2-0711', 'GLM-4.5',
        'DeepSeek-V3.1-Terminus-thinking', 'DeepSeek-V3.2-Exp-thinking', 'qwen3-235b-a22b-instruct-2507', 'qwen3-next-80b-a3b-instruct',
        'Kimi-K2-0905',
        'Gemini-2.5-Flash-Lite-Preview-2509', 'Claude-Sonnet-4.5-thinking', 'GPT-5-high', 'GPT-5-medium', 'Gemini-2.5-Flash.1',
        'Gemini-2.5-Flash-Preview-2509', 'GLM-4.6', 'qwen3-max-0923', "Gemini-3-pro", "GPT-5.1", "DeepSeek-V3.2-thinking"
    ]
    all_datasets = score["名称"].tolist()
    data = score[all_models].values.T
    num_models = len(all_models)
    num_datasets = len(all_datasets)
    score_matrix = data
    num_rounds = 12
    total_scores = np.zeros(num_models)

    # 模拟瑞士轮比赛
    num_simulations = 100000
    all_rankings = np.zeros(num_models).reshape(-1)

    # 淘汰阈值（例如，每轮淘汰1个模型）
    elimination_threshold = ii

    for sim_num in range(num_simulations):
        rankings = list(range(num_models))
        remaining_models = list(range(num_models))  # 保存还在比赛中的模型
        total_scores.fill(0)  # 每次模拟重新初始化得分

        for round_num in range(num_rounds):
            if len(remaining_models) <= 1:
                break  # 如果只剩下一个模型，就可以结束比赛
            
            # 当前轮使用的数据集
            datasets_in_round = datasets_selection[round_num]

            # 根据当前得分排序，排名靠前的模型得分更高
            sorted_rankings = sorted(remaining_models, key=lambda x: total_scores[x], reverse=True)

            # 根据得分将模型分组
            groups = {}
            for model in sorted_rankings:
                score = total_scores[model]
                if score not in groups:
                    groups[score] = []
                groups[score].append(model)

            # 对每个分组内的模型随机匹配
            for score, group in groups.items():
                random.seed(time.time())
                random.shuffle(group)
                for i in range(0, len(group), 2):
                    if i + 1 < len(group):
                        model_a = group[i]
                        model_b = group[i + 1]
                        
                        # 比较模型A和模型B在当前轮次所选择的数据集上的得分
                        score_a = np.mean(score_matrix[model_a, datasets_in_round])
                        score_b = np.mean(score_matrix[model_b, datasets_in_round])

                        # 比较两者得分，得分高的获胜
                        if score_a > score_b:
                            winner = model_a
                            loser = model_b
                        else:
                            winner = model_b
                            loser = model_a
                        
                        # 记录本轮的战绩
                        total_scores[winner] += 1
                        total_scores[loser] += 0

            # 淘汰最低得分的模型
            sorted_by_score = sorted(remaining_models, key=lambda x: total_scores[x])
            eliminated_models = sorted_by_score[:elimination_threshold]  # 淘汰最低得分的模型
            remaining_models = [model for model in remaining_models if model not in eliminated_models]

        # 累积所有模拟的排名结果
        all_rankings += total_scores.reshape(-1)
        # print(sim_num, total_scores)

    # 计算平均得分并打印排名
    average_rankings = all_rankings / num_simulations

    # print(all_rankings)

    for i, rank in enumerate(np.argsort(average_rankings)[::-1]):
        # print(f"Overall Score {average_rankings[rank]}: {all_models[rank]}")
        overall_model_scores[all_models[rank]].append(average_rankings[rank])

with open(f"paper_overall.json", 'w') as f:
    import json
    json.dump(overall_model_scores, f, indent=4, ensure_ascii=False)