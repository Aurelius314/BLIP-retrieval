import os
import json
import numpy as np
from tqdm import tqdm
import sys

# --- 用户配置 ---

# 1. 指定包含所有嵌入向量的最终数据集 JSON 文件
json_input_path = '/home/hsh/data/bj_blip2_final.json'
# json_input_path = '/home/hsh/data/zj_blip2_final.json'
# json_input_path = '/home/hsh/data/combined_blip2.json'
# json_input_path = '/home/hsh/Y3S1/data/tw_blip2_final.json' 

# --- 配置结束 ---


def evaluate_retrieval(json_path):
    """
    执行以结构化文本搜图的检索评测。
    查询(Query): 结构化文本向量
    被检索库(Gallery): 图片向量
    """
    if not os.path.isfile(json_path):
        print(f"错误: 输入文件 '{json_path}' 不存在。请检查路径配置。")
        sys.exit(1)

    print("正在加载数据并准备向量...")
    
    all_structured_vectors = []
    # 记录每个结构化文本对应的正确的图片向量的索引范围
    text_to_correct_img_indices_map = {}
    all_image_vectors = []
    
    current_img_idx = 0
    current_text_idx = 0
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc="加载JSON文件"):
        if 'image_embeddings_blip2' in item and 'structured_embedding_blip2' in item:
            all_structured_vectors.append(item['structured_embedding_blip2'])

            correct_img_indices = []
            for img_path, img_vector in item['image_embeddings_blip2'].items():
                all_image_vectors.append(img_vector)
                correct_img_indices.append(current_img_idx)
                current_img_idx += 1

            text_to_correct_img_indices_map[current_text_idx] = correct_img_indices
            current_text_idx += 1

    if not all_structured_vectors or not all_image_vectors:
        print("未能加载足够的向量进行评测，请检查数据。")
        return

    print(f"\n数据加载完毕。共加载 {len(all_structured_vectors)} 个结构化文本向量（查询），{len(all_image_vectors)} 个图片向量（被检索库）。")
    print("开始计算相似度并进行评测...")

    # 转换为numpy数组
    query_vectors = np.array(all_structured_vectors, dtype=np.float32)
    gallery_vectors = np.array(all_image_vectors, dtype=np.float32)

    # L2 归一化
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    gallery_vectors /= np.linalg.norm(gallery_vectors, axis=1, keepdims=True)

    # 计算相似度
    similarity_scores = np.dot(query_vectors, gallery_vectors.T)
    sorted_indices = np.argsort(similarity_scores, axis=1)[:, ::-1]

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total_queries = len(query_vectors)

    for i in tqdm(range(total_queries), desc="评测进度"):
        correct_indices = set(text_to_correct_img_indices_map[i])
        
        # 检查Top K结果中是否有任何一个正确的图片
        if any(idx in correct_indices for idx in sorted_indices[i][:1]):
            recall_at_1 += 1
        if any(idx in correct_indices for idx in sorted_indices[i][:5]):
            recall_at_5 += 1
        if any(idx in correct_indices for idx in sorted_indices[i][:10]):
            recall_at_10 += 1

    r1 = (recall_at_1 / total_queries) * 100
    r5 = (recall_at_5 / total_queries) * 100
    r10 = (recall_at_10 / total_queries) * 100
    
    print("\n" + "="*50)
    print("以文搜图 (Structured Text -> Image) 检索评测结果")
    print("="*50)
    print(f"  - 总查询文本数: {total_queries}")
    print(f"  - Recall@1:  {r1:.2f}%")
    print(f"  - Recall@5:  {r5:.2f}%")
    print(f"  - Recall@10: {r10:.2f}%")
    print("="*50)

if __name__ == '__main__':
    evaluate_retrieval(json_input_path)

# 数据加载完毕。共加载 3058 个结构化文本向量（查询），6116 个图片向量（被检索库）。
# 开始计算相似度并进行评测...
# 评测进度: 100%|████████████████| 3058/3058 [00:00<00:00, 148320.71it/s]

# ==================================================
# 以文搜图 (Structured Text -> Image) 检索评测结果
# ==================================================
#   - 总查询文本数: 3058
#   - Recall@1:  0.03%
#   - Recall@5:  0.16%
#   - Recall@10: 0.36%
# ==================================================