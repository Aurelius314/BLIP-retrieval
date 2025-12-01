import os
import json
import numpy as np
from tqdm import tqdm
import sys
import torch
from transformers import BlipForImageTextRetrieval, AutoProcessor

# --- 用户配置 ---

# 1. 指定包含所有嵌入向量的最终数据集 JSON 文件
# json_input_path = '/home/hsh/data/combined_blip1.json' 
# json_input_path = '/home/hsh/data/zj_blip1.json' 
# json_input_path = '/home/hsh/Y3S1/data/tw_blip1.json' 
json_input_path = '/home/hsh/data/bj_blip1.json' 

# 2. 加载BLIP1模型用于获取投影层
blip_model_name = "Salesforce/blip-itm-large-coco"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 配置结束 ---


def evaluate_retrieval(json_path):
    """
    执行以结构化文本搜图的检索评测。
    查询(Query): 结构化文本向量（768维）
    被检索库(Gallery): 图片向量（1024维）
    
    注意：需要先通过投影层将文本和图像特征投影到相同维度（通常是256维）
    """
    if not os.path.isfile(json_path):
        print(f"错误: 输入文件 '{json_path}' 不存在。请检查路径配置。")
        sys.exit(1)

    # 加载BLIP1模型以获取投影层
    print("正在加载BLIP1模型以获取投影层...")
    try:
        model = BlipForImageTextRetrieval.from_pretrained(blip_model_name, local_files_only=True)
        model.to(device)
        model.eval()
        
        # 检查模型是否有投影层
        if not hasattr(model, 'text_proj') or not hasattr(model, 'vision_proj'):
            print("警告: 模型没有找到投影层，尝试打印模型结构...")
            print("模型属性:", [attr for attr in dir(model) if not attr.startswith('_')])
            raise AttributeError("模型没有找到 text_proj 或 vision_proj 层")
        
        print("模型加载完成，投影层已就绪")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型已下载并可以访问")
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
        if 'image_embeddings_blip1' in item and 'structured_embedding_blip1' in item:
            all_structured_vectors.append(item['structured_embedding_blip1'])

            correct_img_indices = []
            for img_path, img_vector in item['image_embeddings_blip1'].items():
                all_image_vectors.append(img_vector)
                correct_img_indices.append(current_img_idx)
                current_img_idx += 1

            text_to_correct_img_indices_map[current_text_idx] = correct_img_indices
            current_text_idx += 1

    if not all_structured_vectors or not all_image_vectors:
        print("未能加载足够的向量进行评测，请检查数据。")
        return

    print(f"\n数据加载完毕。共加载 {len(all_structured_vectors)} 个结构化文本向量（查询，768维），{len(all_image_vectors)} 个图片向量（被检索库，1024维）。")
    print("开始通过投影层统一维度...")

    # 转换为numpy数组
    query_vectors_raw = np.array(all_structured_vectors, dtype=np.float32)  # shape: [N, 768]
    gallery_vectors_raw = np.array(all_image_vectors, dtype=np.float32)    # shape: [M, 1024]

    # 使用投影层将特征投影到相同维度
    print("正在投影文本特征（768 -> 256）...")
    with torch.no_grad():
        # 将文本特征投影
        query_vectors_tensor = torch.from_numpy(query_vectors_raw).to(device)  # [N, 768]
        query_vectors_proj = model.text_proj(query_vectors_tensor)  # [N, embed_dim] 通常是 256
        query_vectors = torch.nn.functional.normalize(query_vectors_proj, dim=-1)  # L2归一化
        query_vectors = query_vectors.cpu().numpy().astype(np.float32)

    print("正在投影图像特征（1024 -> 256）...")
    with torch.no_grad():
        # 将图像特征投影
        gallery_vectors_tensor = torch.from_numpy(gallery_vectors_raw).to(device)  # [M, 1024]
        gallery_vectors_proj = model.vision_proj(gallery_vectors_tensor)  # [M, embed_dim] 通常是 256
        gallery_vectors = torch.nn.functional.normalize(gallery_vectors_proj, dim=-1)  # L2归一化
        gallery_vectors = gallery_vectors.cpu().numpy().astype(np.float32)

    print(f"投影完成。文本特征维度: {query_vectors.shape}, 图像特征维度: {gallery_vectors.shape}")
    print("开始计算相似度并进行评测...")

    # 计算相似度（特征已经归一化，直接计算点积即可得到余弦相似度）
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

# 数据加载完毕。共加载 3058 个结构化文本向量（查询，768维），6116 个图片向量（被检索库，1024维）。
# 开始通过投影层统一维度...
# 正在投影文本特征（768 -> 256）...
# 正在投影图像特征（1024 -> 256）...
# 投影完成。文本特征维度: (3058, 256), 图像特征维度: (6116, 256)
# 开始计算相似度并进行评测...
# 评测进度: 100%|█████████████| 3058/3058 [00:00<00:00, 151154.11it/s]

# ==================================================
# 以文搜图 (Structured Text -> Image) 检索评测结果
# ==================================================
#   - 总查询文本数: 3058
#   - Recall@1:  0.07%
#   - Recall@5:  0.16%
#   - Recall@10: 0.33%
# ==================================================
