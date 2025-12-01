import os
import json
from tqdm import tqdm
import torch
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor

# ==============================
# 路径配置
# ==============================
input_path = "/mnt/disk60T/dataset/Culture/Museum/Final_version/Final_Zhejiang.json"
output_path = "/home/hsh/data/zj_blip1.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)


# ==============================
# 模型加载 (Transformers)
# ==============================
blip_itm_large_coco_name = "Salesforce/blip-itm-large-coco"
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco", local_files_only=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco", local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("BLIP1-large-for-zj 模型加载完成")

# ==============================
# 编码函数
# ==============================
@torch.no_grad()
def encode_text(text: str):
    if not text or not isinstance(text, str):
        raise ValueError(f"text is required")
    inputs = processor(text=[text], return_tensors="pt", truncation=True).to(device)
    outputs = model.text_encoder(**inputs)
    # 取 [CLS] token embedding
    cls_emb = outputs.last_hidden_state[:, 0, :]  # shape [1, hidden_dim]
    return cls_emb[0].cpu().numpy().tolist()

@torch.no_grad()
def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise ValueError(f"url is required")
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    outputs = model.vision_model(**inputs)
    # 取平均池化后的 embedding
    img_emb = outputs.last_hidden_state.mean(dim=1)  # shape [1, hidden_dim]
    return img_emb[0].cpu().numpy().tolist()

# ==============================
# 主循环
# ==============================
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data, desc=f"编码 {os.path.basename(input_path)}"):
        # ===== 1. 简介文本编码 =====
        jianjie = item.get("简介", "")
        jianjie_emb = encode_text(jianjie)
        item["jianjie_embedding_blip1"] = jianjie_emb

        # ===== 2. 拼接结构化信息并编码 =====
        name = item.get("名称", "未知")
        era = item.get("藏品年代", "未知")
        level = item.get("藏品级别", "未知")
        size = item.get("尺寸(cm)", "未知")
        ctype = item.get("馆藏类型", "未知")

        structured_text = f"名称：{name}，年代：{era}，级别：{level}，尺寸：{size}，类型：{ctype}"
        structured_emb = encode_text(structured_text)
        item["structured_embedding_blip1"] = structured_emb

        # ===== 3. 图片编码 =====
        image_paths = item.get("images", [])
        image_embeddings = {}
        for img_path in image_paths:
            emb = encode_image(img_path)
            if emb is not None:
                image_embeddings[img_path] = emb
        item["image_embeddings_blip1"] = image_embeddings

# ===== 保存结果 =====
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 所有文件编码完成，结果已保存到：", output_path)
