from lavis.common.registry import registry
import os
# lavis_root = "/home/hsh/LAVIS/lavis"
# registry.register_path("library_root", lavis_root)
import torch
from PIL import Image
import json
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

# ========= 配置 =========
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "blip2_feature_extractor"
model_type = "pretrain" 

input_path = "/mnt/disk60T/dataset/Culture/Museum/Final_version/Final_Zhejiang.json"
output_path = "/home/hsh/data/zj_blip2.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

model, vis_processors, txt_processors = load_model_and_preprocess(
    name=model_name,
    model_type=model_type,
    is_eval=True,
    device=device,
)
print("blip2_feature_extractor 加载完成")

# === text encoding ===
@torch.no_grad()
def encode_text(text: str):
    if not text or not isinstance(text, str):
        raise ValueError(f"text is required")
    text_input = txt_processors["eval"](text)
    tokens = model.tokenizer(text_input, max_length=512, truncation=True, return_tensors="pt").to(device)
    text_output = model.Qformer.bert(tokens.input_ids, attention_mask=tokens.attention_mask, return_dict=True,)
    text_embeds = text_output.last_hidden_state # ([1, len, 768])
    text_embed_cls = text_embeds[:, 0, :]  # [1, 768] - 使用 CLS token
    # text_embed_mean = text_embeds.mean(dim=1)  # 所有 token 的平均值
    return text_embed_cls.cpu().numpy().tolist()

# === image encoding ===
@torch.no_grad()
def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise ValueError(f"image is required")
    raw_image = Image.open(image_path).convert("RGB")
    image_input = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    img_features = model.extract_features({"image": image_input}, mode="image")
    img_embeds = img_features.image_embeds  # [1, 32, 768]
    img_embeds_mean = img_embeds.mean(dim=1)    # [1, 768] - 平均所有 query tokens
    return img_embeds_mean.cpu().numpy().tolist()

# ==============================
# 主循环
# ==============================
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data, desc=f"编码 {os.path.basename(input_path)}"):
        # ===== 1. 简介文本编码 =====
        jianjie = item.get("简介", "")
        jianjie_emb = encode_text(jianjie)
        item["jianjie_embedding_blip2"] = jianjie_emb

        # ===== 2. 拼接结构化信息并编码 =====
        name = item.get("名称", "未知")
        era = item.get("藏品年代", "未知")
        level = item.get("藏品级别", "未知")
        size = item.get("尺寸(cm)", "未知")
        ctype = item.get("馆藏类型", "未知")

        structured_text = f"名称：{name}，年代：{era}，级别：{level}，尺寸：{size}，类型：{ctype}"
        structured_emb = encode_text(structured_text)
        item["structured_embedding_blip2"] = structured_emb

        # ===== 3. 图片编码 =====
        image_paths = item.get("images", [])
        image_embeddings = {}
        for img_path in image_paths:
            emb = encode_image(img_path)
            if emb is not None:
                image_embeddings[img_path] = emb
        item["image_embeddings_blip2"] = image_embeddings

# ===== 保存结果 =====
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 所有文件编码完成，结果已保存到：", output_path)
