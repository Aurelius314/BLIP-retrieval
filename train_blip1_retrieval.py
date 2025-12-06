import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import BlipForImageTextRetrieval, AutoProcessor
from tqdm import tqdm
import urllib.request
from io import BytesIO

# ==============================
# 工具函数
# ==============================
def load_image(image_path: str) -> Image.Image:
    """
    加载图像，兼容URL和本地路径
    Args:
        image_path: 图像路径（URL或本地路径）
    Returns:
        PIL Image对象
    """
    if image_path.startswith(('http://', 'https://')):
        # URL图片：下载后打开
        try:
            with urllib.request.urlopen(image_path, timeout=10) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            raise FileNotFoundError(f"无法加载URL图片 {image_path}: {e}")
    else:
        # 本地路径
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        return Image.open(image_path).convert('RGB')


# ==============================
# 数据类定义
# ==============================
@dataclass
class Sample:
    image: str
    text: str
    group_id: int
    sample_id: int


# ==============================
# 数据集类
# ==============================
class JsonlRetrievalDataset(Dataset):
    """从jsonl文件加载图文检索数据集"""
    
    def __init__(self, jsonl_path: str, text_type: str = None, image_size: int = 384, is_train: bool = True):
        """
        Args:
            jsonl_path: jsonl文件路径
            text_type: 文本类型过滤，'structured' 或 'jianjie'，None表示不过滤
            image_size: 图像尺寸
            is_train: 是否为训练集
        """
        self.text_type = text_type
        self.is_train = is_train
        
        # 加载数据
        self.samples: List[Sample] = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 如果指定了text_type，只加载匹配的数据
                if text_type is None or data.get('text_type') == text_type:
                    self.samples.append(Sample(
                        image=data['image'],
                        text=data['text'],
                        group_id=data['group_id'],
                        sample_id=data['sample_id']
                    ))
        
        print(f"加载了 {len(self.samples)} 个样本 (text_type={text_type})")
        
        # 图像预处理
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ])
        
        # 构建group_id到image_id的映射（用于ITC loss）
        self.group_to_img_id: Dict[int, int] = {}
        img_id = 0
        for sample in self.samples:
            if sample.group_id not in self.group_to_img_id:
                self.group_to_img_id[sample.group_id] = img_id
                img_id += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载图像（兼容URL和本地路径）
        image = load_image(sample.image)
        image = self.transform(image)
        # 返回图像、文本、图像ID
        return image, sample.text, self.group_to_img_id[sample.group_id]


class JsonlRetrievalEvalDataset(Dataset):
    """评估数据集，用于计算检索指标"""
    
    def __init__(self, jsonl_path: str, text_type: str = None, image_size: int = 384):
        self.text_type = text_type
        
        # 按group_id分组
        grouped: Dict[int, Dict] = {}
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if text_type is None or data.get('text_type') == text_type:
                    group_id = data['group_id']
                    if group_id not in grouped:
                        grouped[group_id] = {
                            'image': data['image'],
                            'texts': []
                        }
                    grouped[group_id]['texts'].append(data['text'])
        
        # 转换为列表
        self.annotation: List[Dict] = []
        for group_id, info in grouped.items():
            self.annotation.append({
                'group_id': group_id,
                'image': info['image'],
                'texts': info['texts']
            })
        
        print(f"评估集: {len(self.annotation)} 个图像组, {sum(len(ann['texts']) for ann in self.annotation)} 个文本")
        
        # 构建文本列表和映射
        self.texts: List[str] = []
        self.img2txt: Dict[int, List[int]] = {}
        self.txt2img: Dict[int, int] = {}
        
        txt_id = 0
        for img_idx, ann in enumerate(self.annotation):
            self.img2txt[img_idx] = []
            for text in ann['texts']:
                self.texts.append(text)
                self.img2txt[img_idx].append(txt_id)
                self.txt2img[txt_id] = img_idx
                txt_id += 1
        
        # 图像预处理
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ])
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        ann = self.annotation[idx]
        # 加载图像（兼容URL和本地路径）
        image = load_image(ann['image'])
        image = self.transform(image)
        return image, idx


# ==============================
# 损失计算函数
# ==============================
def compute_itc_loss(image_features, text_features, idx, temperature=0.07):
    """
    计算 Image-Text Contrastive Loss
    image_features: [batch_size, embed_dim]
    text_features: [batch_size, embed_dim]
    idx: [batch_size] - 图像ID，相同ID表示是正样本对
    """
    batch_size = image_features.shape[0]
    
    # 归一化特征
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度矩阵
    sims = image_features @ text_features.t() / temperature  # [batch_size, batch_size]
    
    # 构建正样本标签：相同idx的位置为正样本
    labels = (idx.unsqueeze(1) == idx.unsqueeze(0)).float()  # [batch_size, batch_size]
    labels = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)  # 归一化，避免除零
    
    # Image-to-Text loss
    loss_i2t = -torch.sum(labels * F.log_softmax(sims, dim=1), dim=1).mean()
    
    # Text-to-Image loss
    loss_t2i = -torch.sum(labels.t() * F.log_softmax(sims.t(), dim=1), dim=1).mean()
    
    loss_itc = (loss_i2t + loss_t2i) / 2
    return loss_itc


def get_enc_token_id(processor):
    """获取encoder token ID，参考BLIP官方的init_tokenizer实现"""
    if hasattr(processor.tokenizer, 'enc_token_id'):
        return processor.tokenizer.enc_token_id
    elif hasattr(processor.tokenizer, 'additional_special_tokens_ids') and len(processor.tokenizer.additional_special_tokens_ids) > 0:
        # BLIP官方做法：使用additional_special_tokens的第一个作为enc_token_id
        return processor.tokenizer.additional_special_tokens_ids[0]
    elif hasattr(processor.tokenizer, 'cls_token_id') and processor.tokenizer.cls_token_id is not None:
        return processor.tokenizer.cls_token_id
    else:
        # 如果没有，返回None，表示不修改第一个token
        return None


def compute_itm_loss(model, processor, images, texts, image_embeds, idx, device, temperature=0.07):
    """
    计算 Image-Text Matching Loss
    参考 BLIP 官方实现：对每个样本选择一个负样本（通过multinomial采样）
    """
    batch_size = images.shape[0]
    
    # 计算图像和文本特征用于相似度计算（用于选择负样本）
    image_features = model.vision_proj(image_embeds[:, 0, :])
    image_features = F.normalize(image_features, dim=-1)
    
    text_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=35
    ).to(device)
    text_outputs = model.text_encoder(**text_inputs)
    text_features = model.text_proj(text_outputs.last_hidden_state[:, 0, :])
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度矩阵（用于选择负样本）
    with torch.no_grad():
        sim_i2t = image_features @ text_features.t() / temperature
        sim_t2i = text_features @ image_features.t() / temperature
        
        # 构建mask：相同idx的位置为正样本，需要mask掉
        mask = torch.eq(idx.unsqueeze(1), idx.unsqueeze(0))  # [batch_size, batch_size]
        
        # 计算权重（softmax后mask掉正样本）
        weights_i2t = F.softmax(sim_i2t, dim=1)
        weights_i2t.masked_fill_(mask, 0)
        
        weights_t2i = F.softmax(sim_t2i, dim=1)
        weights_t2i.masked_fill_(mask, 0)
    
    # 准备encoder input ids（正样本）
    encoder_input_ids = text_inputs.input_ids.clone()
    enc_token_id = get_enc_token_id(processor)
    if enc_token_id is not None:
        encoder_input_ids[:, 0] = enc_token_id
    
    # 前向传播正样本对
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
    output_pos = model.text_encoder(
        input_ids=encoder_input_ids,
        attention_mask=text_inputs.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True
    )
    
    # 为每个文本选择一个负样本图像（通过multinomial采样）
    image_embeds_neg = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        image_embeds_neg.append(image_embeds[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    
    # 为每个图像选择一个负样本文本（通过multinomial采样）
    text_ids_neg = []
    text_atts_neg = []
    for b in range(batch_size):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(encoder_input_ids[neg_idx])
        text_atts_neg.append(text_inputs.attention_mask[neg_idx])
    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)
    
    # 拼接正负样本
    text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
    text_atts_all = torch.cat([text_inputs.attention_mask, text_atts_neg], dim=0)
    image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
    image_atts_all = torch.cat([image_atts, image_atts], dim=0)
    
    # 前向传播负样本对
    output_neg = model.text_encoder(
        input_ids=text_ids_all,
        attention_mask=text_atts_all,
        encoder_hidden_states=image_embeds_all,
        encoder_attention_mask=image_atts_all,
        return_dict=True
    )
    
    # ITM head预测
    vl_embeddings = torch.cat([
        output_pos.last_hidden_state[:, 0, :],
        output_neg.last_hidden_state[:, 0, :]
    ], dim=0)
    itm_logits = model.itm_head(vl_embeddings)
    
    # ITM标签：正样本为1，负样本为0
    itm_labels = torch.cat([
        torch.ones(batch_size, dtype=torch.long),
        torch.zeros(2 * batch_size, dtype=torch.long)
    ], dim=0).to(device)
    
    loss_itm = F.cross_entropy(itm_logits, itm_labels)
    
    # 清理中间变量
    del image_features, text_features, sim_i2t, sim_t2i, weights_i2t, weights_t2i
    del image_embeds_neg, text_ids_neg, text_atts_neg, text_ids_all, text_atts_all
    del image_embeds_all, image_atts_all, output_pos, output_neg, vl_embeddings, itm_logits
    
    return loss_itm


# ==============================
# 训练函数
# ==============================
def train_one_epoch(model, processor, data_loader, optimizer, epoch, device, alpha=0.4, grad_accum_steps=1):
    model.train()
    total_loss_itc = 0.0
    total_loss_itm = 0.0
    total_samples = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()  # 在循环开始前清零梯度
    
    for batch_idx, (images, texts, idxs) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        idxs = idxs.to(device, non_blocking=True)
        
        # 动态调整alpha
        if epoch > 0:
            current_alpha = alpha
        else:
            current_alpha = alpha * min(1, batch_idx / len(data_loader))
        
        # 提取图像特征
        image_outputs = model.vision_model(pixel_values=images)
        image_embeds = image_outputs.last_hidden_state
        image_features = model.vision_proj(image_embeds[:, 0, :])  # [batch_size, embed_dim]
        
        # 提取文本特征
        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=35
        ).to(device)
        text_outputs = model.text_encoder(**text_inputs)
        text_features = model.text_proj(text_outputs.last_hidden_state[:, 0, :])  # [batch_size, embed_dim]
        
        # 计算 ITC 损失
        loss_itc = compute_itc_loss(image_features, text_features, idxs)
        
        # 计算 ITM 损失（减少计算频率以节省内存）
        if batch_idx % 4 == 0:  # 每4个batch计算一次ITM
            loss_itm = compute_itm_loss(
                model, processor, images, texts, 
                image_embeds, idxs, device
            )
        else:
            loss_itm = torch.tensor(0.0, device=device)
        
        # 总损失（考虑梯度累积）
        loss = (loss_itc + current_alpha * loss_itm) / grad_accum_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积：每grad_accum_steps步更新一次
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss_itc += loss_itc.item() * grad_accum_steps
        total_loss_itm += loss_itm.item() if isinstance(loss_itm, torch.Tensor) else loss_itm
        total_samples += len(images)
        
        # 清理中间变量
        del image_outputs, image_embeds, image_features
        del text_inputs, text_outputs, text_features
        del loss_itc, loss_itm, loss
        
        pbar.set_postfix({
            'loss_itc': f'{total_loss_itc / (batch_idx + 1):.4f}',
            'loss_itm': f'{total_loss_itm / max(1, (batch_idx + 1) // 4):.4f}',
        })
    
    # 处理最后一个不完整的梯度累积批次
    if (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_loss_itc = total_loss_itc / len(data_loader)
    avg_loss_itm = total_loss_itm / max(1, len(data_loader) // 4)
    return {'loss_itc': avg_loss_itc, 'loss_itm': avg_loss_itm}


# ==============================
# 评估函数
# ==============================
@torch.no_grad()
def evaluate(model, processor, eval_dataset, device, k_test=8):
    """计算检索指标"""
    model.eval()
    
    # 计算文本特征
    print("计算文本特征...")
    texts = eval_dataset.texts
    text_embeds = []
    text_batch_size = 256
    
    for i in tqdm(range(0, len(texts), text_batch_size), desc="文本编码"):
        batch_texts = texts[i:i+text_batch_size]
        text_inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=35
        ).to(device)
        
        text_outputs = model.text_encoder(**text_inputs)
        text_embed = model.text_proj(text_outputs.last_hidden_state[:, 0, :])
        text_embed = F.normalize(text_embed, dim=-1)
        text_embeds.append(text_embed)
    
    text_embeds = torch.cat(text_embeds, dim=0)
    
    # 计算图像特征
    print("计算图像特征...")
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)
    image_embeds = []
    image_features_list = []
    
    for images, _ in tqdm(eval_loader, desc="图像编码"):
        images = images.to(device)
        # images 已经是预处理后的张量，直接使用
        image_outputs = model.vision_model(pixel_values=images)
        image_feat = image_outputs.last_hidden_state
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeds.append(image_embed)
        image_features_list.append(image_feat.cpu())
    
    image_embeds = torch.cat(image_embeds, dim=0)
    image_features = torch.cat(image_features_list, dim=0)
    
    # 计算相似度矩阵
    sims_matrix = image_embeds @ text_embeds.t()
    
    # Image-to-Text检索
    score_matrix_i2t = torch.full(
        (len(eval_dataset), len(texts)), -100.0, device=device
    )
    
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_features[i].repeat(k_test, 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        
        topk_texts = [texts[j] for j in topk_idx.cpu().tolist()]
        text_inputs = processor.tokenizer(
            topk_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=35
        ).to(device)
        # 对于 ITM 任务，第一个 token 应该是 encoder token
        enc_token_id = get_enc_token_id(processor)
        if enc_token_id is not None:
            text_inputs.input_ids[:, 0] = enc_token_id
        
        outputs = model.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True
        )
        score = model.itm_head(outputs.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score + topk_sim
    
    # Text-to-Image检索
    sims_matrix_t = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(eval_dataset)), -100.0, device=device
    )
    
    for i, sims in enumerate(sims_matrix_t):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_features[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)
        
        text_inputs = processor.tokenizer(
            [texts[i]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=35
        ).to(device)
        # 对于 ITM 任务，第一个 token 应该是 encoder token
        enc_token_id = get_enc_token_id(processor)
        if enc_token_id is not None:
            text_inputs.input_ids[:, 0] = enc_token_id
        
        outputs = model.text_encoder(
            input_ids=text_inputs.input_ids.repeat(k_test, 1),
            attention_mask=text_inputs.attention_mask.repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True
        )
        score = model.itm_head(outputs.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score + topk_sim
    
    # 计算指标
    import numpy as np
    scores_i2t = score_matrix_i2t.cpu().numpy()
    scores_t2i = score_matrix_t2i.cpu().numpy()
    
    # Image-to-Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for txt_idx in eval_dataset.img2txt[index]:
            tmp = np.where(inds == txt_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    # Text-to-Image
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == eval_dataset.txt2img[index])[0][0]
    
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    return {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'r_mean': (tr1 + tr5 + tr10 + ir1 + ir5 + ir10) / 6
    }


# ==============================
# 主函数
# ==============================
def main():
    parser = argparse.ArgumentParser(description='BLIP1 图文检索 Finetune')
    parser.add_argument('--train_jsonl', type=str, required=True, help='训练集jsonl路径')
    parser.add_argument('--val_jsonl', type=str, required=True, help='验证集jsonl路径')
    parser.add_argument('--test_jsonl', type=str, required=True, help='测试集jsonl路径')
    parser.add_argument('--text_type', type=str, choices=['structured', 'jianjie'], required=True,
                       help='文本类型: structured(结构性文本) 或 jianjie(简介文本)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--pretrained_model', type=str, 
                       default='Salesforce/blip-itm-large-coco',
                       help='预训练模型路径或名称')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--image_size', type=int, default=384, help='图像尺寸')
    parser.add_argument('--alpha', type=float, default=0.4, help='ITC和ITM损失权重')
    parser.add_argument('--k_test', type=int, default=8, help='检索时top-k')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='梯度累积步数，用于减少内存使用')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    print(f"文本类型: {args.text_type}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载模型
    print("加载模型...")
    model = BlipForImageTextRetrieval.from_pretrained(args.pretrained_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.pretrained_model, local_files_only=True)
    model.to(device)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = JsonlRetrievalDataset(
        args.train_jsonl,
        text_type=args.text_type,
        image_size=args.image_size,
        is_train=True
    )
    val_dataset = JsonlRetrievalEvalDataset(
        args.val_jsonl,
        text_type=args.text_type,
        image_size=args.image_size
    )
    test_dataset = JsonlRetrievalEvalDataset(
        args.test_jsonl,
        text_type=args.text_type,
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # 学习率调度器（cosine）
    def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr=0):
        lr = min_lr + (init_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / max_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    import numpy as np
    
    best_score = 0.0
    best_epoch = 0
    
    print("开始训练...")
    for epoch in range(args.epochs):
        # 学习率调度
        cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, min_lr=0)
        
        # 训练
        train_stats = train_one_epoch(
            model, processor, train_loader, optimizer, epoch, device, args.alpha, args.grad_accum_steps
        )
        
        # 验证
        print(f"\nEpoch {epoch} 验证中...")
        val_metrics = evaluate(model, processor, val_dataset, device, args.k_test)
        print(f"验证指标: {val_metrics}")
        
        # 测试
        test_metrics = evaluate(model, processor, test_dataset, device, args.k_test)
        print(f"测试指标: {test_metrics}")
        
        # 保存最佳模型
        if val_metrics['r_mean'] > best_score:
            best_score = val_metrics['r_mean']
            best_epoch = epoch
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'text_type': args.text_type
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f"保存最佳模型 (r_mean={best_score:.2f})")
        
        # 记录日志
        log = {
            'epoch': epoch,
            'train': train_stats,
            'val': val_metrics,
            'test': test_metrics,
            'best_epoch': best_epoch,
            'best_score': best_score
        }
        with open(os.path.join(args.output_dir, 'log.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        print("-" * 60)
    
    print(f"\n训练完成！最佳模型在epoch {best_epoch}, r_mean={best_score:.2f}")


if __name__ == '__main__':
    main()

