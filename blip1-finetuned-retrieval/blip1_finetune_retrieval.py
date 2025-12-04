import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# 把 BLIP 代码目录加入路径，方便直接复用模型和工具函数
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIP_DIR = os.path.join(ROOT_DIR, "BLIP")
if BLIP_DIR not in sys.path:
    sys.path.append(BLIP_DIR)

import ruamel_yaml as yaml  # type: ignore

from models.blip_retrieval import blip_retrieval  # type: ignore
import utils  # type: ignore
from utils import cosine_lr_schedule  # type: ignore


@dataclass
class Sample:
    split: str
    image: str
    image_id: int
    caption: str


class CustomRetrievalTrainDataset(Dataset):
    """简单的 (image, caption) 训练集，直接从 json 里按 split 过滤。"""

    def __init__(self, ann_file: str, image_root: str, split: str, image_size: int = 384):
        super().__init__()
        self.image_root = image_root
        self.split = split

        with open(ann_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        self.samples: List[Sample] = [
            Sample(
                split=ann["split"],
                image=ann["image"],
                image_id=int(ann["image_id"]),
                caption=ann["caption"],
            )
            for ann in all_data
            if ann["split"] == split
        ]

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # 建一个 image_id 映射，和 coco_karpathy_train 接口保持一致
        self.img_ids: Dict[int, int] = {}
        n = 0
        for s in self.samples:
            if s.image_id not in self.img_ids:
                self.img_ids[s.image_id] = n
                n += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = os.path.join(self.image_root, sample.image)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = sample.caption
        return image, caption, self.img_ids[sample.image_id]


class CustomRetrievalEvalDataset(Dataset):
    """
    评估用数据集，模仿 coco_karpathy_retrieval_eval：
    - self.text: 所有文本列表
    - self.image: 所有图片文件名列表
    - self.txt2img / self.img2txt: 检索指标需要的映射
    """

    def __init__(self, ann_file: str, image_root: str, split: str, image_size: int = 384):
        super().__init__()
        self.image_root = image_root

        with open(ann_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # 先按 image_id 聚合成 {image_id: {"image": ..., "captions": [...]}}
        grouped: Dict[int, Dict[str, object]] = {}
        for ann in all_data:
            if ann["split"] != split:
                continue
            img_id = int(ann["image_id"])
            if img_id not in grouped:
                grouped[img_id] = {"image": ann["image"], "captions": []}
            grouped[img_id]["captions"].append(ann["caption"])

        # 转成列表形式
        self.annotation: List[Dict[str, object]] = []
        for img_id, v in grouped.items():
            self.annotation.append(
                {"image_id": img_id, "image": v["image"], "caption": v["captions"]}
            )

        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

        from data.utils import pre_caption  # 直接用 BLIP 原有的文本预处理

        self.text: List[str] = []
        self.image: List[str] = []
        self.txt2img: Dict[int, int] = {}
        self.img2txt: Dict[int, List[int]] = {}

        txt_id = 0
        max_words = 30
        for img_idx, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_idx] = []
            for cap in ann["caption"]:
                self.text.append(pre_caption(cap, max_words))
                self.img2txt[img_idx].append(txt_id)
                self.txt2img[txt_id] = img_idx
                txt_id += 1

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index: int):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, index


def train_one_epoch(model, data_loader, optimizer, epoch, device, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_itm", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_ita", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Train Epoch: [{epoch}]"
    print_freq = 10

    for i, (image, caption, idx) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if epoch > 0:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: f"{meter.global_avg:.3f}" for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def compute_retrieval_scores(model, data_loader, device, config):
    """基本照抄 BLIP 官方 evaluation + itm_eval，用在自定义数据集上。"""
    model.eval()

    print("Computing features for evaluation...")
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            mode="text",
        )
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0, device=device
    )

    # 这里简单起见，不做多卡切分，直接单卡跑
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        encoder_output = image_feats[i].repeat(config["k_test"], 1, 1).to(device)
        encoder_att = torch.ones(
            encoder_output.size()[:-1], dtype=torch.long, device=device
        )
        output = model.text_encoder(
            text_ids[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score + topk_sim

    sims_matrix_t = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0, device=device
    )

    for i, sims in enumerate(sims_matrix_t):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(
            encoder_output.size()[:-1], dtype=torch.long, device=device
        )
        output = model.text_encoder(
            text_ids[i].repeat(config["k_test"], 1),
            attention_mask=text_atts[i].repeat(config["k_test"], 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score + topk_sim

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def compute_itm_metrics(scores_i2t, scores_t2i, txt2img, img2txt) -> Dict[str, float]:
    import numpy as np

    # Images -> Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text -> Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    return {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(__file__), "retrieval_custom.yaml"
        ),
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "output_retrieval"),
        type=str,
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    utils.init_distributed_mode(argparse.Namespace(distributed=False, world_size=1, rank=0))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    print("Creating datasets...")
    train_dataset = CustomRetrievalTrainDataset(
        ann_file=config["ann_file"],
        image_root=os.path.join(ROOT_DIR, config["image_root"]),
        split="train",
        image_size=config["image_size"],
    )
    val_dataset = CustomRetrievalEvalDataset(
        ann_file=config["ann_file"],
        image_root=os.path.join(ROOT_DIR, config["image_root"]),
        split="val",
        image_size=config["image_size"],
    )
    test_dataset = CustomRetrievalEvalDataset(
        ann_file=config["ann_file"],
        image_root=os.path.join(ROOT_DIR, config["image_root"]),
        split="test",
        image_size=config["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size_train"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_val"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size_val"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Creating model...")
    model = blip_retrieval(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=config["init_lr"], weight_decay=config["weight_decay"]
    )

    best = 0.0
    best_epoch = 0

    print("Start training")
    for epoch in range(config["max_epoch"]):
        cosine_lr_schedule(
            optimizer,
            epoch,
            config["max_epoch"],
            config["init_lr"],
            config["min_lr"],
        )

        train_stats = train_one_epoch(model, train_loader, optimizer, epoch, device, config)

        val_i2t, val_t2i = compute_retrieval_scores(model, val_loader, device, config)
        test_i2t, test_t2i = compute_retrieval_scores(model, test_loader, device, config)

        val_result = compute_itm_metrics(
            val_i2t, val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt
        )
        print("Val:", val_result)

        if val_result["r_mean"] > best:
            best = val_result["r_mean"]
            best_epoch = epoch
            save_obj = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(
                save_obj, os.path.join(args.output_dir, "checkpoint_best.pth")
            )

            test_result = compute_itm_metrics(
                test_i2t, test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt
            )
            print("Test:", test_result)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_result.items()},
            "epoch": epoch,
            "best_epoch": best_epoch,
        }
        with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    main()


