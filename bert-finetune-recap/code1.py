import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载预训练模型（这是已经训练好的基础模型）
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # 这是在大量文本上预训练好的模型
    num_labels=2  # 我们的任务：二分类（比如判断情感正负）
).to(device)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. 准备你的专属数据（这是你要让模型学习的新任务）
train_texts = [
    "这个产品太棒了！",
    "质量很差，不推荐",
    "非常满意",
    "完全浪费钱"
]
train_labels = [1, 0, 1, 0]  # 1=正面，0=负面

# 3. 数据预处理
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_labels)

# 4. 设置微调参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # 只训练3轮（而不是从零开始的几十轮）
    per_device_train_batch_size=2,
    learning_rate=2e-5,  # 使用较小的学习率，避免破坏原有知识
)

# 5. 开始微调（Finetune）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()  # 这一步就是在做 finetune！

# 6. 使用微调后的模型
text = "这个东西很好用"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print(f"预测结果: {'正面' if prediction == 1 else '负面'}")

# Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# {'train_runtime': 4.2791, 'train_samples_per_second': 2.804, 'train_steps_per_second': 1.402, 'train_loss': 0.7045536041259766, 'epoch': 3.0}
# 100%|██████| 6/6 [00:04<00:00,  1.40it/s] 
# 预测结果: 正面

# results
# └─checkpoint-6   (ckpt after (4/2) × 3 steps)
#         config.json (结构配置信息)
#         model.safetensors (微调后的模型权重!!!) <== 这两个是核心文件
#         optimizer.pt (优化器状态，中断可恢复训练)
#         rng_state.pth (随机数生成器状态，保证训练的可重复性)
#         scheduler.pt (学习率调度器状态)
#         trainer_state.json (训练状态日志)
#         training_args.bin (所有训练参数的二进制存储) <== 这五个是训练状态文件（用于恢复训练）