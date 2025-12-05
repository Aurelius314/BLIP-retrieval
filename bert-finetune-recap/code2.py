# === 实际应用场景 ===

# 场景1: 只想使用训练好的模型（最常见）
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\Administrator\Desktop\crawler\results\checkpoint-6")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("finetuned模型和分词器加载完成！")

# 直接使用
text = "不太行"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)


# # 场景2: 继续训练（如果觉得训练不够）
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# )
# # 从检查点恢复训练
# trainer.train(resume_from_checkpoint="./results/checkpoint-6")
# print("从检查点继续训练！")