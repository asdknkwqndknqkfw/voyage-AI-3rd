import torch
import time
import wandb
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

# 기본 설정
wandb.login()
model_name = "facebook/opt-350m"
dataset_name = "sahil2801/CodeAlpaca-20k"
lora_rs = [8, 128, 256]
lora_alpha = 32
lora_dropout = 0.1


# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 데이터셋 로드 및 전처리
def format_and_tokenize(batch):
    texts = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

dataset = load_dataset(dataset_name)
dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset["train"].column_names)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 실험 루프
results = []

for r in lora_rs:
    print(f"\n🚀 Training LoRA r={r}")
    
    wandb.init(
        project="hanghae99-week8",
        name=f"r{r}",
        group="week8-LoRA",
        reinit=True
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )

    model_lora = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"/tmp/clm-lora-r{r}",
        per_device_train_batch_size=4,
        save_steps=500,
        num_train_epochs=1,
        logging_dir=f"/tmp/logs-r{r}",
        logging_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        max_steps=1500
    )

    trainer = SFTTrainer(
        model=model_lora,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collator
    )

    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    final_loss = train_result.training_loss
    max_memory = round(torch.cuda.max_memory_allocated(0) / 1024**3, 1)

    print(f"✅ Done: r={r}, Loss={final_loss:.4f}, Time={training_time:.1f}s, Max Memory={max_memory}GB")

    results.append({
        "lora_r": r,
        "final_loss": final_loss,
        "training_time": training_time,
        "max_memory": max_memory
    })

    wandb.log({
        "lora_r": r,
        "final_loss": final_loss,
        "training_time": training_time,
        "max_memory": max_memory
    })

    model_lora.save_pretrained(f"/tmp/clm-lora-r{r}")
    wandb.finish()