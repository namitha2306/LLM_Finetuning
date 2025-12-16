import os
import json
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig
import bitsandbytes as bnb


# ================================================================
# 0. WANDB LOGIN
# ================================================================
wandb.login()
os.environ["WANDB_PROJECT"] = "passport-llm"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"



# ================================================================
# 1. PATHS
# ================================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROJECT_ROOT = "/media/dev/big-disk/Namitha/passport-llm"
DATA_PATH = f"{PROJECT_ROOT}/data/dataset.jsonl"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs/qwen_passport_extractor"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2. LOAD DATASET + FORMAT CHATML
# ================================================================
print("Loading dataset...")

raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]

dataset = raw_dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

def format_instruction(example):
    instruction = example["instruction"]
    ocr_text = example["input"]
    expected_json = json.dumps(example["output"], ensure_ascii=False)
    text = f"[INST] {instruction}\n\n{ocr_text} [/INST] {expected_json}"
    return {"text": text}

train_dataset = train_dataset.map(format_instruction)
val_dataset = val_dataset.map(format_instruction)

print("Formatted sample:\n", train_dataset[0]["text"][:300])


# ================================================================
# 3. LOAD MODEL WITH QLoRA (4-bit)
# ================================================================
print("\nLoading model with QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# ================================================================
# 4. LORA CONFIG FOR QWEN
# ================================================================
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


# ================================================================
# 5. TRAINING ARGUMENTS (COMPATIBLE WITH OLD TRANSFORMERS)
# ================================================================
print("\nSetting training arguments...")

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    save_steps=200,
    logging_steps=25,

    warmup_ratio=0.05,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",

    bf16=True,
    max_grad_norm=0.3,
    weight_decay=0.001,

    save_total_limit=3,
    report_to="wandb",    # W&B LOGGING WORKS
)


# ================================================================
# 6. SFT TRAINER
# ================================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_arguments,
    peft_config=peft_config,
)



print("\n--- Starting Finetuning ---\n")
trainer.train()


# ================================================================
# 7. SAVE MODEL
# ================================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✔ Fine-tuning complete! Model saved at: {OUTPUT_DIR}")


# ================================================================
# 8. MANUAL VALIDATION LOSS LOGGING TO WANDB
# ================================================================
print("\nRunning manual evaluation...")

eval_results = trainer.evaluate()
wandb.log({"manual_eval_loss": eval_results["eval_loss"]})

print("Logged manual eval loss to W&B:", eval_results["eval_loss"])


# ================================================================
# 9. INFERENCE FUNCTION FOR QUICK TEST
# ================================================================
# ================================================================
# 8. FIXED INFERENCE FUNCTION FOR QWEN + LoRA + 4-BIT
# ================================================================
def generate_output(model, tokenizer, sample, device="cuda"):
    instruction = sample["instruction"]
    ocr = sample["input"]
    prompt = f"[INST] {instruction}\n\n{ocr} [/INST]"

    # Convert the model to FP16 for safe inference
    model = model.half()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("[/INST]")[-1].strip() if "[/INST]" in text else text.strip()
