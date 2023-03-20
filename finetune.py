import os

from transformers import AutoTokenizer, AutoModel ,TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from MyDataset import CLMDataset, CLMDataCollator

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value']
)

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# from peft import prepare_model_for_int8_training
# model = prepare_model_for_int8_training(model)

model = get_peft_model(model, peft_config)
print(model)
model.print_trainable_parameters()

train_dataset = CLMDataset()
collate_fn = CLMDataCollator(tokenizer, max_length=2048)

os.environ["WANDB_DISABLED"] = "true"
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    logging_dir='./logs/rn_log',
    learning_rate=1e-3,
    save_steps=1000,
    weight_decay=0.01,
    save_total_limit=4,
    save_strategy='steps',
    # deepspeed=deepspeed_config,
    report_to=None
    # resume_from_checkpoint='./results/checkpoint-12000'
)

model.half()
trainer = Trainer(
    model=model,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    args=training_args
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()
