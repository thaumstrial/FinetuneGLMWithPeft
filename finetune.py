import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, \
    default_data_collator, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import Dataset

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

# In practice, if we only have a few sentences, we can set num_epochs about 100
device = 'gpu'
num_epochs = 1
lr = 1e-3


# process dataset
def get_train_dataset(max_length=2048):
    input_ids = []
    labels = []

    # Refer to  https://github.com/THUDM/GLM
    data = [
        # history, query, response
        ([], '你是谁？\n [gMASK] ', '我是你，你是谁？')
    ]

    # Code from https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py
    for history, query, response in data:

        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        input_ids.append(prompt)
        labels.append(response)

    input_ids = tokenizer(input_ids, return_tensors="pt",
                          padding='max_length', max_length=max_length, truncation=True).input_ids
    labels = tokenizer(labels, return_tensors="pt",
                       padding='max_length', max_length=max_length, truncation=True).input_ids
    return {
        'input_ids': input_ids,
        'labels': labels
    }


train_dataset = Dataset.from_dict(get_train_dataset())
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=1, pin_memory=True
)

# For manual training, you can also use the Trainer from hugginface.

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to(device)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        print(batch)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

