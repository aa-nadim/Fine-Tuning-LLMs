import replicate

import os
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoTokenizer
from datasets import load_dataset
import wandb

wandb.init(project="flux_train_replicate", entity="aa-nadim")

# Set up training parameters
training_args = TrainingArguments(
    output_dir="./results", 
    evaluation_strategy="steps",
    learning_rate=0.0004,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=100, 
    report_to="wandb", 
    gradient_checkpointing=False, 
    no_cuda=True,
)


model_name = "black-forest-labs/FLUX.1-schnell" 
model = AutoModelForImageClassification.from_pretrained(model_name)

dataset = load_dataset("/akib.zip")

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Preprocessing dataset
dataset = dataset.map(preprocess_function, batched=True)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  
    eval_dataset=dataset["test"],  
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Log final model to WandB (optional)
wandb.log({"final_model": model})
