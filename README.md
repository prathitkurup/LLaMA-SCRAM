# LLaMA-SCRAM

LLaMA Student Collaboration Response Analyzing Model is a sentiment anaalysis tool to analyze student group work dynamics. We finetuned the LLaMA large language model on student responses from software engineering courses and tested on various hyperparameters. This repo contains the best model we developed, including the finetuning and inference files as well as data on other created models. Our SCRAM model is currently private but can be accessed through HuggingFace: https://huggingface.co/pk-kpm/HPC-full-275stp-model.  

## Overview
We created this model using Meta's LLaMA 7 billion parameter model. Data for finetuning was collected from student responses in software engineering courses at Bowdoin College and North Carolina State University. We classified 4,300 student responses (as "Positive," "Negative," "Neutral," or "Mixed") by hand and split into train and test sets. LLaMA-SCRAM models were tested and improved by adjusting the prompt and multiple training parameters in the finetuning phase. We analyzed inference results with each variation to identify the best performing models and their parameters.  

## Parameters
We use the following prompt from the Stanford-Alpaca research team for fine-tuning and inference of the SCRAM model:  
 ```
 Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
 
 ### Instruction:
 {instruction}
 
 ### Input:
 {input}
 
 ### Response:
 ```

Training parameters are defined as follows:  
`training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=25,
    save_steps=25,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)`  

## Testing Results
Precision:  0.9723865877712031  
Recall:  0.9517374517374517  
F1 Score:  0.961951219512195  
Accuracy:  0.9372485921158488
