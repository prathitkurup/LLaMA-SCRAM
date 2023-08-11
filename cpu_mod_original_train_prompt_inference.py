# Compiled inference modified for running on CPU
# Run inference on text classification based on original train prompt structure
# Based on Samwit's inference file (for text gen): https://github.com/samwit/llm-tutorials/blob/main/YT_Alpaca7B_Local_Inference.ipynb
# Use 'tutorial_venv': /mnt/research/k.preslermarshall/pkurup/alpaca_lora_tut/tutorial_venv

from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig, BitsAndBytesConfig
import textwrap

import json
import csv

from tqdm import tqdm

# Change as needed:
model_name = "pk-kpm/HPC-full-275stp-model"
eval_input_file = 'full_test_data.json'
eval_output_file = 'results/original_prompt_testing_results_275stp.csv'


tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = LLaMAForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map={"":"cpu"},
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True),
    offload_folder="offload",
    cache_dir="/mnt/research/k.preslermarshall/pkurup/.cache/hf"
)
model = PeftModel.from_pretrained(model, model_name)
def alpaca_talk(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]#.cuda()

    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    generated_response = ""
    for s in generation_output.sequences:
        generated_response = generated_response + (tokenizer.decode(s))
    return generated_response[generated_response.rindex(":") + 2:].strip()

file = open(eval_input_file)
test_data = json.load(file)

fields = ['Comment', 'Actual Sentiment', 'Model-Labeled Sentiment', 'Label Number', 'Converted Actual Sentiment', 'Converted Model-Labeled Sentiment', 'New Comparison Label']

with open(eval_output_file, mode ='w') as file: 
    csvwriter = csv.writer(file) 
    csvwriter.writerow(fields) 

    for element in tqdm(test_data, total = len(test_data)): 
        comment = element["input"]
        actual_sentiment = element["output"]

        input_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
                    ### Instruction:
                    Detect the sentiment of the comment.
                    ### Input: 
                    {comment}
                    ### Response:
                    """

        model_labeled_sentiment = alpaca_talk(input_text)

        number_label = 1 if model_labeled_sentiment.lower() == actual_sentiment.lower() else 0
        
        converted_actual_sentiment = ""
        converted_model_labeled_sentiment = ""

        if actual_sentiment == "Positive" or actual_sentiment == "positive" or actual_sentiment == "Neutral" or model_labeled_sentiment == "neutral":
            converted_actual_sentiment = "Positive"
        elif actual_sentiment == "Negative" or actual_sentiment == "Bad" or actual_sentiment == "negative" or actual_sentiment == "Positive & negative" or actual_sentiment == "positive & negative":
            converted_actual_sentiment = "Negative"
        else:
            converted_actual_sentiment = "UNKNOWN LABEL"
            print(comment)

        if model_labeled_sentiment == "Positive" or model_labeled_sentiment == "positive" or model_labeled_sentiment == "good" or model_labeled_sentiment == "Good" or model_labeled_sentiment == "very good" or model_labeled_sentiment == "Pretty good" or model_labeled_sentiment == "great":
            converted_model_labeled_sentiment = "Positive"
        elif model_labeled_sentiment == "Negative" or model_labeled_sentiment == "Bad" or model_labeled_sentiment == "negative":
            converted_model_labeled_sentiment = "Negative"
        elif model_labeled_sentiment == "Neutral" or model_labeled_sentiment == "neutral":
            converted_model_labeled_sentiment = "Positive"
        elif model_labeled_sentiment == "Positive & negative" or model_labeled_sentiment == "positive & negative" :
            converted_model_labeled_sentiment = "Negative"
        else:
            converted_model_labeled_sentiment = "UNKNOWN LABEL"
            print(comment)
        
        new_number_label = 1 if converted_actual_sentiment.lower() == converted_model_labeled_sentiment.lower() else 0
        
        output_list = [comment, actual_sentiment, model_labeled_sentiment, number_label, converted_actual_sentiment, converted_model_labeled_sentiment, new_number_label]

        csvwriter.writerow(output_list)