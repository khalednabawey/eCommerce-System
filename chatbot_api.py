from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Load the model and tokenizer globally (adjust the model name accordingly)
# model_name = "khalednabawi11/fine_tuned_dialo-gpt"
model_name = "khalednabawi11/fine_tuned_gpt-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Pydantic model for incoming request body


class ChatRequest(BaseModel):
    prompt: str


def clean_response(response):
    # Remove placeholders that look like {{Token}} using regular expressions
    cleaned_response = re.sub(r"\{\{.*?\}\}", "", response).strip()
    cleaned_response = re.sub(r'\s+', ' ', response).strip()
    return cleaned_response


def generate_response(prompt, max_length=150):
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define tokens to remove from the prompt
    tokens_to_remove = [
        '{{Order Number}}', '{{Website URL}}', '{{Account Type}}', '{{Person Name}}',
        '{{Account Category}}', '{{Currency Symbol}}', '{{Refund Amount}}',
        '{{Delivery City}}', '{{Delivery Country}}', '{{Invoice Number}}'
    ]

    # Clean up the prompt by removing unnecessary placeholders
    clean_prompt = prompt
    for token in tokens_to_remove:
        clean_prompt = clean_prompt.replace(token, "").strip()

    # Tokenize input and move inputs to the correct device
    inputs = tokenizer(
        clean_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate outputs with specified parameters for better control over generation
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        # Adjust based on input length
        max_length=max_length + inputs["input_ids"].shape[1],
        do_sample=True,
        temperature=0.6,
        no_repeat_ngram_size=3,
        num_beams=3,
        early_stopping=True,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.2
    )

    # Decode the response and move back to CPU
    response = tokenizer.decode(outputs[0].to('cpu'), skip_special_tokens=True)

    # Remove the clean prompt from the response
    response = response.replace(clean_prompt, '').strip()

    # Remove any unwanted tokens from the response
    for token in tokens_to_remove:
        response = response.replace(token, "").strip()

    # Split response by new lines and strip spaces
    lines = [line.strip()
             for line in re.split(r'\n', response) if line.strip()]

    # Clean up periods before digits if needed
    lines = [re.sub(r'\s*\.\s*(?=\d)', "", line) for line in lines]

    # Join the lines into a single response, formatting numbered steps properly
    response = '\n'.join(lines).strip()

    # Ensure the response ends on a complete sentence
    if response:
        last_punct_index = max(response.rfind(
            '.'), response.rfind('!'), response.rfind('?'))
        if last_punct_index != -1:
            response = response[:last_punct_index + 1].strip()

    # Return the final cleaned response
    return response


@app.post("/generate-response")
def chat(request: ChatRequest):
    response = generate_response(request.prompt)
    return {"response": response}


def run_api(host="127.0.0.1"):
    uvicorn.run(app, host=host, port=8000)
