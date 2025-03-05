# myapp/summarizer.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("Falconsai/text_summarization")
model = T5ForConditionalGeneration.from_pretrained("Falconsai/text_summarization")

def summarize_text(text):
    # Encode the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary ids
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary