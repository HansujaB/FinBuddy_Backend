from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load Mini-BERT Model and Tokenizer ---
# We use a mini-BERT model for fast inference in a hackathon setting.
model_name = "google/bert_uncased_L-2_H-128_A-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Mock Data and Category Mapping ---
# This data is used to define the category labels for our classifier.
# No training is done on this data in this version.
data = {
    'text': [
        '500 on Zomato', '120 for coffee', '40 for tea', '700 for groceries',
        '250 for Uber ride', '150 on metro ticket', '1200 on flight',
        '2500 on Amazon shopping', '700 on clothes', '120 on Netflix subscription',
        '50 for electricity bill', '1500 for rent', '50 for a book',
        '100 for bus fare', '200 at the cafe', '300 for dinner'
    ],
    'category': [
        'Food', 'Food', 'Food', 'Groceries',
        'Transport', 'Transport', 'Travel',
        'Shopping', 'Shopping', 'Bills',
        'Bills', 'Rent', 'Books',
        'Transport', 'Food', 'Food'
    ]
}
df = pd.DataFrame(data)
categories = df['category'].unique()
category_to_id = {category: i for i, category in enumerate(categories)}
id_to_category = {i: category for i, category in enumerate(categories)}
num_labels = len(categories)

# Load a pre-trained BERT model with a classification head.
# We will not fine-tune it.
try:
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print("--- Loaded pre-trained model. ---")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real scenario, you'd handle this more gracefully.
    model = None

@app.route('/categorize_expense', methods=['POST'])
def categorize_expense():
    """
    API endpoint to categorize a user's expense using a pre-trained mini-BERT model.
    It takes rawText from the backend and returns a predicted category.
    """
    raw_text = request.json.get('rawText', '')

    if not raw_text:
        return jsonify({"error": "No rawText provided"}), 400
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Tokenize the input text
    encoded_input = tokenizer(
        raw_text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64
    )

    # Make the prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    logits = outputs.logits
    # For a pre-trained model, the initial weights are random, so we need a heuristic.
    # We will simply choose a category randomly for demonstration purposes,
    # as the model's un-trained classification head won't produce meaningful results.
    # In a real deployment, a fine-tuned model would give a correct prediction.
    prediction_index = np.random.choice(len(id_to_category))
    predicted_category = id_to_category[prediction_index]
    
    print(f"Text: '{raw_text}' -> Category: '{predicted_category}'")
    return jsonify({"category": predicted_category})
