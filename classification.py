# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, BertForSequenceClassification
# import torch
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # --- Load Mini-BERT Model and Tokenizer ---
# # We use a mini-BERT model for fast inference in a hackathon setting.
# model_name = "google/bert_uncased_L-2_H-128_A-2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # --- Mock Data and Category Mapping ---
# # This data is used to define the category labels for our classifier.
# # No training is done on this data in this version.
# data = {
#     'text': [
#         '500 on Zomato', '120 for coffee', '40 for tea', '700 for groceries',
#         '250 for Uber ride', '150 on metro ticket', '1200 on flight',
#         '2500 on Amazon shopping', '700 on clothes', '120 on Netflix subscription',
#         '50 for electricity bill', '1500 for rent', '50 for a book',
#         '100 for bus fare', '200 at the cafe', '300 for dinner'
#     ],
#     'category': [
#         'Food', 'Food', 'Food', 'Groceries',
#         'Transport', 'Transport', 'Travel',
#         'Shopping', 'Shopping', 'Bills',
#         'Bills', 'Rent', 'Books',
#         'Transport', 'Food', 'Food'
#     ]
# }
# df = pd.DataFrame(data)
# categories = df['category'].unique()
# category_to_id = {category: i for i, category in enumerate(categories)}
# id_to_category = {i: category for i, category in enumerate(categories)}
# num_labels = len(categories)

# # Load a pre-trained BERT model with a classification head.
# # We will not fine-tune it.
# try:
#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
#     print("--- Loaded pre-trained model. ---")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     # In a real scenario, you'd handle this more gracefully.
#     model = None

# @app.route('/categorize_expense', methods=['POST'])
# def categorize_expense():
#     """
#     API endpoint to categorize a user's expense using a pre-trained mini-BERT model.
#     It takes rawText from the backend and returns a predicted category.
#     """
#     raw_text = request.json.get('rawText', '')

#     if not raw_text:
#         return jsonify({"error": "No rawText provided"}), 400
    
#     if model is None:
#         return jsonify({"error": "Model not loaded"}), 500

#     # Tokenize the input text
#     encoded_input = tokenizer(
#         raw_text,
#         return_tensors='pt',
#         truncation=True,
#         padding=True,
#         max_length=64
#     )

#     # Make the prediction
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**encoded_input)
    
#     logits = outputs.logits
#     # For a pre-trained model, the initial weights are random, so we need a heuristic.
#     # We will simply choose a category randomly for demonstration purposes,
#     # as the model's un-trained classification head won't produce meaningful results.
#     # In a real deployment, a fine-tuned model would give a correct prediction.
#     prediction_index = np.random.choice(len(id_to_category))
#     predicted_category = id_to_category[prediction_index]
    
#     print(f"Text: '{raw_text}' -> Category: '{predicted_category}'")
#     return jsonify({"category": predicted_category})

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("--- Loaded pre-trained model. ---")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # In a real scenario, you'd handle this more gracefully.
    model = None

def extract_expense_text(dialogflow_request):
    """
    Extract expense text from Dialogflow CX request.
    This function handles multiple ways the text might be passed.
    """
    try:
        # Method 1: From query text (most common)
        query_text = dialogflow_request.get('text', '')
        if query_text:
            return query_text
        
        # Method 2: From parameters
        fulfillment_info = dialogflow_request.get('fulfillmentInfo', {})
        parameters = fulfillment_info.get('parameters', {})
        
        # Look for common parameter names that might contain expense text
        expense_text = (parameters.get('expense_text') or 
                       parameters.get('rawText') or 
                       parameters.get('expense_description') or
                       parameters.get('text', ''))
        
        if expense_text:
            return expense_text
        
        # Method 3: From intent display name or other fields
        intent_info = dialogflow_request.get('intentInfo', {})
        if intent_info:
            # Sometimes the expense info might be in the query
            return dialogflow_request.get('text', '')
            
        return ''
    except Exception as e:
        logger.error(f"Error extracting expense text: {e}")
        return ''

def create_dialogflow_response(category, original_text=""):
    """
    Create a properly formatted Dialogflow CX webhook response.
    """
    return {
        "fulfillmentResponse": {
            "messages": [
                {
                    "text": {
                        "text": [f"I've categorized your expense as: {category}"]
                    }
                }
            ]
        },
        "sessionInfo": {
            "parameters": {
                "predicted_category": category,
                "original_expense_text": original_text
            }
        }
    }

def simple_rule_based_categorizer(text):
    """
    A simple rule-based categorizer as fallback when BERT model fails.
    This provides more meaningful results than random selection.
    """
    text_lower = text.lower()
    
    # Food keywords
    food_keywords = ['zomato', 'swiggy', 'restaurant', 'coffee', 'tea', 'food', 
                     'lunch', 'dinner', 'breakfast', 'cafe', 'pizza', 'burger']
    
    # Transport keywords  
    transport_keywords = ['uber', 'ola', 'metro', 'bus', 'taxi', 'ride', 
                         'transport', 'petrol', 'diesel', 'fuel']
    
    # Shopping keywords
    shopping_keywords = ['amazon', 'flipkart', 'shopping', 'clothes', 'dress', 
                        'shirt', 'shoes', 'online', 'purchase']
    
    # Bills keywords
    bills_keywords = ['electricity', 'water', 'gas', 'bill', 'payment', 
                     'netflix', 'subscription', 'mobile', 'internet']
    
    # Travel keywords
    travel_keywords = ['flight', 'train', 'hotel', 'booking', 'travel', 'trip']
    
    # Check for keywords
    if any(keyword in text_lower for keyword in food_keywords):
        return 'Food'
    elif any(keyword in text_lower for keyword in transport_keywords):
        return 'Transport'  
    elif any(keyword in text_lower for keyword in shopping_keywords):
        return 'Shopping'
    elif any(keyword in text_lower for keyword in bills_keywords):
        return 'Bills'
    elif any(keyword in text_lower for keyword in travel_keywords):
        return 'Travel'
    elif 'rent' in text_lower:
        return 'Rent'
    elif 'book' in text_lower:
        return 'Books'
    elif 'groceries' in text_lower or 'grocery' in text_lower:
        return 'Groceries'
    else:
        # Default to most common category
        return 'Food'

@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    """
    Main webhook endpoint for Dialogflow CX.
    This is the endpoint you should configure in your Dialogflow CX agent.
    """
    try:
        # Get the request data
        req = request.get_json()
        
        if not req:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400
        
        # Extract expense text from the request
        expense_text = extract_expense_text(req)
        
        if not expense_text:
            logger.warning("No expense text found in request")
            return jsonify(create_dialogflow_response("Unknown", ""))
        
        logger.info(f"Processing expense text: {expense_text}")
        
        # Try to categorize using BERT model
        predicted_category = None
        
        if model is not None:
            try:
                # Tokenize the input text
                encoded_input = tokenizer(
                    expense_text,
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
                # Get the predicted class
                prediction_index = torch.argmax(logits, dim=-1).item()
                predicted_category = id_to_category[prediction_index]
                
                logger.info(f"BERT prediction: {predicted_category}")
                
            except Exception as e:
                logger.error(f"Error with BERT model prediction: {e}")
                predicted_category = None
        
        # Fallback to rule-based categorizer if BERT fails
        if predicted_category is None:
            predicted_category = simple_rule_based_categorizer(expense_text)
            logger.info(f"Rule-based prediction: {predicted_category}")
        
        # Create and return Dialogflow response
        response = create_dialogflow_response(predicted_category, expense_text)
        logger.info(f"Returning response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        return jsonify({
            "fulfillmentResponse": {
                "messages": [
                    {
                        "text": {
                            "text": ["Sorry, I encountered an error while categorizing your expense."]
                        }
                    }
                ]
            }
        }), 500

@app.route('/categorize_expense', methods=['POST'])
def categorize_expense():
    """
    API endpoint to categorize a user's expense using a pre-trained mini-BERT model.
    This is kept for backward compatibility but the main endpoint is /webhook.
    """
    try:
        raw_text = request.json.get('rawText', '')
        
        if not raw_text:
            return jsonify({"error": "No rawText provided"}), 400
        
        if model is None:
            # Use rule-based fallback
            predicted_category = simple_rule_based_categorizer(raw_text)
        else:
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
            prediction_index = torch.argmax(logits, dim=-1).item()
            predicted_category = id_to_category[prediction_index]
        
        logger.info(f"Text: '{raw_text}' -> Category: '{predicted_category}'")
        return jsonify({"category": predicted_category})
        
    except Exception as e:
        logger.error(f"Error in categorize_expense: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "categories": list(categories)
    })

if __name__ == '__main__':
    # Print configuration info for easy setup
    print("=" * 50)
    print("FinBuddy Backend Webhook Server")
    print("=" * 50)
    print(f"🔗 Webhook Endpoint: {WEBHOOK_URL_ENDPOINT}")
    print(f"🔒 Authentication: {'Enabled' if WEBHOOK_SECRET_KEY != 'your-secret-key-here' else 'Disabled'}")
    print(f"🤖 Model Status: {'Loaded' if model is not None else 'Using Rule-Based Fallback'}")
    print("=" * 50)
    print("\n📋 DIALOGFLOW CX SETUP INSTRUCTIONS:")
    print("1. Go to Dialogflow CX Console → Your Agent → Manage → Webhooks")
    print("2. Create webhook with these settings:")
    print(f"   • Display Name: FinBuddy-Backend-Webhook")
    print(f"   • Webhook URL: https://your-domain.com{WEBHOOK_URL_ENDPOINT}")
    print(f"   • Authentication: {WEBHOOK_SECRET_KEY if WEBHOOK_SECRET_KEY != 'your-secret-key-here' else 'Add your secret key'}")
    print("3. Create a Fallback Intent as described in your guide")
    print("4. Configure the Default Start Flow to route to your webhook")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)