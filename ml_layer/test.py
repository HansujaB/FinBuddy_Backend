#!/usr/bin/env python3
"""
BERT Classification Test Script
This script tests the BERT model's ability to classify expense text into categories.
Run this to verify your model is working before integrating with Dialogflow.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BERTExpenseClassifier:
    def __init__(self):
        # Use the same model as your main app
        self.model_name = "google/bert_uncased_L-2_H-128_A-2"
        self.tokenizer = None
        self.model = None
        self.categories = None
        self.category_to_id = None
        self.id_to_category = None
        self.num_labels = None
        
    def setup_categories(self):
        """Setup category mappings - same as your main app"""
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
        self.categories = df['category'].unique()
        self.category_to_id = {category: i for i, category in enumerate(self.categories)}
        self.id_to_category = {i: category for i, category in enumerate(self.categories)}
        self.num_labels = len(self.categories)
        
        print(f"📊 Categories: {list(self.categories)}")
        print(f"📊 Number of labels: {self.num_labels}")
        
    def load_model(self):
        """Load the BERT model and tokenizer"""
        try:
            print("🔄 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("🔄 Loading BERT model...")
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels
            )
            
            print("✅ Model and tokenizer loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def simple_rule_based_categorizer(self, text):
        """Rule-based categorizer as fallback - same as your main app"""
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
            return 'Food'  # Default
    
    def predict_single(self, text):
        """Predict category for a single text"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded, using rule-based classifier")
            return self.simple_rule_based_categorizer(text), 0.0
        
        try:
            # Tokenize the input
            encoded_input = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=64
            )
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            logits = outputs.logits
            prediction_index = torch.argmax(logits, dim=-1).item()
            predicted_category = self.id_to_category[prediction_index]
            
            # Get confidence score
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities[0][prediction_index].item()
            
            return predicted_category, confidence
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return self.simple_rule_based_categorizer(text), 0.0
    
    def run_comprehensive_test(self):
        """Run comprehensive tests with various expense texts"""
        
        # Test cases with expected categories
        test_cases = [
            # Food expenses
            ("I spent 250 on Zomato order", "Food"),
            ("Paid 80 for coffee at Starbucks", "Food"),
            ("200 for lunch at restaurant", "Food"),
            ("Had dinner for 400 rupees", "Food"),
            
            # Transport expenses
            ("Took Uber for 150 rupees", "Transport"),
            ("Metro ticket cost 45", "Transport"),
            ("Bus fare was 25", "Transport"),
            ("Filled petrol for 2000", "Transport"),
            
            # Shopping expenses
            ("Bought clothes from Amazon for 1500", "Shopping"),
            ("Purchased shoes for 800", "Shopping"),
            ("Online shopping on Flipkart 600", "Shopping"),
            ("New shirt cost 500", "Shopping"),
            
            # Bills
            ("Electricity bill 1200", "Bills"),
            ("Netflix subscription 200", "Bills"),
            ("Mobile bill payment 400", "Bills"),
            ("Internet bill 800", "Bills"),
            
            # Travel
            ("Flight ticket 8000", "Travel"),
            ("Hotel booking 3000", "Travel"),
            ("Train ticket 500", "Travel"),
            
            # Others
            ("Rent payment 15000", "Rent"),
            ("Bought a book for 300", "Books"),
            ("Grocery shopping 1200", "Groceries"),
            
            # Ambiguous cases
            ("Spent 100 today", "Food"),  # Should default to Food
            ("Payment of 500", "Food"),   # Ambiguous
        ]
        
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE BERT CLASSIFICATION TEST")
        print("="*80)
        
        results = []
        bert_predictions = []
        rule_predictions = []
        expected_labels = []
        
        for i, (text, expected) in enumerate(test_cases, 1):
            # BERT prediction
            bert_pred, confidence = self.predict_single(text)
            
            # Rule-based prediction for comparison
            rule_pred = self.simple_rule_based_categorizer(text)
            
            # Store results
            results.append({
                'text': text,
                'expected': expected,
                'bert_prediction': bert_pred,
                'rule_prediction': rule_pred,
                'confidence': confidence,
                'bert_correct': bert_pred == expected,
                'rule_correct': rule_pred == expected
            })
            
            bert_predictions.append(bert_pred)
            rule_predictions.append(rule_pred)
            expected_labels.append(expected)
            
            # Print result
            bert_status = "✅" if bert_pred == expected else "❌"
            rule_status = "✅" if rule_pred == expected else "❌"
            
            print(f"{i:2d}. Text: '{text[:50]}...' if len(text) > 50 else text")
            print(f"    Expected: {expected}")
            print(f"    BERT:     {bert_pred} {bert_status} (confidence: {confidence:.3f})")
            print(f"    Rule:     {rule_pred} {rule_status}")
            print()
        
        return results, bert_predictions, rule_predictions, expected_labels
    
    def calculate_metrics(self, predictions, expected, method_name):
        """Calculate and display accuracy metrics"""
        accuracy = accuracy_score(expected, predictions)
        
        print(f"\n📊 {method_name} METRICS:")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Classification report
        try:
            report = classification_report(expected, predictions, 
                                         labels=list(self.categories),
                                         target_names=list(self.categories),
                                         zero_division=0)
            print(f"   Classification Report:")
            print(report)
        except Exception as e:
            print(f"   Could not generate classification report: {e}")
        
        return accuracy
    
    def plot_confusion_matrices(self, bert_preds, rule_preds, expected):
        """Plot confusion matrices for both methods"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # BERT Confusion Matrix
            cm_bert = confusion_matrix(expected, bert_preds, labels=list(self.categories))
            sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.categories, yticklabels=self.categories, ax=ax1)
            ax1.set_title('BERT Model Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Rule-based Confusion Matrix
            cm_rule = confusion_matrix(expected, rule_preds, labels=list(self.categories))
            sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Greens',
                       xticklabels=self.categories, yticklabels=self.categories, ax=ax2)
            ax2.set_title('Rule-based Classifier Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Confusion matrices saved as 'classification_comparison.png'")
            
        except Exception as e:
            print(f"❌ Could not create plots: {e}")
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*50)
        print("🎯 INTERACTIVE TESTING MODE")
        print("Enter expense texts to test classification")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            text = input("\nEnter expense text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            bert_pred, confidence = self.predict_single(text)
            rule_pred = self.simple_rule_based_categorizer(text)
            
            print(f"  BERT Prediction: {bert_pred} (confidence: {confidence:.3f})")
            print(f"  Rule Prediction: {rule_pred}")
            
            if bert_pred != rule_pred:
                print("  ⚠️  Different predictions!")

def main():
    """Main function to run all tests"""
    print("🚀 Starting BERT Classification Tests...")
    
    # Initialize classifier
    classifier = BERTExpenseClassifier()
    
    # Setup categories
    classifier.setup_categories()
    
    # Load model
    model_loaded = classifier.load_model()
    
    if not model_loaded:
        print("⚠️  Model failed to load, will use rule-based classifier only")
    
    # Run comprehensive test
    results, bert_preds, rule_preds, expected = classifier.run_comprehensive_test()
    
    # Calculate metrics
    bert_accuracy = classifier.calculate_metrics(bert_preds, expected, "BERT MODEL")
    rule_accuracy = classifier.calculate_metrics(rule_preds, expected, "RULE-BASED")
    
    # Plot confusion matrices
    classifier.plot_confusion_matrices(bert_preds, rule_preds, expected)
    
    # Summary
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    print(f"BERT Model Accuracy:     {bert_accuracy:.3f} ({bert_accuracy*100:.1f}%)")
    print(f"Rule-based Accuracy:     {rule_accuracy:.3f} ({rule_accuracy*100:.1f}%)")
    print(f"Better Method:           {'BERT' if bert_accuracy > rule_accuracy else 'Rule-based'}")
    
    if model_loaded:
        if bert_accuracy < 0.5:
            print("\n⚠️  WARNING: BERT model accuracy is low!")
            print("   This is expected for an untrained model.")
            print("   Consider fine-tuning the model on your expense data.")
        elif bert_accuracy > 0.8:
            print("\n✅ BERT model is performing well!")
        else:
            print("\n📊 BERT model shows moderate performance.")
    
    # Interactive testing
    try:
        classifier.interactive_test()
    except KeyboardInterrupt:
        print("\n\n👋 Testing completed!")

if __name__ == "__main__":
    main()