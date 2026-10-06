from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load the pre-trained models
model = joblib.load("engagement_model_distil_multioutput.pkl")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess input text with BERT
def extract_bert_features(text, tokenizer, bert_model, device='cpu', max_len=128):
    bert_model.eval()
    inputs = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
    return cls_embedding.cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')


# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        post_content = data.get("Post Content", "")
        platform = data.get("Platform", "")

        # Check for valid inputs
        if not post_content or not platform:
            return jsonify({"error": "Invalid input. Both 'Post Content' and 'Platform' are required."}), 400

        # Preprocess text with BERT
        bert_features = extract_bert_features(post_content, tokenizer, bert_model)

        # One-hot encode platform
        platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'YouTube']  # Exact list from training
        platform_features = [1 if platform == p else 0 for p in platforms]

        # Combine features
        platform_features = np.array(platform_features).reshape(1, -1)  # Reshape to 2D
        features = np.hstack([bert_features, platform_features])

        # Ensure feature count matches model expectation
        expected_features = model.n_features_in_
        current_features = features.shape[1]

        if current_features < expected_features:
            # Pad missing columns with zeros
            padding = np.zeros((1, expected_features - current_features))
            features = np.hstack([features, padding])
        elif current_features > expected_features:
            return jsonify({"error": f"Feature mismatch: too many features. Expected {expected_features}, got {current_features}"}), 400

        # Predict engagement
        prediction = model.predict(features)

        # Return predictions as JSON
        return jsonify({
            "Likes": int(prediction[0][0]),
            "Comments": int(prediction[0][1]),
            "Impressions": int(prediction[0][2]),
            "Reach": int(prediction[0][3]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
