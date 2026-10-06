# Social Media Engagement Prediction System

A Python-based machine learning application that predicts social media engagement metrics from post content and platform information.

## Project Overview

Social media engagement depends on several factors, including the content of a post and the platform on which it is published.

This project uses pre-trained BERT and DistilBERT language models to extract meaningful contextual features from social media post content. These text representations are combined with platform-related features and passed to machine learning models to predict engagement.

The project includes both classification and regression approaches:

- Classification to determine whether a post has high or low engagement.
- Multi-output regression to predict multiple engagement metrics simultaneously.

The trained models are integrated into a Flask application that provides an API endpoint for making predictions.

## Key Features

- Predicts social media Likes, Comments, Impressions, and Reach.
- Classifies posts into high-engagement and low-engagement categories.
- Uses BERT and DistilBERT for contextual text feature extraction.
- Combines text embeddings with platform information.
- Uses Random Forest for engagement classification.
- Uses MultiOutputRegressor with Random Forest Regression for multiple engagement predictions.
- Provides a Flask-based web application and prediction API.
- Accepts prediction requests in JSON format.
- Returns prediction results in JSON format.

## Machine Learning Workflow

The project follows the following workflow:

1. Load and preprocess the social media engagement dataset.
2. Handle missing values in the dataset.
3. Process the post content using BERT or DistilBERT tokenization.
4. Generate contextual text embeddings using the pre-trained language models.
5. Encode the social media platform as numerical features.
6. Combine text embeddings with additional platform and engagement-related features.
7. Split the data into training and testing sets.
8. Train Random Forest classification and regression models.
9. Save the trained models using Joblib.
10. Load the trained models through the Flask application.
11. Accept post content and platform information through the prediction API.
12. Return the predicted engagement metrics as a JSON response.

## Models Used

### BERT

BERT is used to generate contextual representations of the social media post content. The `[CLS]` token representation is extracted and used as the text feature vector.

### DistilBERT

DistilBERT is also used as a lightweight alternative for extracting contextual text features from post content.

### Random Forest Classifier

A Random Forest classifier is trained to classify posts into high-engagement and low-engagement categories.

### Multi-Output Random Forest Regression

A multi-output regression approach is used to predict multiple engagement metrics:

- Likes
- Comments
- Impressions
- Reach

## REST API

The Flask application provides a prediction endpoint:

`POST /predict`

The endpoint accepts JSON input containing:

```json
{
    "Post Content": "Your social media post content",
    "Platform": "Instagram"
}

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- BERT
- DistilBERT

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt