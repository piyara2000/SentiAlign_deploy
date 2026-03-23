SentiAlign
Multi-Model Sentiment Conflict Detection for Software Bug Reports

SentiAlign is a sentiment analysis system designed to detect and resolve sentiment conflicts in software bug reports by leveraging multiple AI models and explainable AI techniques.

The system combines predictions from several state-of-the-art sentiment analysis models and applies a meta-classifier to align conflicting outputs, improving reliability and interpretability of sentiment analysis in software engineering contexts.

Features

• Multi-Model Sentiment Analysis
Integrates multiple models such as BERT, RoBERTa, and Senti4SD to analyze developer sentiment.

• Conflict Detection
Identifies instances where models produce conflicting sentiment predictions.

• Meta-Classifier Resolution
A meta-classifier resolves sentiment conflicts and produces a final aligned sentiment prediction.

• Explainable AI (XAI)
Provides explanations for predictions using interpretable AI techniques.

• Interactive Web Interface
Allows users to input text and view sentiment predictions from multiple models.

• Improved Sentiment Agreement
Enhances agreement between models and improves classification reliability.

System Architecture

The SentiAlign pipeline follows these stages:

Text Input
User submits bug report text through the web interface.
Preprocessing
Text is cleaned and prepared for model analysis.
Multi-Model Sentiment Analysis
The input text is analyzed using multiple models.
Conflict Detection
Predictions from different models are compared to identify disagreements.
Meta-Classifier Resolution
A trained classifier resolves the conflict and produces the final sentiment.
Explainability Layer
Provides explanations for the final decision.

Installation
1. Clone the repository
git clone https://github.com/yourusername/SentiAlign.git
cd SentiAlign
2. Install dependencies
pip install -r requirements.txt
3. Build frontend
npm run build
4. Run the backend server
python app.py
5. Open the application
Navigate to:
http://localhost:5000

Example Usage
Open the SentiAlign web interface.
Enter a bug report or developer comment.
Click Analyze.
View predictions from multiple models and the final aligned sentiment.

Example input:

The new update fixed the issue, but the process took far too long and was frustrating.
