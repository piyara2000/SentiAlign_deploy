from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory, abort
import joblib
import torch
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from nltk.corpus import opinion_lexicon
import nltk
import shap
import json as json_module

app = Flask(__name__)
app.secret_key = 'sentialign-secret-key'

FRONTEND_DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/opinion_lexicon')
except LookupError:
    nltk.download('opinion_lexicon', quiet=True)

# Initialize models globally
print("Loading models... This may take a while...")

# Load the resolver model
RESOLVER_MODEL = "sentiment_conflict_resolver.pkl"
try:
    resolver = joblib.load(RESOLVER_MODEL)
    print(f"[OK] Loaded resolver model from {RESOLVER_MODEL}")
except FileNotFoundError:
    print(f"ERROR: Could not find {RESOLVER_MODEL}. Please run meta-classifier-with-xai.py first.")
    resolver = None

# Load BERT model
BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
print(f"Loading BERT model: {BERT_MODEL_NAME}...")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)
bert_model.eval()
print("[OK] BERT model loaded")

# Load RoBERTa model
ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
print(f"Loading RoBERTa model: {ROBERTA_MODEL_NAME}...")
roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
roberta_model.eval()
print("[OK] RoBERTa model loaded")

# Load Senti4SD lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
print("[OK] Senti4SD lexicon loaded")

# Feature names expected by the resolver
FEATURES = [
    "bert_prob_1", "bert_prob_2", "bert_prob_3", "bert_prob_4", "bert_prob_5",
    "roberta_prob_neg", "roberta_prob_neu", "roberta_prob_pos",
    "senti4sd_prob_neg", "senti4sd_prob_neu", "senti4sd_prob_pos"
]

FEATURE_MEANING = {
    "bert_prob_1": "BERT strong negative signal",
    "bert_prob_2": "BERT moderate negative signal",
    "bert_prob_3": "BERT neutral signal",
    "bert_prob_4": "BERT moderate positive signal",
    "bert_prob_5": "BERT strong positive signal",
    "roberta_prob_neg": "RoBERTa negative signal",
    "roberta_prob_neu": "RoBERTa neutral signal",
    "roberta_prob_pos": "RoBERTa positive signal",
    "senti4sd_prob_neg": "Senti4SD negative signal",
    "senti4sd_prob_neu": "Senti4SD neutral signal",
    "senti4sd_prob_pos": "Senti4SD positive signal",
}

# Initialize SHAP explainer
shap_explainer = None


def validate_input_text(text: str):
    """
    Basic sanity checks for user input.

    - Must not be empty
    - Must contain letters (not only numbers / symbols)
    - Must not be overwhelmingly numeric
    - Length limits to avoid extremely short/long nonsense inputs
    """
    if text is None:
        return False, "Please provide text to analyze."

    stripped = text.strip()
    if not stripped:
        return False, "Please provide text to analyze."

    if len(stripped) < 5:
        return False, "Text is too short. Please enter a longer sentence or phrase."

    if len(stripped) > 4000:
        return False, "Text is too long. Please shorten it to under 4000 characters."

    has_alpha = any(ch.isalpha() for ch in stripped)
    if not has_alpha:
        return False, "The input does not contain letters. Please enter natural language text instead of only numbers or symbols."

    non_space_chars = [ch for ch in stripped if not ch.isspace()]
    digit_count = sum(ch.isdigit() for ch in non_space_chars)
    if non_space_chars and digit_count / len(non_space_chars) > 0.7:
        return False, "The input looks mostly numeric. Please enter a sentence or review comment instead of only numbers."

    # Require a minimum number of alphabetic characters overall
    # random noise are rejected
    alpha_count = sum(ch.isalpha() for ch in stripped)
    if alpha_count < 3:
        return False, "The input does not contain enough meaningful words. Please provide a clearer sentence or comment."

    return True, None


def get_bert_probs(text):
    """Get BERT probabilities (5 classes: 1-5 stars)"""
    if not isinstance(text, str) or text.strip() == "":
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy().flatten().tolist()
    return probs


def get_roberta_probs(text):
    """Get RoBERTa probabilities (3 classes: neg, neu, pos)"""
    if not isinstance(text, str) or text.strip() == "":
        return [1/3, 1/3, 1/3]
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy().flatten().tolist()
    return probs


def get_senti4sd_probs(text):
    """Get Senti4SD probabilities using lexicon (3 classes: neg, neu, pos)"""
    if not isinstance(text, str) or text.strip() == "":
        return [1/3, 1/3, 1/3]
    tokens = text.lower().split()
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    score = pos_count - neg_count
    
    if score > 0:
        probs = [0.1, 0.1, 0.8]  # mostly positive
    elif score < 0:
        probs = [0.8, 0.1, 0.1]  # mostly negative
    else:
        probs = [0.2, 0.6, 0.2]  # neutral
    return probs


def bert_label_from_probs(probs):
    """Map BERT 1-5 stars to -1, 0, +1"""
    label = np.argmax(probs) + 1
    if label <= 2:
        return -1
    elif label == 3:
        return 0
    else:
        return 1


def get_base_predictions(text):
    """Get predictions from all three base models"""
    bert_probs = get_bert_probs(text)
    roberta_probs = get_roberta_probs(text)
    senti4sd_probs = get_senti4sd_probs(text)
    
    bert_label = bert_label_from_probs(bert_probs)
    roberta_label = [-1, 0, 1][np.argmax(roberta_probs)]
    senti4sd_label = [-1, 0, 1][np.argmax(senti4sd_probs)]
    
    return {
        'bert': {'label': bert_label, 'probs': bert_probs},
        'roberta': {'label': roberta_label, 'probs': roberta_probs},
        'senti4sd': {'label': senti4sd_label, 'probs': senti4sd_probs}
    }

def prepare_features(base_preds):
    """Prepare feature vector for resolver model"""
    bert_probs = base_preds['bert']['probs']
    roberta_probs = base_preds['roberta']['probs']  # [neg, neu, pos]
    senti4sd_probs = base_preds['senti4sd']['probs']  # [neg, neu, pos]
    
    features = {
        "bert_prob_1": bert_probs[0],
        "bert_prob_2": bert_probs[1],
        "bert_prob_3": bert_probs[2],
        "bert_prob_4": bert_probs[3],
        "bert_prob_5": bert_probs[4],
        "roberta_prob_neg": roberta_probs[0],
        "roberta_prob_neu": roberta_probs[1],
        "roberta_prob_pos": roberta_probs[2],
        "senti4sd_prob_neg": senti4sd_probs[0],
        "senti4sd_prob_neu": senti4sd_probs[1],
        "senti4sd_prob_pos": senti4sd_probs[2],
    }
    return pd.DataFrame([features])[FEATURES]


def shap_to_text_explanation(shap_df, predicted_label, pred_probs, top_k=6):
    """Generate textual explanation from SHAP values"""
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    target_label = label_map[predicted_label]
    
    strong_pos = []
    weak_pos = []
    opposing = []
    
    for _, row in shap_df.head(top_k).iterrows():
        feature = row["feature"]
        shap_val = row["shap_value"]
        meaning = FEATURE_MEANING.get(feature, feature)
        strength = abs(shap_val)
        
        if shap_val > 0:
            if strength > 0.1:
                strong_pos.append((meaning, shap_val))
            else:
                weak_pos.append((meaning, shap_val))
        else:
            opposing.append((meaning, shap_val))
    
    confidence = max(pred_probs)
    confidence_level = (
        "high" if confidence > 0.7 else
        "moderate" if confidence > 0.5 else
        "low"
    )
    
    explanation = (
        f"The system resolved the sentiment as **{target_label}** "
        f"with {confidence_level} confidence ({confidence:.2f}). "
    )
    
    if strong_pos:
        explanation += (
            "This decision was strongly driven by "
            + ", ".join([m for m, _ in strong_pos if "neutral" not in m.lower()][:2])
            + ". "
        )
    
    neutral_support = [m for m, _ in strong_pos if "neutral" in m.lower()]
    if neutral_support:
        explanation += (
            "Neutral signals such as "
            + ", ".join(neutral_support[:1])
            + " acted as moderating evidence rather than opposing the outcome. "
        )
    
    if weak_pos:
        explanation += (
            "Additional support came from "
            + ", ".join([m for m, _ in weak_pos[:2]])
            + ". "
        )
    
    if opposing:
        explanation += (
            "However, signals such as "
            + ", ".join([m for m, _ in opposing[:2]])
            + " opposed this outcome but had weaker influence."
        )
    
    return explanation


@app.route('/')
def index():
    """Render the main page (React build if present, else Jinja template)."""
    if os.path.isdir(FRONTEND_DIST_DIR):
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")
    return render_template('index.html')


@app.route('/results')
def results():
    """Display analysis results on a separate page"""
    # If React build is present, let the SPA handle /results.
    if os.path.isdir(FRONTEND_DIST_DIR):
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    if 'analysis_results' not in session:
        return redirect(url_for('index'))
    
    try:
        results_data = json_module.loads(session.get('analysis_results'))
        input_text = session.get('input_text', '')
        
        # Clear session after displaying
        session.pop('analysis_results', None)
        session.pop('input_text', None)
        
        return render_template('results.html', results=results_data, input_text=input_text)
    except (json_module.JSONDecodeError, KeyError) as e:
        print(f"Error loading results from session: {e}")
        return redirect(url_for('index'))


def _analyze_text_to_response(text: str, include_shap: bool = True):
    """Core analyzer logic (shared by /analyze and /api/analyze).

    include_shap controls whether expensive SHAP feature attributions are computed.
    """
    text = (text or "").strip()

    is_valid, validation_msg = validate_input_text(text)
    if not is_valid:
        return None, (validation_msg, 400)

    # Get base model predictions
    base_preds = get_base_predictions(text)

    # Prepare features for resolver
    features_df = prepare_features(base_preds)

    if resolver is None:
        return None, ("Resolver model not loaded", 500)

    # Get resolver prediction
    resolved_label = resolver.predict(features_df)[0]
    resolved_probs = resolver.predict_proba(features_df)[0]

    prob_dict = {
        -1: resolved_probs[list(resolver.classes_).index(-1)],
        0: resolved_probs[list(resolver.classes_).index(0)],
        1: resolved_probs[list(resolver.classes_).index(1)],
    }

    # Generate explanation (optionally with SHAP)
    explanation = ""
    shap_contrib_list = []

    def _fallback_explanation() -> str:
        confidence = max(prob_dict.values())
        confidence_level = (
            "high"
            if confidence > 0.7
            else "moderate"
            if confidence > 0.5
            else "low"
        )
        label_map_local = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        return (
            f"The system resolved the sentiment as **{label_map_local[resolved_label]}** "
            f"with {confidence_level} confidence ({confidence:.2f}). "
            f"Base models: BERT={label_map_local[base_preds['bert']['label']]}, "
            f"RoBERTa={label_map_local[base_preds['roberta']['label']]}, "
            f"Senti4SD={label_map_local[base_preds['senti4sd']['label']]}."
        )

    if not include_shap:
        # Skip SHAP entirely if user didn't request it
        explanation = _fallback_explanation()
    else:
        try:
            global shap_explainer
            if shap_explainer is None:
                try:
                    background_df = pd.DataFrame(
                        [features_df.mean().values], columns=FEATURES
                    )
                    shap_explainer = shap.Explainer(resolver.predict_proba, background_df)
                except Exception as e:
                    print(f"Warning: Could not initialize SHAP explainer: {e}")
                    shap_explainer = None

            if shap_explainer is not None:
                shap_values = shap_explainer(features_df)
                shap_vals_array = shap_values.values

                pred_class_idx = list(resolver.classes_).index(resolved_label)

                if len(shap_vals_array.shape) == 2:
                    shap_vals = shap_vals_array[0]
                elif len(shap_vals_array.shape) == 3:
                    if shap_vals_array.shape[1] == len(resolver.classes_):
                        shap_vals = shap_vals_array[0][pred_class_idx, :]
                    else:
                        shap_vals = shap_vals_array[0][:, pred_class_idx]
                else:
                    shap_vals = shap_vals_array.flatten()[: len(FEATURES)]

                shap_contrib = (
                    pd.DataFrame({"feature": FEATURES, "shap_value": shap_vals})
                    .sort_values(by="shap_value", key=lambda x: x.abs(), ascending=False)
                )

                explanation = shap_to_text_explanation(
                    shap_contrib,
                    predicted_label=resolved_label,
                    pred_probs=[prob_dict[-1], prob_dict[0], prob_dict[1]],
                    top_k=6,
                )

                shap_contrib_list = [
                    {
                        "feature": row["feature"],
                        "shap_value": float(row["shap_value"]),
                        "meaning": FEATURE_MEANING.get(row["feature"], row["feature"]),
                    }
                    for _, row in shap_contrib.head(10).iterrows()
                ]
            else:
                explanation = _fallback_explanation()
        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            explanation = _fallback_explanation()

    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    response = {
        "success": True,
        "base_predictions": {
            "bert": {
                "label": label_map[base_preds["bert"]["label"]],
                "probabilities": {
                    "class_1": float(base_preds["bert"]["probs"][0]),
                    "class_2": float(base_preds["bert"]["probs"][1]),
                    "class_3": float(base_preds["bert"]["probs"][2]),
                    "class_4": float(base_preds["bert"]["probs"][3]),
                    "class_5": float(base_preds["bert"]["probs"][4]),
                },
            },
            "roberta": {
                "label": label_map[base_preds["roberta"]["label"]],
                "probabilities": {
                    "negative": float(base_preds["roberta"]["probs"][0]),
                    "neutral": float(base_preds["roberta"]["probs"][1]),
                    "positive": float(base_preds["roberta"]["probs"][2]),
                },
            },
            "senti4sd": {
                "label": label_map[base_preds["senti4sd"]["label"]],
                "probabilities": {
                    "negative": float(base_preds["senti4sd"]["probs"][0]),
                    "neutral": float(base_preds["senti4sd"]["probs"][1]),
                    "positive": float(base_preds["senti4sd"]["probs"][2]),
                },
            },
        },
        "resolved": {
            "label": label_map[resolved_label],
            "probabilities": {
                "negative": float(prob_dict[-1]),
                "neutral": float(prob_dict[0]),
                "positive": float(prob_dict[1]),
            },
            "confidence": float(max(prob_dict.values())),
        },
        "explanation": explanation,
        "shap_contributions": shap_contrib_list,
    }

    return response, None


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """React API: analyze text and return results JSON (no session)."""
    try:
        data = request.get_json(silent=True) or {}
        include_shap = bool(data.get("include_shap", False))
        response, err = _analyze_text_to_response(data.get("text"), include_shap=include_shap)
        if err:
            msg, code = err
            return jsonify({"error": msg}), code
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Legacy endpoint: analyze and store results in session for the Jinja /results page."""
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        # Legacy endpoint keeps SHAP enabled by default
        response, err = _analyze_text_to_response(text, include_shap=True)
        if err:
            msg, code = err
            return jsonify({"error": msg}), code

        session['analysis_results'] = json_module.dumps(response)
        session['input_text'] = text
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing text: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    try:
        data = request.get_json(silent=True) or {}
        feedback_label = data.get('feedback_label')
        feedback_text = data.get('feedback_text', '')

        feedback_entry = {
            "input_text": data.get("input_text", ""),
            "resolved_sentiment": data.get("resolved_sentiment", ""),
            "confidence": data.get("confidence", ""),
            "feedback_label": feedback_label,
            "feedback_text": feedback_text
        }

        # Append feedback to a JSON file
        try:
            with open("user_feedback.json", "r") as f:
                feedback_data = json_module.load(f)
        except FileNotFoundError:
            feedback_data = []

        feedback_data.append(feedback_entry)

        with open("user_feedback.json", "w") as f:
            json_module.dump(feedback_data, f, indent=2)

        return jsonify({"success": True})

    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """Legacy endpoint for the Jinja results page (session-based)."""
    try:
        data = request.get_json(silent=True) or {}
        feedback_label = data.get('feedback_label')
        feedback_text = data.get('feedback_text', '')

        feedback_entry = {
            "input_text": session.get("input_text", ""),
            "resolved_sentiment": json_module.loads(
                session.get("analysis_results", "{}")
            ).get("resolved", {}).get("label", ""),
            "confidence": json_module.loads(
                session.get("analysis_results", "{}")
            ).get("resolved", {}).get("confidence", ""),
            "feedback_label": feedback_label,
            "feedback_text": feedback_text
        }

        try:
            with open("user_feedback.json", "r") as f:
                feedback_data = json_module.load(f)
        except FileNotFoundError:
            feedback_data = []

        feedback_data.append(feedback_entry)

        with open("user_feedback.json", "w") as f:
            json_module.dump(feedback_data, f, indent=2)

        return jsonify({"success": True})

    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False}), 500


@app.route('/<path:path>')
def serve_react_static_or_404(path):
    """
    Serve React build assets + SPA fallback when built.
    Keeps /static and Jinja templates working when the build does not exist.
    """
    if path.startswith("api/"):
        abort(404)

    if os.path.isdir(FRONTEND_DIST_DIR):
        full_path = os.path.join(FRONTEND_DIST_DIR, path)
        if os.path.isfile(full_path):
            return send_from_directory(FRONTEND_DIST_DIR, path)
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    abort(404)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("SentiAlign Flask App")
    print("="*50)
    print("\nStarting server...")
    print("Open your browser to http://127.0.0.1:5000")
    print("\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
