# app.py (Render-safe + memory-optimized)
# Key fixes:
# - Lazy-load heavy models (torch/transformers/shap) only when needed
# - Avoid pandas entirely (features are numpy array)
# - SHAP disabled by default; optional via include_shap=true
# - Add /api/warmup route to pre-download HF models and expose errors in logs
# - Better error logging (tracebacks) so Render shows real cause
# - Safer secret key via env var
#
# Recommended Render start command:
#   gunicorn app:app --workers 1 --threads 1 --timeout 180

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    session, send_from_directory, abort
)
import os
import json as json_module
import joblib
import numpy as np
import nltk
from nltk.corpus import opinion_lexicon

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "sentialign-secret-key")

FRONTEND_DIST_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")

# ----------------------------
# NLTK resources (best-effort)
# ----------------------------
# On Render, downloading at boot can fail if network is restricted or slow.
# We keep it best-effort; lexicon will fallback to empty sets.
try:
    nltk.data.find("corpora/opinion_lexicon")
except LookupError:
    try:
        nltk.download("opinion_lexicon", quiet=True)
    except Exception:
        pass

try:
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
except Exception:
    positive_words, negative_words = set(), set()

# ----------------------------
# Resolver model (joblib/pickle)
# ----------------------------
RESOLVER_MODEL_PATH = "sentiment_conflict_resolver.pkl"
try:
    resolver = joblib.load(RESOLVER_MODEL_PATH)
    print(f"[OK] Loaded resolver model from {RESOLVER_MODEL_PATH}")
except FileNotFoundError:
    resolver = None
    print(f"[ERROR] Could not find {RESOLVER_MODEL_PATH}. Upload/commit it to the repo root.")
except Exception as e:
    resolver = None
    print(f"[ERROR] Failed to load resolver pickle: {e}")

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

# ----------------------------
# Lazy-loaded heavy models
# ----------------------------
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
ROBERTA_MODEL_NAME = os.getenv("ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment")

bert_tokenizer = None
bert_model = None
roberta_tokenizer = None
roberta_model = None

# SHAP explainer is also lazy
shap_explainer = None


def _torch():
    """Local torch import to keep startup memory lower."""
    import torch
    return torch


def _hf_is_offline() -> bool:
    """Allow turning HF downloads off via env."""
    return os.getenv("HF_HUB_OFFLINE", "0").strip() in ("1", "true", "True", "yes", "YES")


def get_bert():
    global bert_tokenizer, bert_model
    if bert_model is None or bert_tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch = _torch()
        if _hf_is_offline():
            # If offline, you must pre-bundle model files or mount cache.
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
        else:
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME)

        bert_model.eval()
        bert_model.to(torch.device("cpu"))
        print("[OK] BERT loaded")
    return bert_tokenizer, bert_model


def get_roberta():
    global roberta_tokenizer, roberta_model
    if roberta_model is None or roberta_tokenizer is None:
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        torch = _torch()
        if _hf_is_offline():
            roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME, local_files_only=True)
            roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, local_files_only=True)
        else:
            roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
            roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)

        roberta_model.eval()
        roberta_model.to(torch.device("cpu"))
        print("[OK] RoBERTa loaded")
    return roberta_tokenizer, roberta_model


# ----------------------------
# Input validation
# ----------------------------
def validate_input_text(text: str):
    if text is None:
        return False, "Please provide text to analyze."

    stripped = text.strip()
    if not stripped:
        return False, "Please provide text to analyze."

    if len(stripped) < 5:
        return False, "Text is too short. Please enter a longer sentence or phrase."

    if len(stripped) > 4000:
        return False, "Text is too long. Please shorten it to under 4000 characters."

    if not any(ch.isalpha() for ch in stripped):
        return False, "The input does not contain letters. Please enter natural language text instead of only numbers or symbols."

    non_space_chars = [ch for ch in stripped if not ch.isspace()]
    digit_count = sum(ch.isdigit() for ch in non_space_chars)
    if non_space_chars and digit_count / len(non_space_chars) > 0.7:
        return False, "The input looks mostly numeric. Please enter a sentence or review comment instead of only numbers."

    alpha_count = sum(ch.isalpha() for ch in stripped)
    if alpha_count < 3:
        return False, "The input does not contain enough meaningful words. Please provide a clearer sentence or comment."

    return True, None


# ----------------------------
# Base model predictions
# ----------------------------
def get_bert_probs(text: str):
    """BERT probabilities (5-class)."""
    if not isinstance(text, str) or text.strip() == "":
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    tokenizer, model = get_bert()
    torch = _torch()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().cpu().numpy().flatten().tolist()


def get_roberta_probs(text: str):
    """RoBERTa probabilities (neg, neu, pos)."""
    if not isinstance(text, str) or text.strip() == "":
        return [1 / 3, 1 / 3, 1 / 3]

    tokenizer, model = get_roberta()
    torch = _torch()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().cpu().numpy().flatten().tolist()


def get_senti4sd_probs(text: str):
    """Lexicon-based 3-class probs (neg, neu, pos)."""
    if not isinstance(text, str) or text.strip() == "":
        return [1 / 3, 1 / 3, 1 / 3]

    tokens = text.lower().split()
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    score = pos_count - neg_count

    if score > 0:
        return [0.1, 0.1, 0.8]
    if score < 0:
        return [0.8, 0.1, 0.1]
    return [0.2, 0.6, 0.2]


def bert_label_from_probs(probs):
    """Map 1-5 star BERT to -1, 0, +1."""
    label = int(np.argmax(probs)) + 1
    if label <= 2:
        return -1
    if label == 3:
        return 0
    return 1


def get_base_predictions(text: str):
    bert_probs = get_bert_probs(text)
    roberta_probs = get_roberta_probs(text)
    senti4sd_probs = get_senti4sd_probs(text)

    bert_label = bert_label_from_probs(bert_probs)
    roberta_label = [-1, 0, 1][int(np.argmax(roberta_probs))]
    senti4sd_label = [-1, 0, 1][int(np.argmax(senti4sd_probs))]

    return {
        "bert": {"label": bert_label, "probs": bert_probs},
        "roberta": {"label": roberta_label, "probs": roberta_probs},
        "senti4sd": {"label": senti4sd_label, "probs": senti4sd_probs},
    }


def prepare_features(base_preds):
    """Return a 2D numpy array in the resolver-expected order (no pandas)."""
    bert_probs = base_preds["bert"]["probs"]
    roberta_probs = base_preds["roberta"]["probs"]
    senti4sd_probs = base_preds["senti4sd"]["probs"]

    row = [
        bert_probs[0], bert_probs[1], bert_probs[2], bert_probs[3], bert_probs[4],
        roberta_probs[0], roberta_probs[1], roberta_probs[2],
        senti4sd_probs[0], senti4sd_probs[1], senti4sd_probs[2],
    ]
    return np.array([row], dtype=np.float32)


# ----------------------------
# Explanations
# ----------------------------
def _fallback_explanation(resolved_label, prob_dict, base_preds):
    label_map_local = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    confidence = float(max(prob_dict.values()))
    confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
    return (
        f"The system resolved the sentiment as **{label_map_local[resolved_label]}** "
        f"with {confidence_level} confidence ({confidence:.2f}). "
        f"Base models: BERT={label_map_local[base_preds['bert']['label']]}, "
        f"RoBERTa={label_map_local[base_preds['roberta']['label']]}, "
        f"Senti4SD={label_map_local[base_preds['senti4sd']['label']]}."
    )


def shap_to_text_explanation(shap_rows, predicted_label, pred_probs, top_k=6):
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    target_label = label_map[predicted_label]

    strong_pos, weak_pos, opposing = [], [], []

    for row in shap_rows[:top_k]:
        feature = row["feature"]
        shap_val = row["shap_value"]
        meaning = FEATURE_MEANING.get(feature, feature)
        strength = abs(shap_val)

        if shap_val > 0:
            (strong_pos if strength > 0.1 else weak_pos).append((meaning, shap_val))
        else:
            opposing.append((meaning, shap_val))

    confidence = float(max(pred_probs))
    confidence_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"

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
        explanation += "Additional support came from " + ", ".join([m for m, _ in weak_pos[:2]]) + ". "

    if opposing:
        explanation += (
            "However, signals such as "
            + ", ".join([m for m, _ in opposing[:2]])
            + " opposed this outcome but had weaker influence."
        )

    return explanation


def _compute_shap_for_resolver(features_np, resolved_label):
    """
    Compute SHAP values for resolver prediction.
    IMPORTANT: Expensive. Only call when include_shap=True.
    """
    global shap_explainer

    import shap  # lazy import

    # Very cheap background to keep memory down
    background = np.zeros_like(features_np, dtype=np.float32)

    if shap_explainer is None:
        shap_explainer = shap.Explainer(resolver.predict_proba, background)

    shap_values = shap_explainer(features_np)
    vals = shap_values.values

    pred_class_idx = list(resolver.classes_).index(resolved_label)

    if len(vals.shape) == 2:
        shap_vals = vals[0]
    elif len(vals.shape) == 3:
        shap_vals = vals[0][pred_class_idx, :]
    else:
        shap_vals = vals.flatten()[: len(FEATURES)]

    rows = [{"feature": FEATURES[i], "shap_value": float(shap_vals[i])} for i in range(len(FEATURES))]
    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return rows


# ----------------------------
# Core analyzer
# ----------------------------
def _analyze_text_to_response(text: str, include_shap: bool = False):
    text = (text or "").strip()

    is_valid, validation_msg = validate_input_text(text)
    if not is_valid:
        return None, (validation_msg, 400)

    if resolver is None:
        return None, ("Resolver model not loaded on server", 500)

    base_preds = get_base_predictions(text)
    features_np = prepare_features(base_preds)

    resolved_label = int(resolver.predict(features_np)[0])
    resolved_probs = resolver.predict_proba(features_np)[0]

    classes = list(resolver.classes_)
    prob_dict = {
        -1: float(resolved_probs[classes.index(-1)]),
        0: float(resolved_probs[classes.index(0)]),
        1: float(resolved_probs[classes.index(1)]),
    }

    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    shap_contrib_list = []
    if include_shap:
        try:
            shap_rows = _compute_shap_for_resolver(features_np, resolved_label)
            explanation = shap_to_text_explanation(
                shap_rows,
                predicted_label=resolved_label,
                pred_probs=[prob_dict[-1], prob_dict[0], prob_dict[1]],
                top_k=6
            )
            shap_contrib_list = [
                {
                    "feature": r["feature"],
                    "shap_value": float(r["shap_value"]),
                    "meaning": FEATURE_MEANING.get(r["feature"], r["feature"]),
                }
                for r in shap_rows[:10]
            ]
        except Exception as e:
            print(f"[WARN] SHAP failed: {e}")
            explanation = _fallback_explanation(resolved_label, prob_dict, base_preds)
    else:
        explanation = _fallback_explanation(resolved_label, prob_dict, base_preds)

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


# ----------------------------
# Routes
# ----------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/warmup", methods=["GET"])
def warmup():
    """
    Trigger model downloads and show errors in Render logs.
    Call this once after deploy: /api/warmup
    """
    try:
        get_bert()
        get_roberta()
        return jsonify({"ok": True, "message": "Models loaded/cached"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/")
def index():
    if os.path.isdir(FRONTEND_DIST_DIR):
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")
    return render_template("index.html")


@app.route("/results")
def results():
    if os.path.isdir(FRONTEND_DIST_DIR):
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    if "analysis_results" not in session:
        return redirect(url_for("index"))

    try:
        results_data = json_module.loads(session.get("analysis_results"))
        input_text = session.get("input_text", "")

        session.pop("analysis_results", None)
        session.pop("input_text", None)

        return render_template("results.html", results=results_data, input_text=input_text)
    except Exception:
        return redirect(url_for("index"))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    React API: analyze text and return results JSON (no session).
    SHAP is OFF by default; set include_shap=true from frontend if needed.
    """
    try:
        data = request.get_json(force=True) or {}
        include_shap = bool(data.get("include_shap", False))
        response, err = _analyze_text_to_response(data.get("text"), include_shap=include_shap)
        if err:
            msg, code = err
            return jsonify({"error": msg}), code
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Legacy endpoint: stores results in session for Jinja /results page.
    IMPORTANT: SHAP disabled by default to avoid OOM on free tier.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()

        response, err = _analyze_text_to_response(text, include_shap=False)
        if err:
            msg, code = err
            return jsonify({"error": msg}), code

        session["analysis_results"] = json_module.dumps(response)
        session["input_text"] = text
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    try:
        data = request.get_json(silent=True) or {}
        feedback_entry = {
            "input_text": data.get("input_text", ""),
            "resolved_sentiment": data.get("resolved_sentiment", ""),
            "confidence": data.get("confidence", ""),
            "feedback_label": data.get("feedback_label"),
            "feedback_text": data.get("feedback_text", ""),
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
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """Legacy endpoint for the Jinja results page (session-based)."""
    try:
        data = request.get_json(silent=True) or {}
        feedback_label = data.get("feedback_label")
        feedback_text = data.get("feedback_text", "")

        analysis = json_module.loads(session.get("analysis_results", "{}"))
        resolved_label = analysis.get("resolved", {}).get("label", "")
        confidence = analysis.get("resolved", {}).get("confidence", "")

        feedback_entry = {
            "input_text": session.get("input_text", ""),
            "resolved_sentiment": resolved_label,
            "confidence": confidence,
            "feedback_label": feedback_label,
            "feedback_text": feedback_text,
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
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/<path:path>")
def serve_react_static_or_404(path):
    if path.startswith("api/"):
        abort(404)

    if os.path.isdir(FRONTEND_DIST_DIR):
        full_path = os.path.join(FRONTEND_DIST_DIR, path)
        if os.path.isfile(full_path):
            return send_from_directory(FRONTEND_DIST_DIR, path)
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    abort(404)


if __name__ == "__main__":
    # Local dev only. On Render, use gunicorn.
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
