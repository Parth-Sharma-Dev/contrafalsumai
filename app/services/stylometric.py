import joblib  
import numpy as np
from lime.lime_text import LimeTextExplainer  
import os

from app.utils.text_processing import preprocess_text

PIPELINE_PATH: str = os.path.join("ml_assets", "fake_news_pipeline.pkl")
pipeline = joblib.load(PIPELINE_PATH)

explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])


def predict_proba_wrapper(texts: list[str]) -> np.ndarray:
    cleaned_texts = [preprocess_text(text) for text in texts]
    return pipeline.predict_proba(cleaned_texts)


def analyze_with_lime(text: str, num_features: int = 10) -> dict[str, float]:
    explanation = explainer.explain_instance(
        text, 
        predict_proba_wrapper, 
        labels=(0,), 
        num_features=num_features
    )
    
    suspicious_words: dict[str, float] = {}
    for word, weight in explanation.as_list(label=0):
        if weight > 0:
            suspicious_words[str(word)] = float(round(weight, 4))
            
    return suspicious_words


def get_base_prediction(text: str) -> dict[str, float | int | str]:
    cleaned_text = preprocess_text(text)
    
    prediction = pipeline.predict([cleaned_text])[0]
    probabilities = pipeline.predict_proba([cleaned_text])[0].tolist()
    
    label = int(prediction)
    confidence = float(probabilities[label])
    
    verdict = "REAL" if label == 1 else "FAKE"
    
    return {
        "label": label,
        "verdict": verdict,
        "confidence": confidence
    }
