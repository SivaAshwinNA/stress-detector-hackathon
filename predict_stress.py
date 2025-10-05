# predict_stress.py
from transformers import pipeline

# Load once at startup — takes ~10s the first time, instant later
print("⏳ Loading stress detection model...")
classifier = pipeline(
    "text-classification",
    model="./stress_detector_model",
    tokenizer="./stress_detector_model"
)
print("✅ Model loaded successfully!")

recommendation = {
    0: "Keep routine, light relaxation",
    1: "Take breaks, short meditation",
    2: "Moderate meditation, talk to friends/family",
    3: "Mindfulness, consult counselor",
    4: "Seek professional help immediately"
}

def predict_stress(sentence):
    result = classifier(sentence)
    # Some models return labels like "LABEL_0", "LABEL_1", etc.
    label_str = result[0]['label']
    predicted_level = int(label_str.split('_')[-1])
    rec = recommendation.get(predicted_level, "Unknown")
    return {
        "sentence": sentence,
        "predicted_level": predicted_level,
        "stress_label": label_str,
        "recommendation": rec
    }
