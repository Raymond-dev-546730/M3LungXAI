# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

# Clinical note input
Clinical_Note = """

"""

# Symptom weights mapping
SYMPTOM_WEIGHTS = {
    # Haastrup et al. (2020) Tables 2 & 3: "Predictive values of lung cancer alarm symptoms in the general population: a nationwide cohort study"
    "Loss of appetite": 0.3,           # PPV 0.3% 
    "Hoarseness": 0.3,                 # PPV 0.3% 
    "Dyspnea": 0.2,                    # PPV 0.2% 
    "Persistent worsening cough": 0.2, # PPV 0.2% 
    "Extreme fatigue": 0.1,            # PPV 0.1% 
    
    "Hemoptysis": 0.3,  # Data suppressed (n≤3), assigned max measured PPV (0.3%) per strongest predictor statement
    
    # Symptoms not measured in study - assigned minimum empirical PPV (0.1%)
    "Unexplained weight loss": 0.1,
    "Pleuritic chest pain": 0.1,
    "Recurring lung infections": 0.1,
    "Unexpected wheezing onset": 0.1,
    "Cervical/Axillary lymphadenopathy": 0.1,
    "Headache": 0.1,
    "Bone pain": 0.1,
    "Hippocratic fingers": 0.1,
    "Jaundice": 0.1,
    "New-onset seizures": 0.1,
    "Facial and cervical edema": 0.1,
    "Dysphagia": 0.1,
    "Swollen veins in the Neck & Chest": 0.1,
    "Ipsilateral Anhidrosis": 0.1,
    "Ptosis": 0.1,
    "Ipsilateral Miosis": 0.1
}

# Load the trained model and tokenizer
model_path = "./Symptom_Modality/SpanBERT-SCM-Large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def load_symptom_dataset(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r') as f:
        return json.load(f)

def find_matching_head_symptom(
    symptom_text: str,
    symptom_dataset: Dict[str, List[str]],
    threshold: int = 80 # 80%
) -> Optional[str]:
    best_match_score = 0
    best_match_head = None
    
    for head_symptom, synonyms in symptom_dataset.items():
        # Check match with head symptom
        head_score = fuzz.token_sort_ratio(symptom_text.lower(), head_symptom.lower())
        if head_score > best_match_score and head_score >= threshold:
            best_match_score = head_score
            best_match_head = head_symptom
            
        # Check match with synonyms
        for synonym in synonyms:
            synonym_score = fuzz.token_sort_ratio(symptom_text.lower(), synonym.lower())
            if synonym_score > best_match_score and synonym_score >= threshold:
                best_match_score = synonym_score
                best_match_head = head_symptom
    
    return best_match_head

def calculate_confidence_score(logits: torch.Tensor, pred_idx: int) -> float:
    probabilities = F.softmax(logits, dim=-1)
    confidence = probabilities[pred_idx].item()
    return round(confidence * 100, 2)

def extract_spans_and_calculate_weights(
    text: str,
    tokenizer,
    model,
    device,
    symptom_dataset: Dict[str, List[str]]
) -> Tuple[List[Dict], Dict[str, float], float, float]:
    # offset_mapping returns character indices for each token
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    offsets = tokenized["offset_mapping"].squeeze()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    verified_spans = []
    identified_weights = {symptom: 0.0 for symptom in SYMPTOM_WEIGHTS.keys()}
    used_head_symptoms = set()
    confidence_scores = []
    in_span = False
    span_start = None

    for idx, (pred, (start, end)) in enumerate(zip(predictions, offsets)):
        if pred == 1 and not in_span:  # Start of a new span (1 for SYMPTOM)
            in_span = True
            span_start = int(start)
        elif (pred == 0 or idx == len(predictions)-1) and in_span:  # End of current span
            span_text = text[span_start:int(end)]
            if span_text.strip():
                head_symptom = find_matching_head_symptom(span_text, symptom_dataset)
                confidence_score = calculate_confidence_score(logits[0, idx], pred)
                confidence_scores.append(confidence_score)
                
                if head_symptom and head_symptom not in used_head_symptoms:
                    weight = SYMPTOM_WEIGHTS.get(head_symptom, 0.0)
                    verified_spans.append({
                        "text": span_text,
                        "start": span_start,
                        "end": int(end),
                        "label": "SYMPTOM",
                        "head_symptom": head_symptom,
                        "weight": weight,
                        "confidence_score": confidence_score
                    })
                    identified_weights[head_symptom] = weight
                    used_head_symptoms.add(head_symptom)
            in_span = False

    # Calculate total weight from individual symptoms only
    total_weight = round(sum(identified_weights.values()), 2)
    
    # Calculate overall confidence
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    return verified_spans, identified_weights, total_weight, overall_confidence

# Conservative thresholds - no validated additive model exists
def assess_risk_level(total_weight: float) -> str:
    # Tables 2 & 3: ≥2 symptoms = PPV 0.3%, same as single high-PPV symptom (non-additive)
    
    if total_weight >= 0.8:
        return "High Risk"      # Heavy multi-symptom burden (≥3-4 symptoms)
    elif total_weight >= 0.4:
        return "Moderate Risk"  # Multi-symptom pattern (≥2 symptoms)
    else:
        return "Low Risk" # Single low-moderate symptom only

def set_clinical_note(text: str):
    global Clinical_Note
    Clinical_Note = text


prediction_results = {}


def predict():
    global Clinical_Note
    global prediction_results
    prediction_results = {}
    # Load symptom dataset
    symptom_dataset = load_symptom_dataset("SYMPTOM_DATASET.json")

    # Get predictions
    verified_predictions, weights, total_weight, overall_confidence = extract_spans_and_calculate_weights(
        Clinical_Note,
        tokenizer,
        model,
        device,
        symptom_dataset
    )

    # Get risk level
    risk_level = assess_risk_level(total_weight)

    # Print results
    print(f"[DEBUG] CONFIDENCE: {overall_confidence}%")
    print(f"\nRisk Level: {risk_level}")
    print(f"Total Weight: {total_weight:.2f}%")
    for symptom in verified_predictions:
        print(f"\nDetected Text: \"{symptom['text']}\"")
        print(f"Location: Characters {symptom['start']} to {symptom['end']}")
        print(f"Matched Head Symptom: {symptom['head_symptom']}")
    
    # Clear note after prediction
    Clinical_Note = """"""
    
    prediction_results = {
        'clinical_results': {
            'symptoms': verified_predictions,
            'risk_level': risk_level,
            'risk_weight': total_weight,
            's_confidence': overall_confidence 
        }
    }

if __name__ == "__main__":
    predict()