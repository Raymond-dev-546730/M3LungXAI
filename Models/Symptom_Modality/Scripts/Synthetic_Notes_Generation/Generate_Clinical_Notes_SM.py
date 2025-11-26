# Import required libraries
import random
import os
from llama_cpp import Llama
from transformers import AutoTokenizer
import logging
import json 
import re
from fuzzywuzzy import fuzz

# Model path and settings
mistral_model_path = "./LLM/Mistral-7B-Instruct-v0.3-f16.gguf"
mistral_context_limit = 10000 # Limited context window

# Logging setup
logging.basicConfig(
    filename="dataset_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Query model with specified parameters
def query_model(model, prompt, temperature=0.5, top_p=0.7):
    max_tokens = mistral_context_limit - len(tokenizer.tokenize(prompt))
    max_tokens = max(max_tokens, 1) 

    response = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return response["choices"][0]["text"].strip()

# Initialize the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Load patient names from JSON
with open("patient_names.json", "r") as f:
    patient_names_data = json.load(f)

# Load symptom synonyms from JSON
with open("symptom_synonyms.json", "r") as f:
    symptom_synonyms = json.load(f)

# Load symptom constraints from JSON
with open("symptom_constraints.json", "r") as f:
    symptom_constraints = json.load(f)


# Generate random patient names
def generate_random_name():
    return f"{random.choice(patient_names_data['first_names'])} {random.choice(patient_names_data['last_names'])}"


# Select random symptoms for the clinical note
def select_random_symptoms(symptom_dict, num_symptoms):
    selected_symptoms = random.sample(list(symptom_dict.keys()), num_symptoms)
    symptoms_text = ", ".join([random.choice(symptom_dict[symptom]) for symptom in selected_symptoms])
    return selected_symptoms, symptoms_text


# Get constraints based on selected symptoms
def get_constraints_for_prompt(selected_symptoms):
    constraints_text = []
    for symptom in selected_symptoms:
        if symptom in symptom_constraints:
            constraint = symptom_constraints[symptom]
            recommendations = " ".join(constraint.get("recommendations", []))
            avoidances = " ".join(constraint.get("avoid", []))
            constraints_text.append(f"- {symptom}:\n  Recommendations: {recommendations}\n  Avoid: {avoidances}")
    return "\n".join(constraints_text)


def extract_and_validate_symptoms(note, selected_symptoms, symptom_synonyms):
    # Extract symptoms wrapped in <<...>> and validate against selected symptoms using fuzzywuzzy matching
    pattern = r'<<(.*?)>>'
    matches = re.finditer(pattern, note)
    
    found_spans = []
    valid_symptoms = set()
    found_symptoms = []
    FUZZY_THRESHOLD = 80  # 80% 
    
    # Extract spans and validate symptoms
    for match in matches:
        span_text = match.group(1)
        start_idx = match.start(1)
        end_idx = match.end(1)
        found_symptoms.append(span_text)
        
        # Check against each symptom and its synonyms using fuzzywuzzy matching
        matched_symptom = None
        highest_ratio = 0
        for symptom in selected_symptoms:
            variants = [symptom] + symptom_synonyms.get(symptom, [])
            
            for variant in variants:
                ratio = fuzz.token_set_ratio(span_text.lower(), variant.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    if ratio >= FUZZY_THRESHOLD:
                        matched_symptom = symptom
        
        if matched_symptom:
            found_spans.append({
                "span": span_text,
                "start": start_idx - 2,
                "end": end_idx + 2,
                "label": "SYMPTOM",
                "source_symptom": matched_symptom
            })
            valid_symptoms.add(matched_symptom)
            # Log successful symptom match with confidence score for validation tracking
            logging.info(f"Matched symptom '{span_text}' to '{matched_symptom}' with {highest_ratio}% confidence")
        else:
            # Log unmatched symptoms to identify potential fuzzywuzzy matching issues
            logging.warning(f"Found unmatched symptom marker: '{span_text}' (highest similarity: {highest_ratio}%)")
    
    # Check for missing symptoms
    missing_symptoms = set(selected_symptoms) - valid_symptoms
    if missing_symptoms:
        logging.warning(f"Missing symptoms in note: {', '.join(missing_symptoms)}") # Showcase missing symptoms (if any)
        return None
    
    # Process valid note
    clean_note = re.sub(r'<<|>>', '', note)
    
    # Adjust spans for removed markers
    adjusted_spans = []
    marker_count = 0
    for span in found_spans:
        adjusted_start = span["start"] - (marker_count * 4)
        adjusted_end = span["end"] - (marker_count * 4) - 4
        
        adjusted_spans.append({
            "span": span["span"],
            "start": adjusted_start,
            "end": adjusted_end,
            "label": "SYMPTOM",
            "source_symptom": span["source_symptom"]
        })
        marker_count += 1
    
    return {
        "text": clean_note,
        "spans": adjusted_spans
    }


def save_human_readable_clinical_note(note, note_id, metadata_path="dataset_index.jsonl"):
    # Saves clean version without markup
    clean_note = re.sub(r'<<|>>', '', note)
    
    formatted_note_id = f"note_{note_id:05d}"
    
    raw_note_path = f"raw_notes/{formatted_note_id}.txt"
    os.makedirs("raw_notes", exist_ok=True)
    
    with open(raw_note_path, "w") as note_file:
        note_file.write(clean_note)
    
    metadata = {
        "note_id": formatted_note_id,
        "annotations_path": f"annotated_notes/{formatted_note_id}.json"
    }
    with open(metadata_path, "a") as meta_file:
        meta_file.write(json.dumps(metadata) + "\n")


def save_labeled_clinical_note(labeled_data, note_id, metadata_path="dataset_index.jsonl"):
    # Saves version with symptom annotations and metadata
    formatted_note_id = f"note_{note_id:05d}"
    
    labeled_path = f"annotated_notes/{formatted_note_id}.json"
    os.makedirs("annotated_notes", exist_ok=True)
    
    with open(labeled_path, "w") as f:
        json.dump(labeled_data, f, indent=2)
    
    # Only write metadata if it doesn't already exist for this note
    metadata_exists = False
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as meta_file:
            for line in meta_file:
                entry = json.loads(line)
                if entry.get("note_id") == formatted_note_id:
                    metadata_exists = True
                    break
    
    if not metadata_exists:
        metadata = {
            "note_id": formatted_note_id,
            "annotations_path": f"annotated_notes/{formatted_note_id}.json"
        }
        with open(metadata_path, "a") as meta_file:
            meta_file.write(json.dumps(metadata) + "\n")


def generate_clinical_note(model, note_id):
    while True:
        num_symptoms = random.randint(2, 5)
        selected_symptoms, symptoms_text = select_random_symptoms(symptom_synonyms, num_symptoms)
        patient_name = generate_random_name()
        constraints_text = get_constraints_for_prompt(selected_symptoms)

        logging.info(f"Generating note {note_id} - Symptoms ({num_symptoms}): {', '.join(selected_symptoms)}") # Log symptoms before prompt generation
        
        # Example clinical note for better model guidance
        example_note = """
*Clinical Note*

Patient Name: Sarah Thompson  
Date: 2025-01-01  

Subjective:  
- Chief Complaint: <<Hippocratic fingers>> and <<jaundice>>.  
- History of Present Illness:  
  Sarah Thompson, a 42-year-old female, presents with the above symptoms. She describes a gradual onset of changes over the past few weeks. She denies any significant recent events or exposures. Family history is notable for relevant conditions.

Objective:  
- Vital Signs: BP: 128/76 mmHg, HR: 78 bpm, RR: 16 breaths/min, Temp: 98.7°F.  
- Physical Examination Findings: Consistent with the symptoms described.  

Assessment:  
1. Chronic condition contributing to the reported symptoms.  
2. Potential systemic causes requiring further investigation.  
3. Rule out underlying conditions, including malignancy or other organ system involvement.  

Plan:  
1. Diagnostics:  
   - Order relevant laboratory tests to evaluate organ function and systemic markers.  
   - Conduct imaging to assess potential underlying causes.  

2. Referrals:  
   - Refer to specialists as appropriate for further workup.  

3. Treatment:  
   - Initiate general supportive care measures.  
   - Provide guidance on lifestyle adjustments and symptom monitoring.  

4. Follow-Up:  
   - Schedule follow-up in one week to review results and reassess.  
   - Educate the patient on warning signs requiring immediate medical attention.  
   - Maintain open communication for any new or worsening concerns.  
"""
        
        # Model prompt that instructs it to use <<...>> markers around symptoms
        prompt = f"""
You are a trusted medical assistant tasked with creating realistic and concise clinical notes for patients. Below is an example clinical note to guide your structure and format:

{example_note}

Now, generate a clinical note based on the following information:

Patient Name: {patient_name}
Symptoms: {symptoms_text}

CRITICAL INSTRUCTION: You MUST wrap EVERY SINGLE symptom mentioned in the symptoms above with << >> markers.
Example of correct symptom formatting: Patient presents with <<frequent respiratory infections>> and <<chronic cough>>.

Guidelines for the note:
- Subjective: State the chief complaint and provide a concise history of present illness, including symptom timeline, associated factors, and relevant context.
- Objective: Document key findings from the physical exam, including vital signs and observations (e.g., "BP: 120/80 mmHg, bilateral wheezing").
- Assessment: List likely diagnoses or differential diagnoses with brief reasoning for each.
- Plan: Outline actionable steps, including diagnostics, treatments, referrals, and follow-up plans.

Additional constraints for the symptoms:
{constraints_text}

Requirements for the note:
- Please ensure your finished clinical note is in the same structure as the provided example clinical note.
- EACH symptom MUST be wrapped in << >> markers. FAILURE TO DO THIS WILL RESULT IN UNSATISFACTORY OUTCOMES.
- All symptoms must be included in the final clinical note, and their phrasing should remain consistent with or closely resemble the original format. 
- UNDER NO CIRCUMSTANCES should you include a disclaimer of ANY KIND in the finished clinical note.
- Respond only with your finished clinical note.
"""
        clinical_note = query_model(model, prompt)
        labeled_data = extract_and_validate_symptoms(clinical_note, selected_symptoms, symptom_synonyms)
        
        if labeled_data:
            save_human_readable_clinical_note(clinical_note, note_id)
            save_labeled_clinical_note(labeled_data, note_id)
            logging.info(f"SUCCESS: Generated and validated note {note_id}")
            return clinical_note
        else:
            logging.warning(f"FAILURE: Regenerating note {note_id} due to validation failure")
            # Continues generating until a valid note is produced


# Generate the specified number of clinical notes in num_notes (10k)
def generate_multiple_notes(num_notes):
    model = Llama(
        model_path=mistral_model_path,
        n_gpu_layers=33, # Layers needed to load the entire LLM onto the GPU
        n_ctx=mistral_context_limit
    )
    for i in range(1, num_notes + 1):
        generate_clinical_note(model, note_id=i)
    del model

if __name__ == "__main__":
    num_notes = 10000 # 10k clinical notes
    generate_multiple_notes(num_notes)