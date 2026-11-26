from transformers import pipeline
import re
import numpy as np

# Debugging: Check if numpy is available
print("Numpy Version:", np.__version__)

# Load Hugging Face NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)

def preprocess_text(text):
    """Normalize whitespace and retain capitalization."""
    normalized_text = " ".join(text.split())
    return normalized_text  # Keep original capitalization

def is_valid_entity(entity_text, text):
    """
    Validate entity to avoid over-redaction but ensure valid names/identifiers are included.
    """
    # Allow very short entities if they are standalone words
    if len(entity_text) < 3:
        word_boundary = r'\b' + re.escape(entity_text) + r'\b'
        return bool(re.search(word_boundary, text))
    
    # For longer entities, ensure they are meaningful by matching full words
    word_boundary = r'\b' + re.escape(entity_text) + r'\b'
    return bool(re.search(word_boundary, text, re.IGNORECASE))

def anonymize_text_with_huggingface(text):
    """Anonymize sensitive information using Hugging Face NER."""
    preprocessed_text = preprocess_text(text)
    print(f"Preprocessed Text Sent to NER Model: {preprocessed_text}")
    
    # Run NER
    ner_results = ner_pipeline(preprocessed_text)
    print(f"NER Results: {ner_results}")
    
    redacted_text = preprocessed_text
    for entity in ner_results:
        entity_text = entity["word"]
        entity_label = entity["entity_group"]
        
        if entity_label in ["PER", "LOC", "ORG", "MISC"] and is_valid_entity(entity_text, preprocessed_text):
            # Replace all valid instances of the entity
            redacted_text = re.sub(
                r'\b' + re.escape(entity_text) + r'\b', 
                f"[REDACTED {entity_label}]", 
                redacted_text, 
                flags=re.IGNORECASE
            )

    print(f"Redacted Text: {redacted_text}")
    return redacted_text
