from transformers import pipeline
import spacy

# Initialize Hugging Face pipelines for zero-shot classification and sentiment analysis
theme_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Initialize SpaCy for better sentence tokenization
nlp = spacy.load("en_core_web_sm")

# Define themes for zero-shot classification
candidate_labels = [
    "mental health", "self-care", "stress management", "happiness", "anxiety",
    "family", "friendships", "relationships", "work", "career growth",
    "productivity", "finances", "education", "learning", "fitness",
    "exercise", "diet", "sleep", "travel", "hobbies", "creativity",
    "spirituality", "purpose", "mindfulness", "growth", "motivation"
]

# Mapping of model labels to sentiment categories
label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def extract_themes(entry):
    """
    Extract themes from the journal entry using zero-shot classification.
    """
    try:
        classification_result = theme_classifier(entry, candidate_labels, multi_label=True)
        themes = [
            label for label, score in zip(
                classification_result["labels"], classification_result["scores"]
            ) if score > 0.5  # Adjust confidence threshold as needed
        ]
        return themes
    except Exception as e:
        print(f"Error extracting themes: {e}")
        return []

def analyze_sentiment(entry):
    """
    Analyze the sentiment of the journal entry using a multi-class sentiment analysis model.
    """
    try:
        # Use SpaCy for sentence segmentation
        doc = nlp(entry)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]  # Filter empty sentences

        sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

        # Process each sentence individually
        for sentence in sentences:
            try:
                sentiment = sentiment_analyzer(sentence)[0]
                label = label_mapping.get(sentiment["label"], "neutral")  # Map model label to sentiment
                score = sentiment["score"]
                sentiment_scores[label] += score
                print(f"Sentence: '{sentence}' Sentiment: {label}, Score: {score}")  # Debugging output
            except Exception as inner_e:
                print(f"Error analyzing sentence sentiment: {inner_e}")

        # Aggregate and normalize scores
        total_score = sum(sentiment_scores.values())
        if total_score == 0:
            print("Warning: Total sentiment score is zero. Defaulting to 'neutral'.")
            return "neutral"

        for key in sentiment_scores:
            sentiment_scores[key] /= total_score

        # Determine overall sentiment
        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        print(f"Aggregated Sentiment Scores: {sentiment_scores}")  # Debugging output
        return overall_sentiment
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "neutral"  # Default to neutral on error

def analyze_entry(entry):
    """
    Analyze the journal entry for themes and sentiment.
    """
    try:
        # Extract themes and sentiment
        themes = extract_themes(entry)
        sentiment = analyze_sentiment(entry)
        return {
            "themes": themes,
            "sentiment": sentiment,
        }
    except Exception as e:
        print(f"Error analyzing entry: {e}")
        return None
