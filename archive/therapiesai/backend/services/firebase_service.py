from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

db = None

def initialize_firebase():
    """
    Initializes Firebase and Firestore.
    """
    global db
    if db is not None:
        print("Firebase already initialized.")
        return

    print("Initializing Firebase...")
    try:
        firebase_cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        if not firebase_cred_path:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_PATH environment variable is not set.")

        if not os.path.exists(firebase_cred_path):
            raise FileNotFoundError(f"Service account file not found: {firebase_cred_path}")

        cred = credentials.Certificate(firebase_cred_path)
        initialize_app(cred)

        db = firestore.client()
        print("Firebase initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        raise

def update_streak(user_id):
    """
    Update the user's journaling streak based on journal entries.
    """
    try:
        journal_ref = db.collection("journals")
        query = journal_ref.where("userId", "==", user_id).order_by("timestamp", direction=firestore.Query.DESCENDING)
        docs = query.stream()

        entries = [doc.to_dict() for doc in docs]
        if not entries:
            return 0

        # Calculate streak
        streak = 1
        for i in range(1, len(entries)):
            current_date = entries[i - 1]["timestamp"].date()
            prev_date = entries[i]["timestamp"].date()
            if (current_date - prev_date).days == 1:
                streak += 1
            elif (current_date - prev_date).days > 1:
                break

        # Save streak to Firestore
        user_ref = db.collection("users").document(user_id)
        user_ref.set({"streak": streak}, merge=True)

        return streak
    except Exception as e:
        print(f"Error updating streak: {e}")
        return 0

def save_journal_entry(user_id, entry, happiness, themes, sentiment):
    """
    Saves a journal entry to Firestore and updates the journaling streak.
    """
    try:
        if db is None:
            raise ValueError("Firestore client is not initialized. Call initialize_firebase() first.")

        journal_ref = db.collection("journals")
        journal_ref.add({
            "userId": user_id,
            "entry": entry,
            "happiness": happiness,
            "themes": themes,
            "sentiment": sentiment,
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        print("Journal entry saved successfully.")

        # Update streak after adding the entry
        streak = update_streak(user_id)
        print(f"Updated streak: {streak} days")

        return {"message": "Entry saved successfully", "streak": streak}
    except Exception as e:
        print(f"Error saving journal entry: {e}")
        return {"error": "Failed to save entry"}

def get_user_entries(user_id):
    """
    Fetch all journal entries for a user from Firestore.
    """
    try:
        if db is None:
            raise ValueError("Firestore client is not initialized. Call initialize_firebase() first.")

        journal_ref = db.collection("journals")
        query = journal_ref.where("userId", "==", user_id).stream()
        entries = []
        for doc in query:
            data = doc.to_dict()
            entries.append({
                "entry": data.get("entry", ""),
                "happiness": data.get("happiness", 0),
                "timestamp": data.get("timestamp")
            })
        print(f"Fetched {len(entries)} entries for user {user_id}.")
        return entries
    except Exception as e:
        print(f"Error fetching journal entries: {e}")
        return []
