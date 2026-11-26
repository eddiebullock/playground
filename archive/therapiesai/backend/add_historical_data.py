import firebase_admin
from firebase_admin import credentials, firestore
import random
import datetime

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your key file
firebase_admin.initialize_app(cred)

# Get Firestore Client
db = firestore.client()

# Reference the "journals" collection
journal_collection = db.collection('journals')

# Function to generate random historical data
def generate_historical_data(user_id, num_entries):
    for _ in range(num_entries):
        # Generate random journal entry text
        entry_text = (
            "Today was a mix of emotions. I spent some time reflecting on "
            "personal growth and made progress toward my goals. "
            "I also enjoyed a good meal and a walk outdoors."
        )
        # Create a journal entry
        entry = {
            "entry": entry_text,
            "happiness": random.randint(1, 10),  # Random happiness score
            "timestamp": datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365)),  # Random past date
            "userId": user_id  # Associate data with the specified user
        }
        
        # Add to Firestore
        journal_collection.add(entry)

    print(f"Added {num_entries} historical entries for user: {user_id}")

# Run the function
if __name__ == "__main__":
    user_ids = ["94SfopukcOOvAmalHxHMWUAyzma2", "TQfB0fU5WoTeUNUGB1RLK3v8zuC3"]  # User IDs to add entries for
    num_entries = 50  # Number of historical entries to add per user
    
    for user_id in user_ids:
        generate_historical_data(user_id, num_entries)
