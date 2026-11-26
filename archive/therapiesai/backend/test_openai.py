import openai
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env or environment variables.")

# Initialize the OpenAI API key
openai.api_key = api_key

# Test OpenAI API
def test_openai():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with "gpt-4" if preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about OpenAI API changes."},
            ],
        )
        print("Response:", response["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error: {e}")

# Run the test function
if __name__ == "__main__":
    test_openai()
