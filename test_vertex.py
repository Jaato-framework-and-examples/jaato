"""Simple test to verify VertexAI connection and model access using google-genai SDK."""

from google import genai
from google.genai.types import HttpOptions

PROJECT_ID = "jaato-experiments"
LOCATION = "us-central1"

def main():
    print(f"Initializing GenAI client with project={PROJECT_ID}, location={LOCATION}")

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    print("Sending test prompt to gemini-2.0-flash...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say 'Hello from VertexAI!' in exactly 5 words.",
    )

    print(f"Response: {response.text}")
    print("\nSuccess! VertexAI is working correctly.")

if __name__ == "__main__":
    main()
