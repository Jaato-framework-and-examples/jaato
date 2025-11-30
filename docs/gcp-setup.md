# GCP Setup for Jaato

This document describes the steps to set up a Google Cloud Platform project for running LLM models via Vertex AI.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed
- A Google account with billing enabled

## Steps

### 1. Authenticate with gcloud

```bash
gcloud auth login
```

This opens a browser for authentication. After completing the flow, set up Application Default Credentials (required for the SDK):

```bash
gcloud auth application-default login
```

### 2. Create a new GCP project

```bash
gcloud projects create jaato-experiments --name="Jaato Experiments"
```

### 3. Link a billing account

First, list available billing accounts:

```bash
gcloud billing accounts list
```

Then link one to the project:

```bash
gcloud billing projects link jaato-experiments --billing-account=YOUR_BILLING_ACCOUNT_ID
```

### 4. Enable the Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com --project=jaato-experiments
```

### 5. Set default project and region

```bash
gcloud config set project jaato-experiments
gcloud config set ai/region us-central1
```

> **Note:** Use `us-central1` for best model availability. European regions like `europe-west1` may have limited model access.

## Python Environment Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Install the Vertex AI SDK

```bash
.venv/bin/pip install google-cloud-aiplatform
```

This installs `google-genai`, the recommended SDK for Vertex AI generative models.

## Verification

Test the setup with this script:

```python
from google import genai

PROJECT_ID = "jaato-experiments"
LOCATION = "us-central1"

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello, Vertex AI!",
)

print(response.text)
```

Run it:

```bash
.venv/bin/python test_vertex.py
```

## Project Configuration

| Setting | Value |
|---------|-------|
| Project ID | `jaato-experiments` |
| Region | `us-central1` |
| API | Vertex AI (`aiplatform.googleapis.com`) |
| SDK | `google-genai` |
| Model | `gemini-2.0-flash` |

## Notes

- The older `vertexai` SDK is deprecated (June 2025) and will be removed in June 2026
- Use the `google-genai` SDK with `vertexai=True` for Vertex AI access
- Application Default Credentials (ADC) are automatically picked up by the SDK
