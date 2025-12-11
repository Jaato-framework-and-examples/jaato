# Google GenAI Setup for Jaato

This document describes how to set up authentication for Google's Gemini models. Jaato supports two endpoints:

| Endpoint | Use Case | Auth Method |
|----------|----------|-------------|
| **Google AI Studio** | Personal development, quick prototyping | API Key |
| **Vertex AI** | Organization/enterprise, production workloads | GCP credentials |

## Quick Start

### Option A: Google AI Studio (Simplest)

Best for personal projects and quick experimentation.

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set the environment variable:
   ```bash
   export GOOGLE_GENAI_API_KEY=your-api-key
   ```
3. Run jaato - it will auto-detect the API key and use AI Studio

### Option B: Vertex AI (Enterprise)

Best for organization accounts with billing controls and IAM.

1. Set up GCP credentials (see detailed steps below)
2. Set environment variables:
   ```bash
   export JAATO_GOOGLE_PROJECT=your-project-id
   export JAATO_GOOGLE_LOCATION=us-central1
   ```
3. Run jaato - it will use ADC or service account credentials

---

## Google AI Studio Setup

### 1. Create an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API key" in the left sidebar
3. Click "Create API key"
4. Copy the key

### 2. Configure Environment

```bash
# Set the API key
export GOOGLE_GENAI_API_KEY=your-api-key

# Or add to .env file
echo "GOOGLE_GENAI_API_KEY=your-api-key" >> .env
```

### 3. Verify

```python
from google import genai

client = genai.Client(api_key="your-api-key")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello!"
)
print(response.text)
```

---

## Vertex AI Setup

### Prerequisites

- Google Cloud SDK (`gcloud`) installed
- A Google account with billing enabled

### 1. Authenticate with gcloud

```bash
gcloud auth login
```

This opens a browser for authentication. After completing the flow, set up Application Default Credentials:

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

### 6. Configure Environment

```bash
# Using JAATO variables (recommended)
export JAATO_GOOGLE_PROJECT=jaato-experiments
export JAATO_GOOGLE_LOCATION=us-central1

# Or using standard GCP variables
export GOOGLE_CLOUD_PROJECT=jaato-experiments

# Or using legacy variables (still supported)
export PROJECT_ID=jaato-experiments
export LOCATION=us-central1
```

---

## Authentication Methods

Jaato supports multiple authentication methods for Vertex AI:

### Application Default Credentials (ADC) - Recommended for Development

ADC automatically finds credentials in this order:
1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
2. User credentials from `gcloud auth application-default login`
3. GCE/GKE metadata server (when running on Google Cloud)

```bash
# Set up ADC for local development
gcloud auth application-default login
```

### Service Account Key - For CI/CD and Production

1. Create a service account:
   ```bash
   gcloud iam service-accounts create jaato-sa \
     --display-name="Jaato Service Account"
   ```

2. Grant required role:
   ```bash
   gcloud projects add-iam-policy-binding jaato-experiments \
     --member="serviceAccount:jaato-sa@jaato-experiments.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. Create and download key:
   ```bash
   gcloud iam service-accounts keys create jaato-sa-key.json \
     --iam-account=jaato-sa@jaato-experiments.iam.gserviceaccount.com
   ```

4. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/jaato-sa-key.json
   ```

---

## Environment Variables Reference

| Variable | Description | Required For |
|----------|-------------|--------------|
| `GOOGLE_GENAI_API_KEY` | API key for AI Studio | AI Studio |
| `JAATO_GOOGLE_PROJECT` | GCP project ID | Vertex AI |
| `JAATO_GOOGLE_LOCATION` | GCP region | Vertex AI |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account key | Vertex AI (SA auth) |
| `JAATO_GOOGLE_AUTH_METHOD` | Force auth method: `auto`, `api_key`, `service_account_file`, `adc` | Optional |
| `JAATO_GOOGLE_USE_VERTEX` | Force endpoint: `true` (Vertex) or `false` (AI Studio) | Optional |

**Legacy variables (still supported):**
| Variable | Equivalent |
|----------|------------|
| `PROJECT_ID` | `JAATO_GOOGLE_PROJECT` |
| `LOCATION` | `JAATO_GOOGLE_LOCATION` |
| `GOOGLE_CLOUD_PROJECT` | `JAATO_GOOGLE_PROJECT` |

---

## Python Environment Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2. Install dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

This installs `google-genai`, the recommended SDK for Google's generative models.

---

## Verification

### Test AI Studio

```python
from google import genai

client = genai.Client(api_key="your-api-key")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, AI Studio!",
)
print(response.text)
```

### Test Vertex AI

```python
from google import genai

client = genai.Client(
    vertexai=True,
    project="jaato-experiments",
    location="us-central1",
)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, Vertex AI!",
)
print(response.text)
```

---

## Troubleshooting

### "No credentials found"

```
CredentialsNotFoundError: No credentials found for authentication method: auto
```

**Fix:** Set up credentials using one of:
- `gcloud auth application-default login` (for ADC)
- `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` (for service account)
- `export GOOGLE_GENAI_API_KEY=your-key` (for AI Studio)

### "Permission denied"

```
CredentialsPermissionError: Credentials lack required permissions
```

**Fix:** Grant the required IAM role:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### "Project not found"

```
ProjectConfigurationError: Project ID is required for Vertex AI
```

**Fix:** Set the project:
```bash
export JAATO_GOOGLE_PROJECT=your-project-id
```

---

## Notes

- The older `vertexai` SDK is deprecated (June 2025) and will be removed in June 2026
- Use the `google-genai` SDK with `vertexai=True` for Vertex AI access
- AI Studio is region-agnostic; Vertex AI requires explicit region selection
- Use `us-central1` for best model availability on Vertex AI
