# AI Security Incident Triage & GenAI Analyst

An end-to-end AI engineering project for security incident triage combining classical machine learning, a PyTorch neural network, hosted Generative AI, application-level guardrails, FastAPI model serving, automated testing, GitHub Actions CI and Docker containerization.

The project uses the Microsoft Security Incident Prediction (GUIDE) dataset to classify security incidents into three triage categories:

- BenignPositive
- FalsePositive
- TruePositive

A GenAI analyst layer then produces a conservative, human-readable analysis of the classifier result.

---

## Architecture

```text
Security Incident
        |
        v
Preprocessing
        |
        v
PyTorch Classifier
        |
        v
Prediction + Model Confidence
        |
        v
Confidence Routing
        |
        +---- Low confidence ----> Deterministic Safe Fallback
        |
        v
Hosted Qwen3-8B
        |
        v
Structural Validation
        |
        v
Grounding Guard
        |
        +---- Invalid / Unsupported ----> Safe Fallback
        |
        v
Validated Analyst Response
        |
        v
FastAPI REST API
```

---

## Machine Learning Development

A Logistic Regression model was first implemented as a classical machine learning baseline.

Both the baseline and the PyTorch neural network were evaluated on the same leakage-aware, group-aware development validation split.

| Model | Macro F1 |
|---|---:|
| Logistic Regression baseline | 0.7387 |
| PyTorch Neural Network | 0.7455 |

The PyTorch model improved development Macro-F1 from:

```text
0.7387 -> 0.7455
```

The reported metrics are development-validation results based on the first 100,000 GUIDE training rows used during model development.

They should not be interpreted as independent final test-set performance.

---

## PyTorch Model

The neural network uses:

- Learned embeddings for categorical features
- Scaled numerical features
- Weighted CrossEntropyLoss
- AdamW optimization
- Dropout regularization
- Early stopping based on validation Macro-F1

The model contains approximately 449k trainable parameters.

Architecture:

```text
Categorical Embeddings
        +
Numerical Features
        |
        v
242-dimensional input
        |
        v
Linear(242 -> 128)
ReLU
Dropout
        |
        v
Linear(128 -> 64)
ReLU
Dropout
        |
        v
Linear(64 -> 3)
        |
        v
Class Logits
```

---

## Feature Processing

The predictive model combines categorical and numerical features.

Categorical values are mapped using vocabularies learned only from training data.

Unknown values are mapped to a dedicated unknown category, while missing categorical values use a dedicated missing-value token.

Numerical features include:

- Hour
- DayOfWeek
- IsWeekend
- HasMitreTechnique

Timestamp-derived features are recreated during inference using the same logic used during training.

Numerical values are imputed using training-derived medians and transformed using the saved training scaler.

The preprocessing artifacts are saved separately from the PyTorch model to preserve training-serving consistency.

---

## Production Inference

The trained PyTorch model and preprocessing artifacts are loaded by:

```text
app/classifier.py
```

The production inference module is independent of notebook state.

A parity test was performed using the same GUIDE incident row used during development.

The production module reproduced the notebook prediction:

```text
Predicted class: TruePositive
Confidence: 0.9901

BenignPositive : 0.0015
FalsePositive  : 0.0085
TruePositive   : 0.9901
```

This verifies consistency between notebook experimentation and production inference.

---

## GenAI Analyst

The application integrates the hosted model:

```text
Qwen/Qwen3-8B
```

through Hugging Face Inference Providers.

The GenAI layer receives:

- Classifier prediction
- Model confidence
- Limited incident explanation context

The explanation context may contain information such as:

- Incident category
- Entity type
- Evidence role
- MITRE technique availability

The ground-truth incident label is never passed to the language model.

The required output format is:

```text
Assessment:
Recommended action:
Caveat:
```

---

## GenAI Model Exploration

Several language models were explored during development.

### FLAN-T5 Small

The model frequently echoed instructions or failed to follow the requested response structure.

### FLAN-T5 Base

The larger model produced more fluent text but still did not reliably perform the analyst task.

### SmolLM2-360M-Instruct

The model showed better task understanding but produced semantic and schema-related problems.

### SmolLM2-1.7B-Instruct

Local inference was attempted but was not reliably feasible within the available CPU/RAM environment.

This was treated as a local compute limitation rather than a model-quality evaluation.

### Qwen3-8B

The final application therefore uses hosted:

```text
Qwen/Qwen3-8B
```

through Hugging Face Inference Providers.

This allows a stronger language model to be used while keeping the predictive PyTorch classifier local.

---

## Prompt Engineering

The hosted LLM receives explicit grounding instructions.

The prompt instructs the model to:

- Use only supplied information
- Avoid inferring that an IP address is malicious
- Avoid identifying entities as attackers or attack sources without evidence
- Avoid claiming that explanation-context fields caused the classifier prediction
- Avoid unsupported containment recommendations
- Treat unavailable information as unknown
- Treat classifier confidence as model confidence rather than proof

The language model must return exactly three sections:

```text
Assessment:
Recommended action:
Caveat:
```

---

## GenAI Guardrails

Generated responses pass through application-level validation before being returned by the API.

### Structural Validation

The validator checks for the required sections:

- Assessment
- Recommended action
- Caveat

Responses that do not follow the required structure are rejected.

### Grounding Guard

A deterministic grounding guard checks for known unsupported claim patterns.

Examples include:

- Calling an IP address malicious without evidence
- Identifying an IP as an attack source
- Claiming that an attack is confirmed without supporting information
- Recommending unsupported blocking or isolation actions
- Claiming that limited explanation fields caused the classifier prediction

This guard is an application-level safety mechanism.

It is not intended to be a complete hallucination detector.

If validation fails, the generated answer is discarded and replaced with a deterministic safe fallback.

---

## Safe Fallback

The deterministic fallback:

- Reports the classifier prediction
- Reports model confidence
- Summarizes only supplied explanation context
- Recommends analyst review
- Preserves uncertainty
- Avoids unsupported causal explanations

Example:

```text
Assessment: The classifier predicted TruePositive with a model confidence
score of 0.9999. The supplied explanation context includes category
InitialAccess, entity type Ip, and evidence role Related.

Recommended action: Review the available incident evidence and associated
context before making a final analyst disposition.

Caveat: No MITRE technique information was supplied. The explanation
context does not establish why the classifier produced its prediction.
```

---

## Confidence Routing

The application currently uses a configurable operational routing threshold of:

```text
0.80
```

Low-confidence predictions skip the hosted LLM call:

```text
Classifier
    |
    v
Confidence < 0.80
    |
    v
LLM call skipped
    |
    v
Safe Fallback
```

This avoids unnecessary external API usage and latency for cases already routed conservatively.

The threshold is an application routing policy.

It should not be interpreted as a calibrated probability of correctness.

---

## External Provider Failure Handling

Hosted GenAI inference is protected with application-level error handling.

If the provider cannot return a response because of:

- Network failure
- Provider outage
- Authentication problems
- Rate limiting
- Inference errors

the application does not crash.

Instead, it returns the deterministic safe fallback.

---

## FastAPI REST API

The application exposes three endpoints.

### GET /health

Health check.

Example:

```json
{
  "status": "ok"
}
```

---

### POST /predict

Runs PyTorch classifier inference.

Returns:

- Predicted incident class
- Model confidence
- Probability for each class

Example:

```json
{
  "predicted_class": "TruePositive",
  "confidence": 0.9998565912,
  "probabilities": {
    "BenignPositive": 0.0000299,
    "FalsePositive": 0.0001134,
    "TruePositive": 0.9998566
  }
}
```

---

### POST /analyze

Runs the complete AI analysis pipeline.

```text
Incident
   |
   v
Pydantic Validation
   |
   v
PyTorch Classifier
   |
   v
Prediction + Confidence
   |
   v
Confidence Routing
   |
   v
Hosted Qwen3-8B
   |
   v
Structural Validation
   |
   v
Grounding Guard
   |
   v
Validated LLM Response
        OR
Deterministic Safe Fallback
```

Example:

```json
{
  "prediction": {
    "predicted_class": "TruePositive",
    "confidence": 0.9998565912,
    "probabilities": {
      "BenignPositive": 0.0000299,
      "FalsePositive": 0.0001134,
      "TruePositive": 0.9998566
    }
  },
  "route": "safe_fallback",
  "reason": "grounding_violation",
  "analyst_response": "Assessment: The classifier predicted TruePositive..."
}
```

Possible routing reasons include:

```text
validated_llm_response
low_classifier_confidence
invalid_llm_structure
grounding_violation
llm_inference_error
llm_not_configured
```

---

## Request Validation

Pydantic models validate API requests before they reach the classifier.

Required fields are enforced.

Unexpected fields are rejected using:

```text
extra="forbid"
```

For example, a request missing:

```text
Category
```

returns an HTTP validation error before model inference occurs.

---

## Automated Testing

The application includes automated tests covering its major execution paths.

Current local test result:

```text
9 passed
```

Tests cover:

- API health
- PyTorch prediction
- Probability consistency
- Required request fields
- Rejection of unexpected request fields
- Valid LLM response routing
- Grounding violations
- Invalid LLM output structure
- Hosted inference failure handling
- Low-confidence LLM short-circuiting

---

## Mocked LLM Testing

External Hugging Face calls are mocked during automated tests.

This allows the test suite to simulate:

```text
Valid LLM response
        |
        v
llm_response
```

```text
Grounding violation
        |
        v
safe_fallback
```

```text
Invalid structure
        |
        v
safe_fallback
```

```text
Provider failure
        |
        v
safe_fallback
```

```text
Low classifier confidence
        |
        v
LLM call skipped
```

This keeps the test suite:

- Fast
- Deterministic
- Independent of Internet connectivity
- Independent of Hugging Face availability
- Independent of API credentials

---

## Continuous Integration with GitHub Actions

The repository includes a GitHub Actions CI workflow:

```text
.github/workflows/ci.yml
```

The workflow automatically runs when:

```text
Code is pushed to main
        OR
A Pull Request targets main
```

CI pipeline:

```text
GitHub Push / Pull Request
            |
            v
Checkout Repository
            |
            v
Set Up Python 3.13
            |
            v
Install requirements-dev.txt
            |
            v
Run pytest
            |
            v
Tests Pass / Fail
```

The workflow runs:

```bash
python -m pytest -v
```

Hosted LLM credentials are not required by CI because external LLM inference is mocked during automated tests.

No Hugging Face token is stored in the GitHub Actions workflow.

---

## Docker

The application is containerized using Docker.

The production image contains:

- FastAPI application
- PyTorch classifier
- Saved preprocessing artifacts
- Production Python dependencies

Raw datasets, development files, local virtual environments, and secrets are excluded from the image.

### Build

```bash
docker build -t ai-security-incident-triage .
```

### Run without hosted GenAI credentials

```bash
docker run --rm -p 8000:8000 ai-security-incident-triage
```

The `/health` and `/predict` endpoints remain functional.

The application can also safely fall back when hosted GenAI is not configured.

### Run with hosted GenAI

```bash
docker run --rm -p 8000:8000 --env-file .env ai-security-incident-triage
```

The Hugging Face token is supplied only at runtime.

It is not stored inside the Docker image.

---

## Docker Validation

The Dockerized application was validated using:

```text
GET  /health
POST /predict
POST /analyze
```

The Dockerized `/predict` endpoint reproduced the expected PyTorch inference result.

The Dockerized `/analyze` pipeline successfully:

- Loaded the PyTorch classifier
- Called the hosted Hugging Face model
- Applied the grounding guard
- Rejected an unsupported generated response
- Returned the deterministic safe fallback

---

## Project Structure

```text
ai-security-incident-triage/
|
├── .github/
│   └── workflows/
│       └── ci.yml
|
├── app/
│   ├── __init__.py
│   ├── classifier.py
│   ├── genai.py
│   ├── main.py
│   └── schemas.py
|
├── models/
│   ├── pytorch_incident_triage.pt
│   └── pytorch_preprocessing.joblib
|
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ml_baseline.ipynb
│   ├── 03_pytorch_model.ipynb
│   └── 04_genai_analyst.ipynb
|
├── tests/
│   └── test_api.py
|
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

Activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

---

## Hugging Face Configuration

Create a local file named:

```text
.env
```

containing:

```text
HF_TOKEN=your_huggingface_token
```

The `.env` file is excluded from Git and the Docker build context.

Never commit credentials or API tokens to the repository.

---

## Run Locally

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

```text
GET  /health
POST /predict
POST /analyze
```

---

## Run Tests

```bash
python -m pytest -v
```

Current local result:

```text
9 passed
```

---

## Dependencies

Production dependencies are stored in:

```text
requirements.txt
```

Development and testing dependencies are stored in:

```text
requirements-dev.txt
```

The development requirements include the production requirements plus testing tools such as pytest and httpx.

---

## Technologies

### Machine Learning

- Python
- PyTorch
- scikit-learn
- Pandas
- NumPy

### Generative AI

- Hugging Face Inference Providers
- Qwen3-8B
- Prompt engineering
- Application-level guardrails

### API & Software Engineering

- FastAPI
- Pydantic
- REST APIs
- pytest
- Mocking

### DevOps

- Docker
- Git
- GitHub
- GitHub Actions
- Continuous Integration

---

## Dataset

This project uses the Microsoft Security Incident Prediction (GUIDE) dataset.

The prediction task is multiclass security incident triage:

```text
BenignPositive
FalsePositive
TruePositive
```

Raw dataset files are intentionally excluded from the repository.

The official GUIDE test dataset is not used to support the development-validation metrics reported above.

---

## Model Evaluation Notes

Development comparison:

| Model | Macro F1 |
|---|---:|
| Logistic Regression | 0.7387 |
| PyTorch Neural Network | 0.7455 |

FalsePositive was the most difficult class during development.

The Logistic Regression baseline used one-hot encoded categorical features.

The PyTorch model instead learns compact categorical embeddings.

Classifier confidence is derived from softmax output.

It should not be interpreted as a statistically calibrated probability of correctness.

---

## Security and Reliability Design

The application intentionally separates predictive machine learning from Generative AI.

```text
Predictive ML
     |
     v
Prediction
     |
     v
GenAI Analyst Assistance
```

The LLM does not determine the incident classification.

The PyTorch classifier produces the prediction first.

The language model operates only as a downstream analyst-assistance layer.

This separation allows the application to:

- Preserve classifier output independently of the LLM
- Reject unsupported generated explanations
- Skip GenAI inference for low-confidence cases
- Continue operating if the hosted provider fails
- Return deterministic fallback responses
- Test the predictive and GenAI layers independently

---

## Key Engineering Concepts Demonstrated

This project demonstrates:

- Exploratory data analysis
- Data preprocessing
- Leakage-aware validation
- Group-aware train/validation splitting
- Classical ML baseline development
- Multiclass classification
- Deep learning with PyTorch
- Learned categorical embeddings
- Weighted CrossEntropyLoss
- AdamW optimization
- Dropout regularization
- Early stopping
- Macro-F1 evaluation
- Saved model artifacts
- Reusable production inference
- Training-serving preprocessing consistency
- FastAPI model serving
- REST API development
- Pydantic request validation
- Generative AI integration
- Hosted LLM inference
- Prompt engineering
- Confidence-based routing
- Structural output validation
- Deterministic grounding guardrails
- Safe fallback design
- External provider failure handling
- Automated testing
- Mocking external LLM calls
- Docker containerization
- Runtime secret management
- GitHub Actions
- Continuous Integration
- Separation of development and production dependencies

---

## Purpose

This project was developed as an AI engineering portfolio project demonstrating the transition from model experimentation to a tested, containerized, and CI-enabled AI application.

The focus is not only on predictive performance, but also on the engineering required to integrate machine learning and generative AI into a reliable application workflow.
