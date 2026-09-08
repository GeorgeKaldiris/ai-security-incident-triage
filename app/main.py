from fastapi import FastAPI, HTTPException

from app.classifier import IncidentClassifier
from app.genai import IncidentAnalyst
from app.schemas import (
    IncidentInput,
    PredictionResponse,
    AnalysisResponse,
)


app = FastAPI(
    title="AI Security Incident Triage API",
    description=(
        "PyTorch-based security incident "
        "triage and GenAI analyst service."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Load application services once at startup
# ---------------------------------------------------------

classifier = IncidentClassifier()
analyst = IncidentAnalyst()


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_incident(
    incident: IncidentInput,
):

    try:

        incident_data = (
            incident.model_dump(
                mode="json"
            )
        )

        prediction = (
            classifier.predict(
                incident_data
            )
        )

        return prediction

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


# ---------------------------------------------------------
# Full AI analysis endpoint
# ---------------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
)
def analyze_incident(
    incident: IncidentInput,
):

    try:

        incident_data = (
            incident.model_dump(
                mode="json"
            )
        )

        prediction = (
            classifier.predict(
                incident_data
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Classifier prediction failed.",
        ) from exc


    # -----------------------------------------------------
    # Build explanation context for the GenAI layer
    # -----------------------------------------------------

    mitre_techniques = (
        incident_data.get(
            "MitreTechniques"
        )
    )

    if not mitre_techniques:

        mitre_techniques = (
            "Not available"
        )


    incident_context = {
        "Timestamp":
            incident_data[
                "Timestamp"
            ],

        "Category":
            incident_data[
                "Category"
            ],

        "EntityType":
            incident_data[
                "EntityType"
            ],

        "EvidenceRole":
            incident_data[
                "EvidenceRole"
            ],

        "MitreTechniques":
            mitre_techniques,
    }


    # -----------------------------------------------------
    # GenAI analysis + guardrails + fallback
    # -----------------------------------------------------

    analysis = analyst.analyze(
        prediction=prediction,
        incident_context=
            incident_context,
    )


    return {
        "prediction":
            prediction,

        "route":
            analysis[
                "route"
            ],

        "reason":
            analysis[
                "reason"
            ],

        "analyst_response":
            analysis[
                "response"
            ],
    }