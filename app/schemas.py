from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


CategoricalValue = (
    str
    | int
    | float
    | None
)


class IncidentInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    DetectorId: CategoricalValue = None
    AlertTitle: CategoricalValue = None

    Category: str
    EntityType: str
    EvidenceRole: str

    RegistryKey: CategoricalValue = None
    RegistryValueName: CategoricalValue = None
    RegistryValueData: CategoricalValue = None

    ApplicationId: CategoricalValue = None
    ApplicationName: CategoricalValue = None
    OAuthApplicationId: CategoricalValue = None
    ResourceIdName: CategoricalValue = None

    OSFamily: CategoricalValue = None
    OSVersion: CategoricalValue = None

    CountryCode: CategoricalValue = None
    State: CategoricalValue = None
    City: CategoricalValue = None

    Timestamp: datetime

    MitreTechniques: str | None = None


class PredictionResponse(BaseModel):

    predicted_class: Literal[
        "BenignPositive",
        "FalsePositive",
        "TruePositive",
    ]

    confidence: float

    probabilities: dict[
        str,
        float
    ]


class AnalysisResponse(BaseModel):

    prediction: PredictionResponse

    route: Literal[
        "llm_response",
        "safe_fallback",
    ]

    reason: str

    analyst_response: str