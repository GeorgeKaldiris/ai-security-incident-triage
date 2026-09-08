import os
from pathlib import Path
from typing import Any, Literal, Mapping

from dotenv import load_dotenv
from huggingface_hub import InferenceClient



PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_PATH,
    override=False,
)


HOSTED_MODEL_NAME = "Qwen/Qwen3-8B"

GENAI_ROUTING_THRESHOLD = 0.80


REQUIRED_SECTIONS = [
    "Assessment:",
    "Recommended action:",
    "Caveat:",
]


FORBIDDEN_PHRASES = [
    "malicious ip",
    "source ip",
    "originating from the ip",
    "block the ip",
    "block the source",
    "isolate the source",
    "confirmed attack",
    "prediction is based on the provided context",
    "prediction is based on the supplied context",
]


def build_grounded_messages(
    prediction: Mapping[str, Any],
    incident_context: Mapping[str, Any],
) -> list[dict[str, str]]:

    system_message = """
You are a cybersecurity incident triage assistant.

A separate machine learning classifier has already produced
the incident triage prediction.

Your task is to summarize the supplied result for a human analyst.

GROUNDING RULES:
- Use only facts explicitly supplied in the user message.
- Do not infer that an IP address, device, account, or other entity is malicious.
- Do not infer that an entity is the source, destination, attacker, or victim.
- Do not state that a specific attack occurred unless it is explicitly supplied.
- Do not claim that any context field caused the classifier prediction.
- The incident context shown to you is explanation context only.
  Do not claim that these fields caused, determined, or were the complete
  inputs used for the classifier prediction.
- Do not recommend blocking, isolating, disabling, deleting, or otherwise
  taking containment action when malicious behavior has not been explicitly confirmed.
- Missing information must remain unknown.
- The classifier confidence is model confidence, not proof that the prediction is correct.

Return exactly three lines:

Assessment:
Recommended action:
Caveat:
""".strip()

    user_message = f"""
Classifier prediction: {prediction["predicted_class"]}
Confidence: {prediction["confidence"]:.4f}

Incident context:
Category: {incident_context["Category"]}
Entity type: {incident_context["EntityType"]}
Evidence role: {incident_context["EvidenceRole"]}
MITRE techniques: {incident_context["MitreTechniques"]}

Explain the supplied classifier result conservatively for a human analyst.

/no_think
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def validate_analyst_response(
    response: Any,
) -> dict[str, Any]:

    if not isinstance(
        response,
        str,
    ):

        return {
            "is_valid": False,
            "missing_sections":
                REQUIRED_SECTIONS.copy(),
            "reason":
                "Response is not text.",
        }

    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if section not in response
    ]

    return {
        "is_valid":
            len(missing_sections) == 0,

        "missing_sections":
            missing_sections,

        "reason":
            (
                "Valid response."
                if not missing_sections
                else "Required sections missing."
            ),
    }


def validate_grounding(
    response: Any,
) -> dict[str, Any]:

    if not isinstance(
        response,
        str,
    ):

        return {
            "is_grounded": False,
            "violations": [
                "Response is not text."
            ],
        }

    normalized_response = (
        response.lower()
    )

    violations = [
        phrase
        for phrase in FORBIDDEN_PHRASES
        if phrase in normalized_response
    ]

    return {
        "is_grounded":
            len(violations) == 0,

        "violations":
            violations,
    }


def build_safe_fallback(
    incident_context: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> str:

    mitre_value = (
        incident_context[
            "MitreTechniques"
        ]
    )

    if mitre_value == "Not available":

        mitre_text = (
            "No MITRE technique information "
            "was supplied."
        )

    else:

        mitre_text = (
            "MITRE technique information "
            "is available in the incident context."
        )

    return (
        f"Assessment: The classifier predicted "
        f"{prediction['predicted_class']} with a model "
        f"confidence score of {prediction['confidence']:.4f}. "
        f"The supplied explanation context includes category "
        f"{incident_context['Category']}, entity type "
        f"{incident_context['EntityType']}, and evidence role "
        f"{incident_context['EvidenceRole']}.\n"

        f"Recommended action: Review the available incident "
        f"evidence and associated context before making a final "
        f"analyst disposition.\n"

        f"Caveat: {mitre_text} The explanation context does not "
        f"establish why the classifier produced its prediction."
    )


class IncidentAnalyst:

    def __init__(
        self,
        model_name: str = HOSTED_MODEL_NAME,
        confidence_threshold: float =
            GENAI_ROUTING_THRESHOLD,
    ):

        self.model_name = model_name

        self.confidence_threshold = (
            confidence_threshold
        )

        self.hf_token = os.getenv(
            "HF_TOKEN"
        )

        if self.hf_token:

            self.client = InferenceClient(
                provider="auto",
                api_key=self.hf_token,
            )

        else:

            self.client = None


    def _generate_response(
        self,
        prediction: Mapping[str, Any],
        incident_context: Mapping[str, Any],
    ) -> str | None:

        if self.client is None:

            raise RuntimeError(
                "Hugging Face inference is not configured."
            )

        messages = build_grounded_messages(
            prediction,
            incident_context,
        )

        completion = (
            self.client
            .chat
            .completions
            .create(
                model=self.model_name,
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                top_p=0.8,
            )
        )

        return (
            completion
            .choices[0]
            .message
            .content
        )


    def analyze(
        self,
        prediction: Mapping[str, Any],
        incident_context: Mapping[str, Any],
    ) -> dict[str, Any]:

        confidence = float(
            prediction[
                "confidence"
            ]
        )

        # Operational routing threshold only.
        # It is not a calibrated probability.
        if (
            confidence
            < self.confidence_threshold
        ):

            return {
                "route": "safe_fallback",
                "reason":
                    "low_classifier_confidence",
                "response":
                    build_safe_fallback(
                        incident_context,
                        prediction,
                    ),
            }

        if self.client is None:

            return {
                "route": "safe_fallback",
                "reason":
                    "llm_not_configured",
                "response":
                    build_safe_fallback(
                        incident_context,
                        prediction,
                    ),
            }

        try:

            llm_response = (
                self._generate_response(
                    prediction,
                    incident_context,
                )
            )

        except Exception as exc:

            return {
                "route": "safe_fallback",
                "reason":
                    "llm_inference_error",
                "response":
                    build_safe_fallback(
                        incident_context,
                        prediction,
                    ),
                "error_type":
                    type(exc).__name__,
            }

        structure_result = (
            validate_analyst_response(
                llm_response
            )
        )

        if not structure_result[
            "is_valid"
        ]:

            return {
                "route": "safe_fallback",
                "reason":
                    "invalid_llm_structure",
                "response":
                    build_safe_fallback(
                        incident_context,
                        prediction,
                    ),
            }

        grounding_result = (
            validate_grounding(
                llm_response
            )
        )

        if not grounding_result[
            "is_grounded"
        ]:

            return {
                "route": "safe_fallback",
                "reason":
                    "grounding_violation",
                "response":
                    build_safe_fallback(
                        incident_context,
                        prediction,
                    ),
            }

        return {
            "route": "llm_response",
            "reason":
                "validated_llm_response",
            "response":
                llm_response,
        }


