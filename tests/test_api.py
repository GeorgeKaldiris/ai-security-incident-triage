from fastapi.testclient import TestClient

from app.main import (
    app,
    analyst,
    classifier,
)


client = TestClient(app)


SAMPLE_INCIDENT = {
    "DetectorId": 7,
    "AlertTitle": 6,
    "Category": "InitialAccess",
    "EntityType": "Ip",
    "EvidenceRole": "Related",

    "RegistryKey": None,
    "RegistryValueName": None,
    "RegistryValueData": None,

    "ApplicationId": None,
    "ApplicationName": None,
    "OAuthApplicationId": None,
    "ResourceIdName": None,

    "OSFamily": None,
    "OSVersion": None,

    "CountryCode": None,
    "State": None,
    "City": None,

    "Timestamp":
        "2024-06-04T06:05:15.000Z",

    "MitreTechniques": None,
}


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok"
    }


def test_predict():

    response = client.post(
        "/predict",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["predicted_class"]
        == "TruePositive"
    )

    assert (
        0.0
        <= data["confidence"]
        <= 1.0
    )

    assert set(
        data["probabilities"].keys()
    ) == {
        "BenignPositive",
        "FalsePositive",
        "TruePositive",
    }

    probability_sum = sum(
        data["probabilities"].values()
    )

    assert abs(
        probability_sum - 1.0
    ) < 1e-5


def test_missing_required_field():

    invalid_incident = (
        SAMPLE_INCIDENT.copy()
    )

    invalid_incident.pop(
        "Category"
    )

    response = client.post(
        "/predict",
        json=invalid_incident,
    )

    assert response.status_code == 422


def test_unknown_extra_field_is_rejected():

    invalid_incident = {
        **SAMPLE_INCIDENT,
        "UnknownField": "something",
    }

    response = client.post(
        "/predict",
        json=invalid_incident,
    )

    assert response.status_code == 422


def test_analyze_valid_llm_response(
    monkeypatch,
):

    valid_llm_response = (
        "Assessment: The classifier returned "
        "TruePositive with high model confidence.\n"
        "Recommended action: Review the available "
        "incident evidence and related context.\n"
        "Caveat: The classifier confidence does not "
        "prove that the prediction is correct."
    )

    def fake_generate_response(
        prediction,
        incident_context,
    ):

        return valid_llm_response


    # Pretend that Hugging Face is configured.
    monkeypatch.setattr(
        analyst,
        "client",
        object(),
    )

    # Replace real hosted inference with fake output.
    monkeypatch.setattr(
        analyst,
        "_generate_response",
        fake_generate_response,
    )


    response = client.post(
        "/analyze",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["route"]
        == "llm_response"
    )

    assert (
        data["reason"]
        == "validated_llm_response"
    )

    assert (
        data["analyst_response"]
        == valid_llm_response
    )


def test_analyze_grounding_violation(
    monkeypatch,
):

    unsafe_llm_response = (
        "Assessment: This is a malicious IP.\n"
        "Recommended action: Review the available "
        "incident evidence.\n"
        "Caveat: Additional evidence may be required."
    )

    def fake_generate_response(
        prediction,
        incident_context,
    ):

        return unsafe_llm_response


    monkeypatch.setattr(
        analyst,
        "client",
        object(),
    )

    monkeypatch.setattr(
        analyst,
        "_generate_response",
        fake_generate_response,
    )


    response = client.post(
        "/analyze",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["route"]
        == "safe_fallback"
    )

    assert (
        data["reason"]
        == "grounding_violation"
    )

    assert (
        "malicious IP"
        not in data["analyst_response"]
    )


def test_analyze_invalid_llm_structure(
    monkeypatch,
):

    invalid_llm_response = (
        "The classifier thinks this incident "
        "may require further review."
    )

    def fake_generate_response(
        prediction,
        incident_context,
    ):

        return invalid_llm_response


    monkeypatch.setattr(
        analyst,
        "client",
        object(),
    )

    monkeypatch.setattr(
        analyst,
        "_generate_response",
        fake_generate_response,
    )


    response = client.post(
        "/analyze",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["route"]
        == "safe_fallback"
    )

    assert (
        data["reason"]
        == "invalid_llm_structure"
    )

    assert (
        data["analyst_response"]
        .startswith(
            "Assessment:"
        )
    )


def test_analyze_llm_inference_error(
    monkeypatch,
):

    def fake_generate_response(
        prediction,
        incident_context,
    ):

        raise RuntimeError(
            "Simulated provider failure"
        )


    monkeypatch.setattr(
        analyst,
        "client",
        object(),
    )

    monkeypatch.setattr(
        analyst,
        "_generate_response",
        fake_generate_response,
    )


    response = client.post(
        "/analyze",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["route"]
        == "safe_fallback"
    )

    assert (
        data["reason"]
        == "llm_inference_error"
    )



def test_analyze_low_confidence_skips_llm(
    monkeypatch,
):

    low_confidence_prediction = {
        "predicted_class":
            "FalsePositive",

        "confidence":
            0.55,

        "probabilities": {
            "BenignPositive":
                0.20,

            "FalsePositive":
                0.55,

            "TruePositive":
                0.25,
        },
    }


    def fake_classifier_predict(
        incident,
    ):

        return low_confidence_prediction


    def forbidden_llm_call(
        prediction,
        incident_context,
    ):

        raise AssertionError(
            "LLM should not be called "
            "for low-confidence predictions."
        )


    monkeypatch.setattr(
        classifier,
        "predict",
        fake_classifier_predict,
    )

    monkeypatch.setattr(
        analyst,
        "client",
        object(),
    )

    monkeypatch.setattr(
        analyst,
        "_generate_response",
        forbidden_llm_call,
    )


    response = client.post(
        "/analyze",
        json=SAMPLE_INCIDENT,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["route"]
        == "safe_fallback"
    )

    assert (
        data["reason"]
        == "low_classifier_confidence"
    )