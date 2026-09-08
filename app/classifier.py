from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODELS_DIR
    / "pytorch_incident_triage.pt"
)

PREPROCESSOR_PATH = (
    MODELS_DIR
    / "pytorch_preprocessing.joblib"
)


# ---------------------------------------------------------
# PyTorch model architecture
# ---------------------------------------------------------

class IncidentTriageNetwork(nn.Module):

    def __init__(
        self,
        categorical_features,
        category_cardinalities,
        embedding_dims,
        num_numeric_features,
        num_classes=3,
    ):
        super().__init__()

        self.categorical_features = (
            categorical_features
        )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=
                        category_cardinalities[column],
                    embedding_dim=
                        embedding_dims[column],
                )
                for column
                in categorical_features
            ]
        )

        total_embedding_dim = sum(
            embedding_dims[column]
            for column
            in categorical_features
        )

        total_input_dim = (
            total_embedding_dim
            + num_numeric_features
        )

        self.network = nn.Sequential(
            nn.Linear(
                total_input_dim,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(
                128,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(
                64,
                num_classes
            ),
        )

    def forward(
        self,
        categorical_data,
        numerical_data,
    ):

        embedded_features = []

        for i, embedding_layer in enumerate(
            self.embeddings
        ):

            embedded = embedding_layer(
                categorical_data[:, i]
            )

            embedded_features.append(
                embedded
            )

        categorical_embeddings = torch.cat(
            embedded_features,
            dim=1,
        )

        combined_features = torch.cat(
            [
                categorical_embeddings,
                numerical_data,
            ],
            dim=1,
        )

        logits = self.network(
            combined_features
        )

        return logits


class IncidentClassifier:

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        device: torch.device | None = None,
    ):

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.artifacts = joblib.load(
            preprocessor_path
        )

        self.categorical_features = (
            self.artifacts[
                "categorical_features"
            ]
        )

        self.numeric_features = (
            self.artifacts[
                "numeric_features"
            ]
        )

        self.category_maps = (
            self.artifacts[
                "category_maps"
            ]
        )

        self.category_cardinalities = (
            self.artifacts[
                "category_cardinalities"
            ]
        )

        self.embedding_dims = (
            self.artifacts[
                "embedding_dims"
            ]
        )

        self.numeric_medians = (
            self.artifacts[
                "numeric_medians"
            ]
        )

        self.scaler = (
            self.artifacts[
                "scaler"
            ]
        )

        self.label_to_index = (
            self.artifacts[
                "label_to_index"
            ]
        )

        self.index_to_label = (
            self.artifacts[
                "index_to_label"
            ]
        )

        self.unknown_index = (
            self.artifacts[
                "unknown_index"
            ]
        )

        self.missing_token = (
            self.artifacts[
                "missing_token"
            ]
        )

        self.model = IncidentTriageNetwork(
            categorical_features=
                self.categorical_features,

            category_cardinalities=
                self.category_cardinalities,

            embedding_dims=
                self.embedding_dims,

            num_numeric_features=
                len(
                    self.numeric_features
                ),

            num_classes=
                len(
                    self.index_to_label
                ),
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


    def _prepare_input(
        self,
        incident: Mapping[str, Any],
    ):

        categorical_indices = []

        for column in (
            self.categorical_features
        ):

            value = incident.get(
                column
            )

            if pd.isna(value):

                value = (
                    self.missing_token
                )

            else:

                value = str(
                    value
                )

            category_index = (
                self.category_maps[
                    column
                ].get(
                    value,
                    self.unknown_index,
                )
            )

            categorical_indices.append(
                category_index
            )

        # ---------------------------------------------
        # Timestamp-derived numerical features
        # ---------------------------------------------

        timestamp = pd.to_datetime(
            incident.get(
                "Timestamp"
            ),
            errors="coerce",
            utc=True,
        )

        if pd.isna(timestamp):

            hour = np.nan
            day_of_week = np.nan
            is_weekend = np.nan

        else:

            hour = timestamp.hour

            day_of_week = (
                timestamp.dayofweek
            )

            is_weekend = int(
                day_of_week >= 5
            )

        # ---------------------------------------------
        # MITRE availability indicator
        # ---------------------------------------------

        mitre_techniques = (
            incident.get(
                "MitreTechniques"
            )
        )

        has_mitre_technique = int(
            pd.notna(
                mitre_techniques
            )
        )

        numerical_values = {
            "Hour":
                hour,

            "DayOfWeek":
                day_of_week,

            "IsWeekend":
                is_weekend,

            "HasMitreTechnique":
                has_mitre_technique,
        }

        numerical_df = pd.DataFrame(
            [numerical_values],
            columns=self.numeric_features,
        ).astype(float)

        numerical_df = (
            numerical_df
            .fillna(
                self.numeric_medians
            )
        )

        numerical_scaled = (
            self.scaler
            .transform(
                numerical_df
            )
            .astype(
                np.float32
            )
        )

        categorical_tensor = (
            torch.tensor(
                [categorical_indices],
                dtype=torch.long,
                device=self.device,
            )
        )

        numerical_tensor = (
            torch.tensor(
                numerical_scaled,
                dtype=torch.float32,
                device=self.device,
            )
        )

        return (
            categorical_tensor,
            numerical_tensor,
        )


    def predict(
        self,
        incident: Mapping[str, Any],
    ) -> dict[str, Any]:

        (
            categorical_tensor,
            numerical_tensor,
        ) = self._prepare_input(
            incident
        )

        with torch.no_grad():

            logits = self.model(
                categorical_tensor,
                numerical_tensor,
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1,
                )
            )

        predicted_index = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_index
            ].item()
        )

        class_probabilities = {
            self.index_to_label[index]:
                float(
                    probabilities[
                        0,
                        index
                    ].item()
                )

            for index in range(
                len(
                    self.index_to_label
                )
            )
        }

        return {
            "predicted_class":
                self.index_to_label[
                    predicted_index
                ],

            "confidence":
                confidence,

            "probabilities":
                class_probabilities,
        }