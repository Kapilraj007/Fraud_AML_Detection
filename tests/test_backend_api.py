from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import numpy as np
from sqlalchemy import func, select

from backend.app.core.config import Settings
from backend.app.core.security import hash_password
from backend.app.db.enums import CaseStatus, DriftStatus, ModelStatus, UserRole
from backend.app.db.models import (
    AuditEvent,
    Case,
    CaseTransaction,
    DriftSnapshot,
    Explanation,
    ModelVersion,
    Prediction,
    Transaction,
)
from backend.app.db.repositories import create_user
from backend.app.db.session import (
    create_database_engine,
    create_schema_for_tests,
    create_session_factory,
)
from backend.app.main import create_app
from backend.app.ml.artifacts import LoadedModelBundle
from backend.app.ml.model import LightGBMModel, PlattCalibrator
from backend.app.services.inference import InferenceService


class _Booster:
    best_iteration = 1

    def predict(self, matrix, *, raw_score=False, pred_contrib=False, num_iteration=None):
        values = np.asarray(matrix, dtype=float)
        margin = values[:, 0] - values[:, 1]
        if pred_contrib:
            return np.column_stack((values[:, 0], -values[:, 1], np.zeros(len(values))))
        if raw_score:
            return margin
        return 1.0 / (1.0 + np.exp(-margin))


def _inference() -> InferenceService:
    model = LightGBMModel(
        _Booster(),
        ("amount_log1p", "source_txn_count_1h"),
        PlattCalibrator(),
    )
    return InferenceService(
        LoadedModelBundle(
            model=model,
            manifest={"bundle_id": "api-test-v1", "feature_schema_version": 1},
            feature_schema={"feature_names": list(model.feature_names)},
            metrics={},
            training={},
        )
    )


def test_openapi_and_complete_backend_workflow() -> None:
    asyncio.run(_complete_backend_workflow())


async def _complete_backend_workflow() -> None:
    settings = Settings(
        environment="test",
        database_url="sqlite+aiosqlite:///:memory:",
        jwt_secret_key="api-test-secret-that-is-longer-than-thirty-two-characters",
        model_artifact_dir="missing-test-model-directory",
    )
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    await create_schema_for_tests(engine)
    seeded = await _seed(factory)
    app = create_app(
        settings=settings,
        session_factory=factory,
        inference_service=_inference(),
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            schema = app.openapi()
            required_paths = {
                "/health/live",
                "/health/ready",
                "/api/v1/auth/token",
                "/api/v1/auth/refresh",
                "/api/v1/auth/logout",
                "/api/v1/auth/me",
                "/api/v1/transactions",
                "/api/v1/transactions/{transaction_id}",
                "/api/v1/predictions/score",
                "/api/v1/predictions/batch-score",
                "/api/v1/cases",
                "/api/v1/cases/materialize",
                "/api/v1/cases/{case_id}",
                "/api/v1/cases/{case_id}/claim",
                "/api/v1/cases/{case_id}/start-review",
                "/api/v1/cases/{case_id}/disposition",
                "/api/v1/monitoring/status",
                "/api/v1/monitoring/history",
                "/api/v1/monitoring/retraining-requests",
                "/api/v1/models/active",
                "/api/v1/models/{model_version_id}/promote",
                "/api/v1/audit/events",
            }
            assert required_paths <= set(schema["paths"])
            assert "label" not in schema["components"]["schemas"]["PredictionRequest"]["properties"]

            live = await client.get("/health/live", headers={"X-Request-ID": "api-test-1"})
            assert live.status_code == 200
            assert live.headers["X-Request-ID"] == "api-test-1"
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.get("/api/v1/auth/me")).status_code == 401

            analyst_login = await _login(client, "analyst@example.test")
            senior_login = await _login(client, "senior@example.test")
            admin_login = await _login(client, "admin@example.test")
            analyst_headers = _auth(analyst_login["access_token"])
            senior_headers = _auth(senior_login["access_token"])
            admin_headers = _auth(admin_login["access_token"])
            me = await client.get("/api/v1/auth/me", headers=analyst_headers)
            assert me.status_code == 200
            assert me.json()["role"] == "fraud_analyst"

            rotated = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": analyst_login["refresh_token"]},
            )
            assert rotated.status_code == 200
            replay = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": analyst_login["refresh_token"]},
            )
            assert replay.status_code == 401
            family_revoked = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": rotated.json()["refresh_token"]},
            )
            assert family_revoked.status_code == 401

            invalid_truth = _transaction_payload("truth-event") | {"label": 1}
            assert (
                await client.post(
                    "/api/v1/transactions", json=invalid_truth, headers=analyst_headers
                )
            ).status_code == 422
            nested_truth = _transaction_payload("nested-truth-event")
            nested_truth["raw_attributes"] = {"ground_truth": 1}
            assert (
                await client.post(
                    "/api/v1/transactions", json=nested_truth, headers=analyst_headers
                )
            ).status_code == 422

            created = await client.post(
                "/api/v1/transactions",
                json=_transaction_payload("score-event-1"),
                headers=analyst_headers,
            )
            assert created.status_code == 201, created.text
            created_body = created.json()
            assert created_body["masked_source_account"].endswith("7890")
            assert "source_account_token" not in created_body
            assert (
                await client.get(
                    f"/api/v1/transactions/{created_body['id']}", headers=analyst_headers
                )
            ).status_code == 200
            assert (await client.get("/api/v1/transactions", headers=analyst_headers)).status_code == 200

            unavailable_app = create_app(
                settings=settings,
                session_factory=factory,
                inference_service=InferenceService(),
            )
            unavailable_transport = httpx.ASGITransport(app=unavailable_app)
            async with httpx.AsyncClient(
                transport=unavailable_transport, base_url="http://test"
            ) as unavailable_client:
                unavailable = await unavailable_client.post(
                    "/api/v1/predictions/score",
                    json=_prediction_payload(created_body["id"]),
                    headers=analyst_headers,
                )
                assert unavailable.status_code == 503

            ground_truth = _prediction_payload(created_body["id"])
            ground_truth["features"]["fraud_label"] = 1
            rejected = await client.post(
                "/api/v1/predictions/score", json=ground_truth, headers=analyst_headers
            )
            assert rejected.status_code == 422
            scored = await client.post(
                "/api/v1/predictions/score",
                json=_prediction_payload(created_body["id"]),
                headers=analyst_headers,
            )
            assert scored.status_code == 201, scored.text
            assert 0 <= scored.json()["combined_risk"] <= 1
            duplicate = await client.post(
                "/api/v1/predictions/score",
                json=_prediction_payload(created_body["id"]),
                headers=analyst_headers,
            )
            assert duplicate.status_code == 409

            batch_ids: list[str] = []
            for number in (2, 3):
                response = await client.post(
                    "/api/v1/transactions",
                    json=_transaction_payload(f"score-event-{number}"),
                    headers=analyst_headers,
                )
                assert response.status_code == 201
                batch_ids.append(response.json()["id"])
            batch = await client.post(
                "/api/v1/predictions/batch-score",
                json={"rows": [_prediction_payload(item) for item in batch_ids]},
                headers=analyst_headers,
            )
            assert batch.status_code == 201, batch.text
            assert len(batch.json()) == 2
            oversized = await client.post(
                "/api/v1/predictions/batch-score",
                json={"rows": [_prediction_payload(batch_ids[0])] * 257},
                headers=analyst_headers,
            )
            assert oversized.status_code == 422

            queue_payload = {
                "alert_hour": datetime.now(timezone.utc).replace(
                    minute=0, second=0, microsecond=0
                ).isoformat(),
                "capacity": 2,
            }
            assert (
                await client.post(
                    "/api/v1/cases/materialize",
                    json=queue_payload,
                    headers=analyst_headers,
                )
            ).status_code == 403
            materialized = await client.post(
                "/api/v1/cases/materialize",
                json=queue_payload,
                headers=senior_headers,
            )
            assert materialized.status_code == 200, materialized.text
            assert len(materialized.json()) == 2
            retry = await client.post(
                "/api/v1/cases/materialize",
                json=queue_payload,
                headers=senior_headers,
            )
            assert retry.status_code == 200
            assert [item["id"] for item in retry.json()] == [
                item["id"] for item in materialized.json()
            ]
            materialized_detail = await client.get(
                f"/api/v1/cases/{materialized.json()[0]['id']}",
                headers=analyst_headers,
            )
            assert materialized_detail.status_code == 200
            assert len(materialized_detail.json()["predictions"]) == 1
            assert materialized_detail.json()["predictions"][0][
                "combined_score"
            ] == materialized_detail.json()["risk_score"]

            cases = await client.get("/api/v1/cases", headers=analyst_headers)
            assert cases.status_code == 200
            assert cases.json()[0]["review_id"] == "review-api-1"
            case_id = str(seeded["case_id"])
            detail = await client.get(f"/api/v1/cases/{case_id}", headers=analyst_headers)
            assert detail.status_code == 200
            assert detail.json()["transactions"][0]["event_id"] == "seed-event"
            assert detail.json()["predictions"][0]["combined_score"] == 0.92
            assert detail.json()["predictions"][0]["model_version_id"] == str(
                seeded["active_id"]
            )
            assert detail.json()["predictions"][0]["explanation"]["top_features"][0][
                "feature"
            ] == "amount_log1p"
            claimed = await client.post(
                f"/api/v1/cases/{case_id}/claim",
                json={"expected_status": "open"},
                headers=analyst_headers,
            )
            assert claimed.status_code == 200
            started = await client.post(
                f"/api/v1/cases/{case_id}/start-review",
                json={"expected_status": "claimed"},
                headers=analyst_headers,
            )
            assert started.status_code == 200
            disposition = await client.post(
                f"/api/v1/cases/{case_id}/disposition",
                json={
                    "decision": "escalate",
                    "confidence": 4,
                    "rationale": "Network behavior warrants senior investigation.",
                    "model_agreement": True,
                    "review_duration_seconds": 120,
                },
                headers=analyst_headers,
            )
            assert disposition.status_code == 201, disposition.text
            assert disposition.json()["confidence"] == 0.8

            status_response = await client.get(
                "/api/v1/monitoring/status", headers=analyst_headers
            )
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "drift_detected"
            assert (
                await client.get("/api/v1/monitoring/history", headers=analyst_headers)
            ).status_code == 200
            retraining_payload = {
                "drift_snapshot_id": str(seeded["snapshot_id"]),
                "reason": "Material feature and prediction drift was confirmed.",
            }
            forbidden_retraining = await client.post(
                "/api/v1/monitoring/retraining-requests",
                json=retraining_payload,
                headers=analyst_headers,
            )
            assert forbidden_retraining.status_code == 403
            retraining = await client.post(
                "/api/v1/monitoring/retraining-requests",
                json=retraining_payload,
                headers=senior_headers,
            )
            assert retraining.status_code == 201, retraining.text

            active = await client.get("/api/v1/models/active", headers=analyst_headers)
            assert active.status_code == 200
            assert active.json()["version"] == "api-test-v1"
            promotion_payload = {
                "expected_current_version": "api-test-v1",
                "reason": "Candidate passed the chronological acceptance gates.",
            }
            forbidden_promotion = await client.post(
                f"/api/v1/models/{seeded['candidate_id']}/promote",
                json=promotion_payload,
                headers=analyst_headers,
            )
            assert forbidden_promotion.status_code == 403
            promoted = await client.post(
                f"/api/v1/models/{seeded['candidate_id']}/promote",
                json=promotion_payload,
                headers=admin_headers,
            )
            assert promoted.status_code == 200, promoted.text
            assert promoted.json()["status"] == "active"

            logout = await client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": admin_login["refresh_token"]},
            )
            assert logout.status_code == 200
            assert (
                await client.get("/api/v1/audit/events", headers=analyst_headers)
            ).status_code == 403
            audit_response = await client.get("/api/v1/audit/events", headers=admin_headers)
            assert audit_response.status_code == 200
            assert all("safe_details" in event for event in audit_response.json())

        async with factory() as session:
            audit_count = await session.scalar(select(func.count()).select_from(AuditEvent))
            audit_types = set((await session.execute(select(AuditEvent.event_type))).scalars())
            assert audit_count >= 12
            assert {
                "auth.login_succeeded",
                "prediction.scored",
                "case.disposition_recorded",
                "monitoring.retraining_requested",
                "model.promoted",
            } <= audit_types
    finally:
        await engine.dispose()


async def _seed(factory) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    password_hash = hash_password("Correct-Horse-123")
    async with factory() as session:
        analyst = await create_user(
            session,
            email="analyst@example.test",
            password_hash=password_hash,
            role=UserRole.FRAUD_ANALYST,
            full_name="API Analyst",
        )
        senior = await create_user(
            session,
            email="senior@example.test",
            password_hash=password_hash,
            role=UserRole.SENIOR_INVESTIGATOR,
            full_name="Senior Investigator",
        )
        admin = await create_user(
            session,
            email="admin@example.test",
            password_hash=password_hash,
            role=UserRole.ADMIN,
            full_name="API Admin",
        )
        active = ModelVersion(
            name="aml-primary",
            version="api-test-v1",
            algorithm="LightGBM",
            status=ModelStatus.ACTIVE,
            artifact_uri="artifacts/models/api-test-v1",
            artifact_sha256="a" * 64,
            feature_schema={"feature_names": ["amount_log1p", "source_txn_count_1h"]},
            training_window_start=now - timedelta(days=30),
            training_window_end=now - timedelta(days=2),
            metrics={"pr_auc": 0.5},
            decision_threshold=0.6,
            ensemble_weights={"supervised": 1.0},
            created_by_id=admin.id,
            activated_at=now - timedelta(days=1),
        )
        candidate = ModelVersion(
            name="aml-primary",
            version="api-test-v2",
            algorithm="LightGBM",
            status=ModelStatus.CANDIDATE,
            artifact_uri="artifacts/models/api-test-v2",
            artifact_sha256="b" * 64,
            feature_schema={"feature_names": ["amount_log1p", "source_txn_count_1h"]},
            training_window_start=now - timedelta(days=29),
            training_window_end=now - timedelta(days=1),
            metrics={"pr_auc": 0.6},
            decision_threshold=0.6,
            ensemble_weights={"supervised": 1.0},
            created_by_id=admin.id,
        )
        seed_transaction = Transaction(
            event_id="seed-event",
            dataset="test",
            event_time=now - timedelta(hours=1),
            masked_source_account="****1111",
            masked_destination_account="****2222",
            transaction_type="TRANSFER",
            currency="USD",
            amount=Decimal("2500.00"),
        )
        case = Case(
            review_id="review-api-1",
            status=CaseStatus.OPEN,
            queue_rank=1,
            alert_hour=now - timedelta(hours=1),
            risk_score=0.92,
            study_arm="reliability_gated_case_context",
        )
        session.add_all([active, candidate, seed_transaction, case])
        await session.flush()
        prediction = Prediction(
            transaction_id=seed_transaction.id,
            model_version_id=active.id,
            supervised_score=0.9,
            anomaly_score=0.75,
            combined_score=0.92,
            decision_threshold=0.6,
            predicted_positive=True,
            uncertainty=0.16,
            reliability_score=1.0,
            reliability_status="reliable",
            explanation_gate_reason=None,
            drift_status=DriftStatus.STABLE,
            feature_schema_hash="c" * 64,
            created_at=now - timedelta(minutes=5),
        )
        prediction.explanation = Explanation(
            top_features=[
                {"feature": "amount_log1p", "value": 7.82, "attribution": 0.41}
            ],
            narrative="Elevated amount and recent velocity increased the risk score.",
            is_withheld=False,
            gate_reason=None,
        )
        session.add(prediction)
        await session.flush()
        session.add(
            CaseTransaction(
                case_id=case.id,
                transaction_id=seed_transaction.id,
                prediction_id=prediction.id,
                position=0,
            )
        )
        # A later score for a candidate model must never replace the evidence
        # frozen into the already-materialized case.
        session.add(
            Prediction(
                transaction_id=seed_transaction.id,
                model_version_id=candidate.id,
                supervised_score=0.11,
                anomaly_score=0.21,
                combined_score=0.12,
                decision_threshold=0.6,
                predicted_positive=False,
                uncertainty=0.7,
                reliability_score=0.3,
                reliability_status="unreliable",
                explanation_gate_reason="candidate model score",
                drift_status=DriftStatus.WATCH,
                feature_schema_hash="d" * 64,
                created_at=now,
            )
        )
        snapshot = DriftSnapshot(
            model_version_id=active.id,
            window_start=now - timedelta(hours=2),
            window_end=now - timedelta(hours=1),
            status=DriftStatus.DRIFT_DETECTED,
            feature_metrics={"amount_log1p": {"ks": 0.4}},
            prediction_metrics={"score_mean": 0.7},
            adwin_detected=True,
        )
        session.add(snapshot)
        await session.commit()
        return {
            "case_id": case.id,
            "snapshot_id": snapshot.id,
            "active_id": active.id,
            "candidate_id": candidate.id,
            "analyst_id": analyst.id,
            "senior_id": senior.id,
        }


async def _login(client: httpx.AsyncClient, email: str) -> dict[str, object]:
    response = await client.post(
        "/api/v1/auth/token",
        json={"email": email, "password": "Correct-Horse-123"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _auth(access_token: object) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _transaction_payload(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "dataset": "online-test",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "source_account_token": "source-account-7890",
        "destination_account_token": "destination-account-4321",
        "amount": "123.45",
        "currency": "usd",
        "transaction_type": "TRANSFER",
        "raw_attributes": {"channel": "web"},
    }


def _prediction_payload(transaction_id: str) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "features": {"amount_log1p": 2.0, "source_txn_count_1h": 0.5},
    }
