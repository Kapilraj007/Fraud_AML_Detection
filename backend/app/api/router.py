"""Top-level API router composition."""

from fastapi import APIRouter

from backend.app.api import audit, auth, cases, health, models, monitoring, predictions, transactions


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(transactions.router)
api_router.include_router(predictions.router)
api_router.include_router(cases.router)
api_router.include_router(monitoring.router)
api_router.include_router(models.router)
api_router.include_router(audit.router)

health_router = health.router


__all__ = ["api_router", "health_router"]
