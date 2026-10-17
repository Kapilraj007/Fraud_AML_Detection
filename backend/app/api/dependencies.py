"""Shared FastAPI dependencies and request-safe audit metadata."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
import hashlib
import hmac
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings
from backend.app.core.security import AuthenticationError, decode_token
from backend.app.db.enums import UserRole
from backend.app.db.models import User
from backend.app.db.repositories import get_user_by_id, record_audit_event
from backend.app.services.inference import InferenceService, ModelUnavailableError


bearer_scheme = HTTPBearer(auto_error=False)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.session_factory
    async with factory() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> User:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise _authentication_error("bearer access token is required")
    try:
        decoded = decode_token(
            credentials.credentials,
            expected_type="access",
            settings=settings,
        )
        user_id = UUID(decoded.subject)
    except (AuthenticationError, ValueError):
        raise _authentication_error("access token is invalid or expired") from None
    user = await get_user_by_id(session, user_id)
    if user is None or not user.is_active or user.role != decoded.role:
        raise _authentication_error("access token is no longer authorized")
    return user


def require_roles(*roles: UserRole) -> Callable[..., object]:
    allowed = frozenset(roles)

    async def dependency(
        user: Annotated[User, Depends(get_current_user)],
    ) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="this role is not permitted to perform the operation",
            )
        return user

    return dependency


def get_inference_service(request: Request) -> InferenceService:
    service = getattr(request.app.state, "inference_service", None)
    if service is None or not service.available:
        reason = getattr(request.app.state, "model_load_error", None)
        detail = "no verified active model is loaded"
        if reason:
            detail = f"{detail}: {reason}"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return service


async def audit_request(
    session: AsyncSession,
    request: Request,
    *,
    event_type: str,
    resource_type: str,
    actor: User | None = None,
    resource_id: str | UUID | None = None,
    details: dict[str, object] | None = None,
) -> None:
    """Record only allow-listed operational metadata, never payload contents."""
    await record_audit_event(
        session,
        event_type=event_type,
        resource_type=resource_type,
        actor_user_id=None if actor is None else actor.id,
        resource_id=resource_id,
        request_id=getattr(request.state, "request_id", None),
        ip_hash=_client_hash(request),
        safe_details=dict(details or {}),
    )


def _client_hash(request: Request) -> str | None:
    if request.client is None or not request.client.host:
        return None
    settings: Settings = request.app.state.settings
    return hmac.new(
        settings.jwt_secret_key.get_secret_value().encode("utf-8"),
        request.client.host.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _authentication_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
Inference = Annotated[InferenceService, Depends(get_inference_service)]


__all__ = [
    "CurrentUser",
    "DatabaseSession",
    "Inference",
    "audit_request",
    "get_current_user",
    "get_db_session",
    "get_inference_service",
    "get_request_settings",
    "require_roles",
]
