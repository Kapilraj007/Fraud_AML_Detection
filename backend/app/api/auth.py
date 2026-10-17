"""JWT login, refresh rotation, logout, and current-user routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.api.dependencies import (
    CurrentUser,
    DatabaseSession,
    audit_request,
    get_request_settings,
)
from backend.app.core.config import Settings
from backend.app.core.security import (
    AuthenticationError,
    decode_token,
    issue_token_pair,
    verify_password,
)
from backend.app.db.repositories import (
    RefreshSessionError,
    RefreshTokenReuseError,
    create_refresh_session,
    get_user_by_email,
    get_user_by_id,
    revoke_refresh_family,
    rotate_refresh_session,
)
from backend.app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    TokenResponse,
    UserRead,
)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    session: DatabaseSession,
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> TokenResponse:
    user = await get_user_by_email(session, payload.email)
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password.get_secret_value(), user.password_hash)
    ):
        await audit_request(
            session,
            request,
            event_type="auth.login_failed",
            resource_type="user",
            actor=user,
            details={"reason": "invalid_credentials"},
        )
        await session.commit()
        raise _invalid_credentials()

    pair = issue_token_pair(user.id, user.role, settings=settings)
    await create_refresh_session(
        session,
        user_id=user.id,
        jti=pair.refresh.jti,
        expires_at=pair.refresh.expires_at,
        family_id=pair.family_id,
        session_id=pair.refresh_session_id,
        settings=settings,
    )
    user.last_login_at = datetime.now(timezone.utc)
    await audit_request(
        session,
        request,
        event_type="auth.login_succeeded",
        resource_type="user",
        actor=user,
        resource_id=user.id,
    )
    await session.commit()
    return _token_response(pair)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshRequest,
    request: Request,
    session: DatabaseSession,
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> TokenResponse:
    try:
        decoded = decode_token(
            payload.refresh_token.get_secret_value(),
            expected_type="refresh",
            settings=settings,
        )
        user_id = UUID(decoded.subject)
    except (AuthenticationError, ValueError):
        raise _invalid_refresh() from None
    user = await get_user_by_id(session, user_id)
    if user is None or not user.is_active or user.role != decoded.role:
        raise _invalid_refresh()

    pair = issue_token_pair(
        user.id,
        user.role,
        family_id=decoded.family_id,
        settings=settings,
    )
    try:
        await rotate_refresh_session(
            session,
            current_session_id=decoded.refresh_session_id,
            presented_jti=decoded.jti,
            replacement_jti=pair.refresh.jti,
            replacement_expires_at=pair.refresh.expires_at,
            replacement_session_id=pair.refresh_session_id,
            settings=settings,
        )
    except (RefreshTokenReuseError, RefreshSessionError) as exc:
        await audit_request(
            session,
            request,
            event_type="auth.refresh_rejected",
            resource_type="refresh_session",
            actor=user,
            resource_id=decoded.refresh_session_id,
            details={"reason": exc.__class__.__name__},
        )
        # Reuse detection and expiry revocation are security state and must survive
        # the rejected request.
        await session.commit()
        raise _invalid_refresh() from None

    await audit_request(
        session,
        request,
        event_type="auth.refresh_rotated",
        resource_type="refresh_session",
        actor=user,
        resource_id=pair.refresh_session_id,
    )
    await session.commit()
    return _token_response(pair)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    payload: LogoutRequest,
    request: Request,
    session: DatabaseSession,
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> MessageResponse:
    try:
        decoded = decode_token(
            payload.refresh_token.get_secret_value(),
            expected_type="refresh",
            settings=settings,
        )
        user_id = UUID(decoded.subject)
    except (AuthenticationError, ValueError):
        raise _invalid_refresh() from None
    user = await get_user_by_id(session, user_id)
    if user is None or decoded.family_id is None:
        raise _invalid_refresh()
    await revoke_refresh_family(
        session,
        family_id=decoded.family_id,
        reason="user_logout",
    )
    await audit_request(
        session,
        request,
        event_type="auth.logout",
        resource_type="refresh_session",
        actor=user,
        resource_id=decoded.refresh_session_id,
    )
    await session.commit()
    return MessageResponse(message="logged out")


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)


def _token_response(pair: object) -> TokenResponse:
    now = datetime.now(timezone.utc)
    return TokenResponse(
        access_token=pair.access.token,
        refresh_token=pair.refresh.token,
        expires_in=max(1, int((pair.access.expires_at - now).total_seconds())),
        refresh_expires_in=max(1, int((pair.refresh.expires_at - now).total_seconds())),
    )


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="email or password is incorrect",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _invalid_refresh() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="refresh token is invalid or expired",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = ["router"]
