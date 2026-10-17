"""Create the initial administrator in the configured Neon database.

The password is read from ``ADMIN_PASSWORD`` or an interactive hidden prompt;
it is never accepted as a command-line argument and never printed.
"""

from __future__ import annotations

import argparse
import asyncio
from getpass import getpass
import os
from pathlib import Path
import sys
from typing import Sequence

from sqlalchemy.exc import IntegrityError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.security import hash_password
from backend.app.db.enums import UserRole
from backend.app.db.repositories import create_user, get_user_by_email
from backend.app.db.session import session_scope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Administrator email address")
    parser.add_argument("--full-name", help="Display name")
    return parser


def _password() -> str:
    from_environment = os.environ.get("ADMIN_PASSWORD")
    if from_environment is not None:
        return from_environment
    first = getpass("Administrator password: ")
    second = getpass("Confirm password: ")
    if first != second:
        raise ValueError("password confirmation does not match")
    return first


async def seed_admin(*, email: str, password: str, full_name: str | None) -> bool:
    """Create an active admin; return ``False`` when the email already exists."""
    async with session_scope() as session:
        existing = await get_user_by_email(session, email)
        if existing is not None:
            return False
        await create_user(
            session,
            email=email,
            password_hash=hash_password(password),
            full_name=full_name,
            role=UserRole.ADMIN,
            is_active=True,
        )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        created = asyncio.run(
            seed_admin(
                email=args.email,
                password=_password(),
                full_name=args.full_name,
            )
        )
    except ValueError as exc:
        print(f"admin was not created: {exc}")
        return 1
    except IntegrityError:
        # Do not print SQL parameters: they include the password hash.
        print("admin was not created: the database rejected the record")
        return 1
    if not created:
        print("admin was not created: the email already exists")
        return 2
    print("administrator created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["seed_admin"]
