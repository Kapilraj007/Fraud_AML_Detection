#!/usr/bin/env python3
"""Register a verified model bundle in Neon and optionally activate it.

The bundle is checksum-verified before any database write.  Registration is
separate from training so offline experiments never need production database
credentials and activation remains an explicit human decision.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.db.models import ModelVersion  # noqa: E402
from backend.app.db.repositories import (  # noqa: E402
    get_user_by_email,
    promote_model_version,
    register_verified_model_bundle,
)
from backend.app.db.session import session_scope  # noqa: E402
from backend.app.ml.artifacts import verify_model_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        help="bundle directory (defaults to MODEL_ARTIFACT_DIR)",
    )
    parser.add_argument(
        "--name",
        default="aml-primary",
        help="stable registry name (default: aml-primary)",
    )
    parser.add_argument(
        "--created-by",
        help="email of the administrator registering the candidate",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="retire the current active model and activate this candidate",
    )
    return parser


async def register(arguments: argparse.Namespace) -> ModelVersion:
    settings = get_settings()
    bundle_path = Path(arguments.bundle or settings.model_artifact_dir).resolve()
    verified = verify_model_bundle(bundle_path)

    async with session_scope() as session:
        creator_id = None
        if arguments.created_by:
            creator = await get_user_by_email(session, arguments.created_by)
            if creator is None or not creator.is_active:
                raise ValueError("--created-by must identify an active user")
            creator_id = creator.id

        existing = await session.execute(
            select(ModelVersion).where(
                ModelVersion.name == arguments.name,
                ModelVersion.version == str(verified.manifest["bundle_id"]),
            )
        )
        model = existing.scalar_one_or_none()
        if model is None:
            model = await register_verified_model_bundle(
                session,
                bundle=verified,
                name=arguments.name,
                created_by_id=creator_id,
                ensemble_weights={"supervised": 0.85, "anomaly": 0.15},
            )
        if arguments.activate:
            model = await promote_model_version(session, model_version_id=model.id)
        return model


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        model = asyncio.run(register(arguments))
    except (FileNotFoundError, IntegrityError, LookupError, OSError, ValueError) as error:
        print(f"Model registration failed: {error}", file=sys.stderr)
        return 1
    print(
        f"registered model {model.name}:{model.version} "
        f"as {model.status.value} ({model.id})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
