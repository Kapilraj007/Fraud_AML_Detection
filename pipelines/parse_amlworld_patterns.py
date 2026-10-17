"""Parse AMLworld laundering-pattern ground truth.

The upstream ``*_Patterns.txt`` files are blocks with a BEGIN marker, one or
more transaction CSV records, and a matching END marker.  This module treats
that format as data rather than free text: malformed/nested blocks, mismatched
typologies, and non-transaction lines fail with a source line number.

Transaction fingerprints are SHA-256 hashes over the eleven source fields.
They deliberately do not include a case identifier, so the same transaction
can be associated with more than one laundering case and can be joined to the
transaction CSV without using a label or row position.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd


AMLDATA_COLUMNS = (
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)

_BEGIN = re.compile(
    r"^BEGIN LAUNDERING ATTEMPT - (?P<typology>[A-Z][A-Z-]*)"
    r"(?:\s*:\s*(?P<detail>.*?))?\s*$"
)
_END = re.compile(r"^END LAUNDERING ATTEMPT - (?P<typology>[A-Z][A-Z-]*)\s*$")
_SEP = "\x1f"


class PatternFormatError(ValueError):
    """Raised when an AMLworld pattern file violates its block grammar."""


@dataclass(frozen=True)
class _Block:
    case_index: int
    typology: str
    detail: str | None
    begin_line: int

    @property
    def case_id(self) -> str:
        return f"amlworld-case-{self.case_index:05d}"


def transaction_fingerprint(values: Sequence[object]) -> str:
    """Return a stable content identifier for one AMLworld transaction.

    Values are stripped because both upstream files are unquoted CSV and
    incidental surrounding whitespace is not semantically meaningful.  The
    field count is checked to prevent identifiers for partial records.
    """
    if len(values) != len(AMLDATA_COLUMNS):
        raise ValueError(
            f"expected {len(AMLDATA_COLUMNS)} transaction fields, got {len(values)}"
        )
    canonical = _SEP.join("" if value is None else str(value).strip() for value in values)
    return "amlworld-tx-" + sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_frame(frame: pd.DataFrame) -> pd.Series:
    """Fingerprint AMLworld rows while preserving leading zero identifiers."""
    missing = [column for column in AMLDATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing AMLworld columns: {missing}")
    # itertuples avoids the substantial per-row Series allocation performed by
    # DataFrame.apply; this matters for the multi-million-row source artifact.
    return pd.Series(
        (
            transaction_fingerprint(values)
            for values in frame.loc[:, AMLDATA_COLUMNS].itertuples(index=False, name=None)
        ),
        index=frame.index,
        dtype="object",
    )


def parse_pattern_lines(lines: Iterable[str], source: str = "<memory>") -> pd.DataFrame:
    """Parse pattern lines into one row per transaction/case membership."""
    records: list[dict[str, object]] = []
    current: _Block | None = None
    case_count = 0
    position = 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue

        begin = _BEGIN.fullmatch(line.strip())
        if begin:
            if current is not None:
                raise _error(source, line_number, "nested BEGIN before prior END")
            case_count += 1
            current = _Block(
                case_index=case_count,
                typology=begin.group("typology"),
                detail=(begin.group("detail") or "").strip() or None,
                begin_line=line_number,
            )
            position = 0
            continue

        end = _END.fullmatch(line.strip())
        if end:
            if current is None:
                raise _error(source, line_number, "END outside a laundering block")
            if end.group("typology") != current.typology:
                raise _error(
                    source,
                    line_number,
                    f"END typology {end.group('typology')!r} does not match "
                    f"BEGIN typology {current.typology!r}",
                )
            if position == 0:
                raise _error(source, line_number, "laundering block has no transactions")
            current = None
            continue

        if current is None:
            raise _error(source, line_number, "content outside a laundering block")

        try:
            fields = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            raise _error(source, line_number, f"invalid CSV transaction: {exc}") from exc
        if len(fields) != len(AMLDATA_COLUMNS):
            raise _error(
                source,
                line_number,
                f"expected {len(AMLDATA_COLUMNS)} transaction fields, got {len(fields)}",
            )
        if fields[-1].strip() != "1":
            raise _error(source, line_number, "pattern transaction label must equal 1")

        position += 1
        values = {column: value.strip() for column, value in zip(AMLDATA_COLUMNS, fields)}
        fingerprint = transaction_fingerprint(fields)
        records.append(
            {
                "case_id": current.case_id,
                "typology": current.typology,
                "case_detail": current.detail,
                "case_position": position,
                "pattern_line_number": line_number,
                "transaction_fingerprint": fingerprint,
                # Pattern rows cannot disambiguate identical duplicate source
                # records, so this content-stable identifier is the event key.
                "event_id": fingerprint,
                **values,
            }
        )

    if current is not None:
        raise _error(
            source,
            current.begin_line,
            f"unclosed laundering block {current.case_id} ({current.typology})",
        )
    if case_count == 0:
        raise PatternFormatError(f"{source}: no laundering blocks found")

    return pd.DataFrame.from_records(
        records,
        columns=[
            "case_id",
            "typology",
            "case_detail",
            "case_position",
            "pattern_line_number",
            "transaction_fingerprint",
            "event_id",
            *AMLDATA_COLUMNS,
        ],
    )


def parse_pattern_file(path: str | Path) -> pd.DataFrame:
    """Parse an AMLworld pattern file using UTF-8 with BOM tolerance."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return parse_pattern_lines(handle, source=str(source_path))


def _error(source: str, line_number: int, message: str) -> PatternFormatError:
    return PatternFormatError(f"{source}:{line_number}: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="AMLworld *_Patterns.txt file")
    parser.add_argument("--output", type=Path, help="optional .parquet or .csv output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parsed = parse_pattern_file(args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.casefold() == ".parquet":
            parsed.to_parquet(args.output, index=False)
        elif args.output.suffix.casefold() == ".csv":
            parsed.to_csv(args.output, index=False)
        else:
            raise ValueError("--output must end in .parquet or .csv")
    print(
        f"parsed {parsed['case_id'].nunique():,} cases and {len(parsed):,} "
        f"transaction memberships"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMLDATA_COLUMNS",
    "PatternFormatError",
    "fingerprint_frame",
    "parse_pattern_file",
    "parse_pattern_lines",
    "transaction_fingerprint",
]
