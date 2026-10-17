from __future__ import annotations

from io import StringIO

import pytest

from pipelines.parse_amlworld_patterns import (
    PatternFormatError,
    parse_pattern_lines,
    transaction_fingerprint,
)


TRANSACTION = (
    "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,"
    "2848.96,Euro,ACH,1"
)


def test_parser_emits_stable_case_and_transaction_identifiers() -> None:
    content = f"""
BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
{TRANSACTION}
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - RANDOM: Max 1 hops
{TRANSACTION}
END LAUNDERING ATTEMPT - RANDOM
"""
    parsed = parse_pattern_lines(StringIO(content), source="fixture.txt")

    assert parsed["case_id"].tolist() == [
        "amlworld-case-00001",
        "amlworld-case-00002",
    ]
    assert parsed["typology"].tolist() == ["FAN-OUT", "RANDOM"]
    assert parsed["case_position"].tolist() == [1, 1]
    assert parsed["event_id"].nunique() == 1
    fields = TRANSACTION.split(",")
    assert parsed.loc[0, "event_id"] == transaction_fingerprint(fields)
    assert parsed.loc[0, "From Bank"] == "021174"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "BEGIN LAUNDERING ATTEMPT - FAN-OUT\n"
            "BEGIN LAUNDERING ATTEMPT - RANDOM\n",
            "nested BEGIN",
        ),
        (
            f"BEGIN LAUNDERING ATTEMPT - FAN-OUT\n{TRANSACTION}\n"
            "END LAUNDERING ATTEMPT - FAN-IN\n",
            "does not match",
        ),
        (
            "BEGIN LAUNDERING ATTEMPT - RANDOM\n"
            + TRANSACTION[:-1]
            + "0\nEND LAUNDERING ATTEMPT - RANDOM\n",
            "label must equal 1",
        ),
        ("orphan text\n", "outside a laundering block"),
    ],
)
def test_parser_rejects_malformed_blocks(content: str, message: str) -> None:
    with pytest.raises(PatternFormatError, match=message):
        parse_pattern_lines(StringIO(content), source="bad.txt")


def test_parser_reports_unclosed_block_at_eof() -> None:
    with pytest.raises(PatternFormatError, match="unclosed laundering block"):
        parse_pattern_lines(
            StringIO(f"BEGIN LAUNDERING ATTEMPT - RANDOM\n{TRANSACTION}\n"),
            source="bad.txt",
        )

