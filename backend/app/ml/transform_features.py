"""Dataset adapters and strictly causal transaction features.

The label is preserved in the canonical ``label`` metadata column, but is
never used to calculate a feature. PaySim balances and ``isFlaggedFraud`` are
also deliberately ignored. IEEE-CIS ``TransactionDT`` remains relative time:
the adapter never invents a calendar date. Its account key is explicitly a
pseudo-entity made from card1, addr1, and P_emaildomain when available.

The returned frame is stably sorted by time. A supplied mutable ``state`` is
updated in place so chronological chunks produce the same result as a single
call. Queries exclude the current timestamp, including other equal-time rows.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, MutableMapping, Sequence
from math import cos, isnan, pi, sin
import re
from typing import Any

import numpy as np
import pandas as pd

HOUR, DAY, WEEK, MONTH = 3600, 86400, 604800, 2592000
STATE_VERSION = 3
DEFAULT_CONFIG = {
    "chain_window_seconds": HOUR,
    "paysim_pattern_window_seconds": DAY,
    "paysim_step_seconds": HOUR,
    "max_chain_depth": 12,
    "strict_chunk_order": True,
    "ieee_entity_columns": ("card1", "addr1", "P_emaildomain"),
}

COMMON_FEATURES = [
    "amount", "amount_missing", "source_account_missing",
    "destination_account_missing", "transaction_type_missing", "time_missing",
    "relative_hour_sin", "relative_hour_cos", "relative_day_of_week_sin",
    "relative_day_of_week_cos", "source_txn_count_1h", "source_txn_sum_1h",
    "source_txn_count_24h", "source_txn_sum_24h", "source_txn_count_7d",
    "source_txn_sum_7d", "source_amount_mean_30d",
    "amount_to_source_mean_30d_ratio", "is_new_counterparty",
    "new_device_flag", "source_unique_counterparties_30d",
]
AML_FEATURES = [
    "source_recent_inbound_count", "source_recent_inbound_sum",
    "destination_recent_inbound_count", "destination_recent_inbound_sum",
    "source_fan_out_24h", "destination_fan_in_24h",
    "rapid_pass_through_ratio", "causal_chain_depth",
]
PAYSIM_FEATURES = [
    "paysim_prior_transfer_in_count_24h",
    "paysim_prior_transfer_in_sum_24h",
    "paysim_transfer_to_cashout_flag",
    "paysim_transfer_to_cashout_amount_ratio",
]
FEATURE_COLUMNS = tuple(COMMON_FEATURES + AML_FEATURES + PAYSIM_FEATURES)
METADATA_COLUMNS = (
    "event_id", "dataset", "time_semantics", "event_time",
    "relative_time_seconds", "source_account", "destination_account",
    "transaction_type", "currency", "device_type", "device_info",
    "label_name", "label",
)
OUTPUT_COLUMNS = (*METADATA_COLUMNS, *FEATURE_COLUMNS)
_TRANSFORM_INPUT_COLUMNS = (
    "_raw_event_id", "_event_time", "_time", "_relative", "_source",
    "_destination", "_amount", "_transaction_type", "_currency",
    "_device_type", "_device_info", "_label", "_label_name",
    "_time_semantics", "_source_missing", "_destination_missing",
)
_EMPTY_FEATURES = dict.fromkeys(FEATURE_COLUMNS, 0.0)
_NON_ALNUM_UPPER = re.compile(r"[^A-Z0-9]")
_HISTORY_LIST_KEYS = (
    "times", "amounts", "counterparties", "types", "depths", "devices",
    "amount_cumulative", "amount_observed_cumulative",
)


def transform_features(
    frame: pd.DataFrame,
    dataset: str,
    state: MutableMapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return canonical metadata and past-only numeric features.

    ``state`` is optional, mutable, and updated in place. Reuse it for each
    non-decreasing chronological chunk. The result exposes ``feature_columns``,
    ``metadata_columns``, and the updated ``state`` through ``DataFrame.attrs``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    name = _dataset_name(dataset)
    options = _config(config)
    work_state = _state(state, name, options)
    canonical = _adapt(frame, name, options)
    if canonical.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        _attrs(out, work_state, name, options)
        return out

    canonical["_position"] = np.arange(len(canonical))
    canonical = canonical.sort_values(
        ["_time", "_position"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    times = canonical["_time"].dropna()
    if not times.empty:
        first, last_seen = float(times.min()), work_state["last_processed_time"]
        if options["strict_chunk_order"] and last_seen is not None and first < last_seen:
            raise ValueError("chunks must be supplied in non-decreasing timestamp order")
        if work_state["origin_time"] is None:
            work_state["origin_time"] = first

    rows = [
        _transform(row, name, work_state, options)
        for row in canonical.loc[:, _TRANSFORM_INPUT_COLUMNS].itertuples(
            index=False, name=None
        )
    ]
    if not times.empty:
        end = float(times.max())
        old = work_state["last_processed_time"]
        work_state["last_processed_time"] = end if old is None else max(old, end)
    out = pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
    _feature_dtypes(out)
    _attrs(out, work_state, name, options)
    return out


def _dataset_name(dataset: str) -> str:
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    key = re.sub(r"[^a-z0-9]", "", dataset.casefold())
    aliases = {
        "amlworld": "amlworld", "ibmaml": "amlworld",
        "ibmamlworld": "amlworld", "hismall": "amlworld",
        "hilarge": "amlworld", "paysim": "paysim", "ieee": "ieee-cis",
        "ieeecis": "ieee-cis", "ieeefraud": "ieee-cis",
    }
    if key not in aliases:
        raise ValueError("dataset must be AMLworld, PaySim, or IEEE-CIS")
    return aliases[key]


def _config(given: Mapping[str, Any] | None) -> dict[str, Any]:
    if given is not None and not isinstance(given, Mapping):
        raise TypeError("config must be a mapping or None")
    supplied = {} if given is None else dict(given)
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config: {sorted(unknown)}")
    answer = {**DEFAULT_CONFIG, **supplied}
    for key in ("chain_window_seconds", "paysim_pattern_window_seconds",
                "paysim_step_seconds"):
        answer[key] = float(answer[key])
        if not np.isfinite(answer[key]) or answer[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    answer["max_chain_depth"] = int(answer["max_chain_depth"])
    if answer["max_chain_depth"] < 1:
        raise ValueError("max_chain_depth must be at least one")
    answer["strict_chunk_order"] = bool(answer["strict_chunk_order"])
    columns = answer["ieee_entity_columns"]
    if isinstance(columns, str) or not isinstance(columns, Sequence) or not columns:
        raise ValueError("ieee_entity_columns must be a non-empty sequence")
    answer["ieee_entity_columns"] = tuple(map(str, columns))
    return answer


def _serial_config(options: Mapping[str, Any]) -> dict[str, Any]:
    return {k: list(v) if k == "ieee_entity_columns" else v
            for k, v in options.items()}


def _state(state: MutableMapping[str, Any] | None, dataset: str,
           options: Mapping[str, Any]) -> MutableMapping[str, Any]:
    if state is None:
        state = {}
    if not isinstance(state, MutableMapping):
        raise TypeError("state must be a mutable mapping or None")
    if state:
        if state.get("version") != STATE_VERSION:
            raise ValueError("incompatible feature state version")
        if state.get("dataset") != dataset:
            raise ValueError("state belongs to a different dataset")
        if state.get("config") != _serial_config(options):
            raise ValueError("config cannot change while reusing state")
        return state
    state.update({
        "version": STATE_VERSION, "dataset": dataset,
        "config": _serial_config(options), "origin_time": None,
        "last_processed_time": None, "next_event_sequence": 0,
        "source_history": {}, "inbound_history": {},
    })
    return state


def _lookup(frame: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for column in frame.columns:
        result.setdefault(re.sub(r"[^a-z0-9]", "", str(column).casefold()), column)
    return result


def _column(lookup: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.casefold())
        if key in lookup:
            return lookup[key]
    return None


def _series(frame: pd.DataFrame, lookup: Mapping[str, Any], *names: str) -> pd.Series:
    column = _column(lookup, *names)
    if column is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    return frame[column].copy()


def _adapt(frame: pd.DataFrame, dataset: str,
           options: Mapping[str, Any]) -> pd.DataFrame:
    lookup = _lookup(frame)
    if dataset == "amlworld":
        return _adapt_aml(frame, lookup)
    if dataset == "paysim":
        return _adapt_paysim(frame, lookup, options)
    return _adapt_ieee(frame, lookup, options)


def _adapt_aml(frame: pd.DataFrame, lookup: Mapping[str, Any]) -> pd.DataFrame:
    parsed = pd.to_datetime(
        _series(frame, lookup, "Timestamp", "Date Time", "datetime", "time"),
        errors="coerce", utc=True,
    )
    seconds = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = parsed.notna()
    seconds.loc[valid] = parsed.loc[valid].astype("int64") / 1e9
    source_account = _series(frame, lookup, "Account", "From Account",
                             "Sender Account", "source_account")
    destination_account = _series(frame, lookup, "Account.1", "To Account",
                                  "Receiver Account", "destination_account")
    source = _bank_account(_series(frame, lookup, "From Bank", "Sender Bank"),
                           source_account)
    destination = _bank_account(_series(frame, lookup, "To Bank", "Receiver Bank"),
                                destination_account)
    label = _column(lookup, "Is Laundering", "is_laundering", "laundering", "label")
    return _canonical(
        frame, lookup, seconds, parsed, source, destination,
        pd.to_numeric(_series(frame, lookup, "Amount Paid", "Amount Received",
                              "Transaction Amount", "amount"), errors="coerce"),
        _series(frame, lookup, "Payment Format", "Transaction Type", "type"),
        _series(frame, lookup, "Payment Currency", "Receiving Currency", "currency"),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        label, "calendar_utc", np.nan,
    )


def _adapt_paysim(frame: pd.DataFrame, lookup: Mapping[str, Any],
                  options: Mapping[str, Any]) -> pd.DataFrame:
    # Balance columns and isFlaggedFraud are intentionally never selected.
    relative = pd.to_numeric(_series(frame, lookup, "step"), errors="coerce") \
        * options["paysim_step_seconds"]
    return _canonical(
        frame, lookup, relative,
        pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
        _clean_ids(_series(frame, lookup, "nameOrig", "source_account")),
        _clean_ids(_series(frame, lookup, "nameDest", "destination_account")),
        pd.to_numeric(_series(frame, lookup, "amount"), errors="coerce"),
        _series(frame, lookup, "type"),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        _column(lookup, "isFraud", "fraud", "label"), "relative_step", relative,
    )


def _adapt_ieee(frame: pd.DataFrame, lookup: Mapping[str, Any],
                options: Mapping[str, Any]) -> pd.DataFrame:
    relative = pd.to_numeric(_series(frame, lookup, "TransactionDT",
                                     "relative_time_seconds"), errors="coerce")
    source = _ieee_entity(frame, lookup, options["ieee_entity_columns"])
    destination = pd.Series([pd.NA] * len(frame), index=frame.index)
    return _canonical(
        frame, lookup, relative,
        pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
        source, destination,
        pd.to_numeric(_series(frame, lookup, "TransactionAmt", "amount"), errors="coerce"),
        _series(frame, lookup, "ProductCD", "type"),
        pd.Series([pd.NA] * len(frame), index=frame.index),
        _series(frame, lookup, "DeviceType", "device_type"),
        _series(frame, lookup, "DeviceInfo", "device_info", "id_30"),
        _column(lookup, "isFraud", "fraud", "label"),
        "relative_seconds_unknown_origin", relative,
    )


def _canonical(frame: pd.DataFrame, lookup: Mapping[str, Any], time: pd.Series,
               event_time: pd.Series, source: pd.Series, destination: pd.Series,
               amount: pd.Series, transaction_type: pd.Series, currency: pd.Series,
               device_type: pd.Series, device_info: pd.Series,
               label_column: Any | None, semantics: str,
               provided_relative: Any) -> pd.DataFrame:
    event_id = _series(frame, lookup, "Transaction ID", "TransactionID",
                       "transaction_id", "event_id")
    label = (frame[label_column].copy() if label_column is not None
             else pd.Series([pd.NA] * len(frame), index=frame.index))
    if np.isscalar(provided_relative):
        provided_relative = pd.Series([provided_relative] * len(frame), index=frame.index)
    return pd.DataFrame({
        "_raw_event_id": event_id, "_event_time": event_time, "_time": time,
        "_relative": provided_relative, "_source": source,
        "_destination": destination, "_amount": amount,
        "_transaction_type": transaction_type, "_currency": currency,
        "_device_type": device_type, "_device_info": device_info,
        "_label": label, "_label_name": str(label_column) if label_column else pd.NA,
        "_time_semantics": semantics, "_source_missing": source.isna(),
        "_destination_missing": destination.isna(),
    }, index=frame.index)


def _identifier(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return pd.NA
    if isinstance(value, str):
        text = value.strip()
        return text if text else pd.NA
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if _missing(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _clean_ids(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(dtype=object, na_value=None)
    return pd.Series([_identifier(value) for value in raw],
                     index=values.index, dtype="object")


def _bank_account(bank: pd.Series, account: pd.Series) -> pd.Series:
    # The AMLworld ingestion contract reads identifiers as strings to preserve
    # leading zeroes.  Keep that overwhelmingly common path in vectorized
    # pandas operations; the scalar fallback retains support for mixed fixture
    # and caller-provided dtypes.
    if (pd.api.types.is_string_dtype(bank.dtype)
            and pd.api.types.is_string_dtype(account.dtype)):
        clean_bank = bank.astype("string").str.strip()
        clean_account = account.astype("string").str.strip()
        clean_bank = clean_bank.mask(clean_bank.eq(""), pd.NA)
        clean_account = clean_account.mask(clean_account.eq(""), pd.NA)
        combined = clean_account.copy()
        with_bank = clean_account.notna() & clean_bank.notna()
        combined.loc[with_bank] = clean_bank.loc[with_bank].str.cat(
            clean_account.loc[with_bank], sep=":"
        )
        return combined.astype("object")

    banks = bank.to_numpy(dtype=object, na_value=None)
    accounts = account.to_numpy(dtype=object, na_value=None)
    combined = []
    for bank_value, account_value in zip(banks, accounts):
        clean_account = _identifier(account_value)
        if clean_account is pd.NA:
            combined.append(pd.NA)
        else:
            clean_bank = _identifier(bank_value)
            combined.append(
                clean_account if clean_bank is pd.NA
                else f"{clean_bank}:{clean_account}"
            )
    return pd.Series(combined, index=account.index, dtype="object")


def _ieee_entity(frame: pd.DataFrame, lookup: Mapping[str, Any],
                 requested: Sequence[str]) -> pd.Series:
    columns = [(name, _column(lookup, name)) for name in requested]
    columns = [(name, column) for name, column in columns if column is not None]
    if not columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    entities = []
    for position in range(len(frame)):
        parts, observed = [], False
        for logical, actual in columns:
            value = _identifier(frame.iloc[position][actual])
            if _missing(value):
                parts.append(f"{logical}=<missing>")
            else:
                observed = True
                parts.append(f"{logical}={value}")
        entities.append("ieee-pseudo:" + "|".join(parts) if observed else pd.NA)
    return pd.Series(entities, index=frame.index, dtype="object")


def _transform(row: Sequence[Any], dataset: str,
               state: MutableMapping[str, Any],
               options: Mapping[str, Any]) -> tuple[Any, ...]:
    (
        raw_id, event_time, raw_time, raw_relative, raw_source,
        raw_destination, raw_amount, raw_tx_type, currency, raw_device_type,
        raw_device_info, label, label_name, time_semantics, source_missing,
        destination_missing,
    ) = row
    now = _number(raw_time)
    amount = _number(raw_amount)
    source = None if _missing(raw_source) else str(raw_source)
    destination = (None if _missing(raw_destination)
                   else str(raw_destination))
    tx_type = (None if _missing(raw_tx_type) else str(raw_tx_type).strip())
    tx_type_key = _tx_type(tx_type)
    device_type = (None if _missing(raw_device_type)
                   else str(raw_device_type).strip())
    device_info = (None if _missing(raw_device_info)
                   else str(raw_device_info).strip())
    sequence = int(state["next_event_sequence"])
    state["next_event_sequence"] = sequence + 1
    event_id = f"{dataset}:{sequence}" if _missing(raw_id) else raw_id

    relative = _number(raw_relative)
    if relative is None and now is not None and state["origin_time"] is not None:
        relative = now - float(state["origin_time"])

    features = _EMPTY_FEATURES.copy()
    features.update({
        "amount": np.nan if amount is None else amount,
        "amount_missing": int(amount is None),
        "source_account_missing": int(bool(source_missing)),
        "destination_account_missing": int(bool(destination_missing)),
        "transaction_type_missing": int(tx_type is None),
        "time_missing": int(now is None),
    })
    if relative is not None:
        hour_angle = 2 * pi * ((relative % DAY) / DAY)
        week_angle = 2 * pi * ((relative % WEEK) / WEEK)
        features.update({
            "relative_hour_sin": sin(hour_angle),
            "relative_hour_cos": cos(hour_angle),
            "relative_day_of_week_sin": sin(week_angle),
            "relative_day_of_week_cos": cos(week_angle),
        })

    chain_depth = 0
    retention = max(MONTH, options["chain_window_seconds"],
                    options["paysim_pattern_window_seconds"])
    if now is not None and source is not None:
        outgoing = _history(state["source_history"], source)
        _prune(outgoing, now - retention)
        window_bounds: dict[float, tuple[int, int]] = {}
        for suffix, window in (("1h", HOUR), ("24h", DAY), ("7d", WEEK)):
            left, right = _bounds(outgoing, now, window)
            window_bounds[window] = (left, right)
            total, _ = _amount_sum(outgoing, left, right)
            count = right - left
            features[f"source_txn_count_{suffix}"] = count
            features[f"source_txn_sum_{suffix}"] = total
        left_30d, right_30d = _bounds(outgoing, now, MONTH)
        total_30d, observed_amounts = _amount_sum(outgoing, left_30d, right_30d)
        mean = total_30d / observed_amounts if observed_amounts else 0.0
        counterparties = _unique_values(
            outgoing["counterparties"], left_30d, right_30d
        )
        features["source_amount_mean_30d"] = mean
        features["amount_to_source_mean_30d_ratio"] = _ratio(amount, mean)
        features["source_unique_counterparties_30d"] = len(counterparties)
        if destination is not None:
            features["is_new_counterparty"] = int(destination not in counterparties)
        if device_info is not None:
            features["new_device_flag"] = int(not _contains_value(
                outgoing["devices"], left_30d, right_30d, device_info
            ))

        if dataset == "amlworld":
            left_24h, right_24h = window_bounds[DAY]
            features["source_fan_out_24h"] = len(_unique_values(
                outgoing["counterparties"], left_24h, right_24h
            ))
            incoming = _history(state["inbound_history"], source)
            _prune(incoming, now - retention)
            left, right = _bounds(incoming, now, options["chain_window_seconds"])
            count = right - left
            total, _ = _amount_sum(incoming, left, right)
            features["source_recent_inbound_count"] = count
            features["source_recent_inbound_sum"] = total
            features["rapid_pass_through_ratio"] = _ratio(amount, total)
            chain_depth = min(
                options["max_chain_depth"],
                _maximum(incoming["depths"], left, right, default=0) + 1,
            )
            features["causal_chain_depth"] = chain_depth
        elif dataset == "paysim" and tx_type_key == "CASHOUT":
            incoming = _history(state["inbound_history"], source)
            _prune(incoming, now - retention)
            count, total = _typed_stats(
                incoming, now, options["paysim_pattern_window_seconds"], "TRANSFER"
            )
            features["paysim_prior_transfer_in_count_24h"] = count
            features["paysim_prior_transfer_in_sum_24h"] = total
            features["paysim_transfer_to_cashout_flag"] = int(count > 0)
            features["paysim_transfer_to_cashout_amount_ratio"] = _ratio(amount, total)

    if now is not None and destination is not None and dataset == "amlworld":
        incoming = _history(state["inbound_history"], destination)
        _prune(incoming, now - retention)
        left, right = _bounds(incoming, now, options["chain_window_seconds"])
        count = right - left
        total, _ = _amount_sum(incoming, left, right)
        features["destination_recent_inbound_count"] = count
        features["destination_recent_inbound_sum"] = total
        day_left, day_right = _bounds(incoming, now, DAY)
        features["destination_fan_in_24h"] = len(_unique_values(
            incoming["counterparties"], day_left, day_right
        ))

    metadata = (
        event_id, dataset, time_semantics, event_time,
        np.nan if relative is None else relative,
        pd.NA if source is None else source,
        pd.NA if destination is None else destination,
        pd.NA if tx_type is None else tx_type,
        currency,
        pd.NA if device_type is None else device_type,
        pd.NA if device_info is None else device_info,
        label_name, label,
    )

    # Append after calculating features. bisect_left(now) also hides all
    # equal-time rows, even when an equal timestamp crosses a chunk boundary.
    if now is not None and source is not None:
        _append(_history(state["source_history"], source), now, amount,
                destination, tx_type_key, chain_depth, device_info)
    if now is not None and destination is not None:
        _append(_history(state["inbound_history"], destination), now, amount,
                source, tx_type_key, chain_depth, None)
    return metadata + tuple(features[name] for name in FEATURE_COLUMNS)


def _history(container: MutableMapping[str, Any], entity: str) -> dict[str, Any]:
    if entity not in container:
        container[entity] = {key: [] for key in _HISTORY_LIST_KEYS}
        container[entity]["amount_base"] = 0.0
        container[entity]["amount_observed_base"] = 0
    return container[entity]


def _append(history: MutableMapping[str, Any], timestamp: float,
            amount: float | None, counterparty: str | None,
            normalized_transaction_type: str, depth: int,
            device_info: str | None) -> None:
    history["times"].append(float(timestamp))
    history["amounts"].append(amount)
    history["counterparties"].append(counterparty)
    history["types"].append(normalized_transaction_type)
    history["depths"].append(int(depth))
    history["devices"].append(device_info)
    previous_total = (history["amount_cumulative"][-1]
                      if history["amount_cumulative"]
                      else history["amount_base"])
    previous_observed = (history["amount_observed_cumulative"][-1]
                         if history["amount_observed_cumulative"]
                         else history["amount_observed_base"])
    history["amount_cumulative"].append(
        previous_total + (0.0 if amount is None else float(amount))
    )
    history["amount_observed_cumulative"].append(
        previous_observed + int(amount is not None)
    )


def _prune(history: MutableMapping[str, Any], cutoff: float) -> None:
    drop = bisect_left(history["times"], cutoff)
    if drop:
        history["amount_base"] = history["amount_cumulative"][drop - 1]
        history["amount_observed_base"] = (
            history["amount_observed_cumulative"][drop - 1]
        )
        for key in _HISTORY_LIST_KEYS:
            del history[key][:drop]


def _bounds(history: Mapping[str, Any], now: float,
            window: float) -> tuple[int, int]:
    return (bisect_left(history["times"], now - window),
            bisect_left(history["times"], now))


def _stats(history: Mapping[str, Any], now: float, window: float
           ) -> tuple[int, float, float, list[int], set[str]]:
    left, right = _bounds(history, now, window)
    amounts = [float(value) for value in history["amounts"][left:right]
               if value is not None]
    counterparties = {str(value) for value in history["counterparties"][left:right]
                      if value is not None}
    total = float(sum(amounts))
    return (right - left, total, total / len(amounts) if amounts else 0.0,
            [int(value) for value in history["depths"][left:right]], counterparties)


def _amount_sum(
    history: Mapping[str, Any], left: int, right: int
) -> tuple[float, int]:
    if left == right:
        return 0.0, 0
    totals = history["amount_cumulative"]
    observations = history["amount_observed_cumulative"]
    before_total = totals[left - 1] if left else history["amount_base"]
    before_observed = (
        observations[left - 1] if left else history["amount_observed_base"]
    )
    return (
        float(totals[right - 1] - before_total),
        int(observations[right - 1] - before_observed),
    )


def _unique_values(values: Sequence[Any], left: int, right: int) -> set[str]:
    return {
        str(values[position])
        for position in range(left, right)
        if values[position] is not None
    }


def _contains_value(
    values: Sequence[Any], left: int, right: int, wanted: Any
) -> bool:
    return any(values[position] == wanted for position in range(left, right))


def _maximum(
    values: Sequence[Any], left: int, right: int, *, default: int
) -> int:
    answer = default
    for position in range(left, right):
        value = int(values[position])
        if value > answer:
            answer = value
    return answer


def _typed_stats(history: Mapping[str, Any], now: float, window: float,
                 wanted: str) -> tuple[int, float]:
    left, right = _bounds(history, now, window)
    count, total = 0, 0.0
    types, amounts = history["types"], history["amounts"]
    for position in range(left, right):
        transaction_type, amount = types[position], amounts[position]
        if transaction_type == wanted:
            count += 1
            if amount is not None:
                total += amount
    return count, total


def _tx_type(value: str | None) -> str:
    return "" if value is None else _NON_ALNUM_UPPER.sub("", value.upper())


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return 0.0
    return numerator / denominator


def _number(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return answer if np.isfinite(answer) else None


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return False
    if isinstance(value, (float, np.floating)):
        return isnan(float(value))
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _feature_dtypes(frame: pd.DataFrame) -> None:
    integer = {
        name for name in FEATURE_COLUMNS
        if "count" in name or name.endswith("_missing") or name.endswith("_flag")
        or name.startswith("is_new_") or "unique_counterparties" in name
        or "fan_in" in name or "fan_out" in name or name == "causal_chain_depth"
    }
    for name in FEATURE_COLUMNS:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        frame[name] = (numeric.fillna(0).astype("int64") if name in integer
                       else numeric.astype("float64"))


def _attrs(frame: pd.DataFrame, state: MutableMapping[str, Any], dataset: str,
           options: Mapping[str, Any]) -> None:
    frame.attrs.update({
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS), "label_column": "label",
        "dataset": dataset, "state": state, "config": _serial_config(options),
    })
    if dataset == "ieee-cis":
        frame.attrs["entity_semantics"] = (
            "source is a pseudo entity made from available "
            f"{list(options['ieee_entity_columns'])}; destination is unobserved; "
            "graph features are disabled"
        )
        frame.attrs["time_semantics"] = (
            "TransactionDT relative seconds; no calendar origin is asserted"
        )


__all__ = ["transform_features", "FEATURE_COLUMNS", "METADATA_COLUMNS"]
