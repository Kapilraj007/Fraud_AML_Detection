import type { DriftSnapshot, JsonValue } from "../../lib/types";

export type KsReading = {
  feature: string;
  statistic: number | null;
  pValue: number | null;
  drifted: boolean | null;
};

function isRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: JsonValue | undefined): number | null {
  const number = typeof value === "number" ? value : Number.NaN;
  return Number.isFinite(number) ? number : null;
}

function firstNumber(
  value: Record<string, JsonValue>,
  keys: string[],
): number | null {
  for (const key of keys) {
    const candidate = finiteNumber(value[key]);
    if (candidate !== null) return candidate;
  }
  return null;
}

function readingFromRecord(
  feature: string,
  value: Record<string, JsonValue>,
): KsReading | null {
  const statistic = firstNumber(value, ["ks", "statistic", "ks_statistic"]);
  const pValue = firstNumber(value, ["p_value", "pValue", "ks_p_value"]);
  const drifted = typeof value.drifted === "boolean" ? value.drifted : null;
  if (statistic === null && pValue === null && drifted === null) return null;
  return { feature, statistic, pValue, drifted };
}

/**
 * Drift payloads are intentionally schema-flexible JSON. This normalizer accepts
 * both the persisted per-feature map and the DriftMonitor `features` array shape.
 */
export function extractKsReadings(snapshot: DriftSnapshot): KsReading[] {
  const readings = new Map<string, KsReading>();
  const featureList = snapshot.feature_metrics.features;

  if (Array.isArray(featureList)) {
    for (const item of featureList) {
      if (!isRecord(item) || typeof item.feature !== "string") continue;
      const reading = readingFromRecord(item.feature, item);
      if (reading) readings.set(reading.feature, reading);
    }
  }

  for (const [feature, value] of Object.entries(snapshot.feature_metrics)) {
    if (feature === "features") continue;
    if (typeof value === "number" && Number.isFinite(value)) {
      readings.set(feature, {
        feature,
        statistic: value,
        pValue: null,
        drifted: null,
      });
      continue;
    }
    if (!isRecord(value)) continue;
    const reading = readingFromRecord(feature, value);
    if (reading) readings.set(feature, reading);
  }

  return [...readings.values()].sort((left, right) =>
    left.feature.localeCompare(right.feature),
  );
}

export function monitoredFeatures(history: DriftSnapshot[], limit = 6): string[] {
  const names = new Set<string>();
  for (const snapshot of history) {
    for (const reading of extractKsReadings(snapshot)) {
      if (reading.statistic !== null) names.add(reading.feature);
    }
  }
  return [...names].sort().slice(0, limit);
}
