import type { CaseStatus, UserRole } from "./types";

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const relativeTimeFormatter = new Intl.RelativeTimeFormat(undefined, {
  numeric: "auto",
});

const compactNumberFormatter = new Intl.NumberFormat(undefined, {
  notation: "compact",
  maximumFractionDigits: 1,
});

export function formatCurrency(
  value: number | string | null | undefined,
  currency = "USD",
): string {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";

  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: currency.toUpperCase(),
      maximumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${currency.toUpperCase()} ${amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  }
}

export function formatDateTime(value: string | Date | null | undefined): string {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : dateTimeFormatter.format(date);
}

export function formatRelativeTime(
  value: string | Date | null | undefined,
  now = new Date(),
): string {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";

  const seconds = (date.getTime() - now.getTime()) / 1_000;
  const ranges: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ["year", 31_536_000],
    ["month", 2_592_000],
    ["week", 604_800],
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
    ["second", 1],
  ];
  const [unit, divisor] = ranges.find(([, size]) => Math.abs(seconds) >= size) ?? ["second", 1];
  return relativeTimeFormatter.format(Math.round(seconds / divisor), unit);
}

export function formatPercent(
  value: number | null | undefined,
  maximumFractionDigits = 1,
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits,
  }).format(value);
}

export function formatCompactNumber(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : compactNumberFormatter.format(value);
}

export function humanize(value: string | null | undefined): string {
  if (!value) return "—";
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function caseStatusLabel(status: CaseStatus): string {
  return humanize(status);
}

export function humanizeRole(role: UserRole | null | undefined): string {
  return humanize(role);
}

export type RiskTone = "critical" | "high" | "medium" | "low";

export function riskTone(score: number): RiskTone {
  if (score >= 0.85) return "critical";
  if (score >= 0.65) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}
