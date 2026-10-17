export type UUID = string;
export type ISODateTime = string;
export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type UserRole = "fraud_analyst" | "senior_investigator" | "admin";
export type CaseStatus =
  | "open"
  | "claimed"
  | "in_review"
  | "escalated"
  | "dismissed"
  | "more_info_requested"
  | "closed";
export type DispositionAction = "escalate" | "dismiss" | "request_more_info";
export type DriftStatus = "stable" | "watch" | "drift_detected";
export type RetrainingStatus =
  | "open"
  | "approved"
  | "running"
  | "completed"
  | "rejected"
  | "failed";
export type ModelStatus = "candidate" | "active" | "retired" | "failed";

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: "bearer";
  expires_in: number;
  refresh_expires_in: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface User {
  id: UUID;
  email: string;
  full_name: string | null;
  role: UserRole;
  is_active: boolean;
  created_at: ISODateTime;
  updated_at: ISODateTime;
  last_login_at: ISODateTime | null;
}

export interface Transaction {
  id: UUID;
  event_id: string;
  dataset: string;
  event_time: ISODateTime;
  masked_source_account: string | null;
  masked_destination_account: string | null;
  amount: string | number;
  currency: string;
  transaction_type: string;
  created_at: ISODateTime;
}

export interface TransactionCreate {
  event_id: string;
  dataset?: string;
  event_time: ISODateTime;
  source_account_token: string;
  destination_account_token: string;
  amount: string | number;
  currency: string;
  transaction_type: string;
  raw_attributes?: Record<string, JsonPrimitive>;
}

export interface ExplanationFeature {
  feature: string;
  display_name?: string;
  value: number;
  direction?: "increases_risk" | "decreases_risk" | string;
  attribution?: number;
  [key: string]: JsonValue | undefined;
}

export interface Explanation {
  top_features: ExplanationFeature[];
  narrative: string | null;
  reliable?: boolean;
  reliability_reasons?: string[];
  is_withheld?: boolean;
  gate_reason?: string | null;
}

export interface Prediction {
  prediction_id: UUID | null;
  transaction_id: UUID | null;
  event_id: string | null;
  supervised_probability: number;
  anomaly_probability: number | null;
  combined_risk: number;
  raw_margin: number;
  model_version: string;
  feature_schema_version: number;
  explanation: Explanation;
  scored_at: ISODateTime;
}

export interface CasePrediction {
  id: UUID;
  transaction_id: UUID;
  model_version_id: UUID;
  supervised_score: number;
  anomaly_score: number;
  combined_score: number;
  decision_threshold: number;
  predicted_positive: boolean;
  uncertainty: number;
  reliability_score: number;
  reliability_status: string;
  explanation_gate_reason: string | null;
  drift_status: DriftStatus;
  explanation: Explanation | null;
  created_at: ISODateTime;
}

export interface Case {
  id: UUID;
  review_id: string;
  status: CaseStatus;
  queue_rank: number;
  risk_score: number;
  assigned_to_id: UUID | null;
  study_arm: string;
  alert_hour: ISODateTime;
  updated_at: ISODateTime;
  created_at: ISODateTime;
}

export interface CaseDisposition {
  id: UUID;
  case_id: UUID;
  analyst_id: UUID;
  action: DispositionAction;
  confidence: number | null;
  notes: string | null;
  model_agreement: boolean | null;
  decision_seconds: number | null;
  created_at: ISODateTime;
}

export interface CaseDetail extends Case {
  transactions: Transaction[];
  dispositions: CaseDisposition[];
  predictions?: CasePrediction[];
}

export interface CaseListParams {
  status?: CaseStatus;
  assigned_to_me?: boolean;
  limit?: number;
  offset?: number;
}

export interface CaseDispositionCreate {
  decision: DispositionAction;
  confidence: number;
  rationale?: string | null;
  model_agreement?: boolean | null;
  review_duration_seconds: number;
}

export interface PredictionRequest {
  transaction_id?: UUID;
  event_id?: string;
  features: Record<string, number>;
}

export interface DriftSnapshot {
  id: UUID;
  model_version_id: UUID;
  window_start: ISODateTime;
  window_end: ISODateTime;
  status: DriftStatus;
  adwin_detected: boolean;
  feature_metrics: Record<string, JsonValue>;
  prediction_metrics: Record<string, JsonValue>;
  created_at: ISODateTime;
}

export interface MonitoringStatus {
  active_model_id: UUID;
  active_model_version: string;
  status: DriftStatus | null;
  latest_snapshot: DriftSnapshot | null;
}

export interface MonitoringHistoryParams {
  model_version_id?: UUID;
  limit?: number;
  offset?: number;
}

export interface RetrainingRequestCreate {
  reason: string;
  drift_snapshot_id: UUID;
}

export interface RetrainingRequest {
  id: UUID;
  status: RetrainingStatus;
  reason: string;
  requested_by_id: UUID;
  drift_snapshot_id: UUID;
  created_at: ISODateTime;
  updated_at: ISODateTime;
}

export interface ModelVersion {
  id: UUID;
  version: string;
  name: string;
  algorithm: string;
  artifact_uri: string;
  artifact_sha256: string;
  feature_schema: Record<string, JsonValue>;
  metrics: Record<string, JsonValue>;
  status: ModelStatus | string;
  created_at: ISODateTime;
  activated_at: ISODateTime | null;
}

export interface ModelPromotionRequest {
  expected_current_version?: string | null;
  reason: string;
}

export interface AuditEvent {
  id: UUID;
  actor_user_id: UUID | null;
  event_type: string;
  resource_type: string;
  resource_id: string | null;
  request_id: string | null;
  safe_details: Record<string, JsonValue>;
  created_at: ISODateTime;
}

export interface AuditListParams {
  event_type?: string;
  resource_type?: string;
  limit?: number;
  offset?: number;
}

export interface TransactionListParams {
  limit?: number;
  offset?: number;
  event_after?: ISODateTime;
  event_before?: ISODateTime;
}

export interface ValidationIssue {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
  [key: string]: unknown;
}
