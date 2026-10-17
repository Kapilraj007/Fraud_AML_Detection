import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  CircleAlert,
  FileSearch,
  LockKeyhole,
  Play,
  RefreshCw,
  ShieldCheck,
  UserRoundPlus,
} from "lucide-react";

import { caseStatusLabel, formatDateTime } from "../../lib/format";
import type { CaseDetail, CaseDispositionCreate, User } from "../../lib/types";
import { DispositionForm } from "./DispositionForm";
import { ExplanationPanel } from "./ExplanationPanel";
import { RiskScoreCard } from "./RiskScoreCard";
import { StatusTimeline } from "./StatusTimeline";
import { TransactionDetails } from "./TransactionDetails";
import {
  getPredictionExplanation,
  getPrimaryPrediction,
  getStudyArmPresentation,
} from "./studyArms";

interface CaseDetailPanelProps {
  caseDetail?: CaseDetail;
  user?: User | null;
  isLoading: boolean;
  error?: string;
  staleMessage?: string;
  elapsedSeconds: number;
  actionPending: boolean;
  onBack: () => void;
  onReload: () => void;
  onClaim: () => Promise<void>;
  onStartReview: () => Promise<void>;
  onDisposition: (payload: CaseDispositionCreate) => Promise<void>;
}

export function CaseDetailPanel({
  caseDetail,
  user,
  isLoading,
  error,
  staleMessage,
  elapsedSeconds,
  actionPending,
  onBack,
  onReload,
  onClaim,
  onStartReview,
  onDisposition,
}: CaseDetailPanelProps) {
  if (isLoading) return <DetailSkeleton />;

  if (error) {
    return (
      <section className="case-detail case-detail--state" role="alert">
        <CircleAlert aria-hidden="true" size={28} />
        <h2>Case unavailable</h2>
        <p>{error}</p>
        <button className="button button--secondary" type="button" onClick={onReload}>
          <RefreshCw aria-hidden="true" size={17} /> Try again
        </button>
      </section>
    );
  }

  if (!caseDetail) {
    return (
      <section className="case-detail case-detail--state">
        <FileSearch aria-hidden="true" size={30} />
        <h2>Select a case</h2>
        <p>Choose a ranked case from the queue to review its evidence and workflow.</p>
      </section>
    );
  }

  const presentation = getStudyArmPresentation(caseDetail.study_arm);
  const prediction = getPrimaryPrediction(caseDetail.predictions);
  const explanation = getPredictionExplanation(prediction);
  const owner = caseDetail.assigned_to_id === user?.id;
  const adminCanAct = user?.role === "admin";
  const canStart = (owner || adminCanAct) && ["claimed", "more_info_requested"].includes(caseDetail.status);
  const canDispose = (owner || adminCanAct) && caseDetail.status === "in_review";
  const terminal = ["escalated", "dismissed", "closed"].includes(caseDetail.status);

  return (
    <article className="case-detail" aria-labelledby="case-detail-heading">
      <button className="case-detail__back" type="button" onClick={onBack}>
        <ArrowLeft aria-hidden="true" size={17} /> Back to queue
      </button>

      {staleMessage ? (
        <div className="case-stale-notice" role="alert">
          <AlertTriangle aria-hidden="true" size={18} />
          <span>{staleMessage}</span>
          <button type="button" onClick={onReload}>Reload latest</button>
        </div>
      ) : null}

      <header className="case-detail__header">
        <div className="case-detail__identity">
          <div className="case-detail__title-row">
            <span className="case-detail__rank">Rank {caseDetail.queue_rank}</span>
            <span className={`status-badge status-badge--${caseDetail.status}`}>
              {caseStatusLabel(caseDetail.status)}
            </span>
          </div>
          <h1 id="case-detail-heading">Review {shortReviewId(caseDetail.review_id)}</h1>
          <p>Alerted {formatDateTime(caseDetail.alert_hour)} · Last updated {formatDateTime(caseDetail.updated_at)}</p>
        </div>
        <div className="study-arm-card" aria-label={`Assigned study arm: ${presentation.label}`}>
          <LockKeyhole aria-hidden="true" size={18} />
          <span>
            <small>Assigned evidence view</small>
            <strong>{presentation.label}</strong>
            <span>{presentation.description}</span>
          </span>
          <span className="study-arm-card__locked">Locked</span>
        </div>
      </header>

      <div className="case-detail__action-bar">
        {caseDetail.status === "open" ? (
          <button className="button button--primary" type="button" disabled={actionPending} onClick={() => onClaim().catch(() => undefined)}>
            <UserRoundPlus aria-hidden="true" size={18} />
            {actionPending ? "Claiming…" : "Claim case"}
          </button>
        ) : null}
        {canStart ? (
          <button className="button button--primary" type="button" disabled={actionPending} onClick={() => onStartReview().catch(() => undefined)}>
            <Play aria-hidden="true" size={18} />
            {actionPending ? "Starting…" : caseDetail.status === "more_info_requested" ? "Resume review" : "Start review"}
          </button>
        ) : null}
        {caseDetail.assigned_to_id && !owner && !adminCanAct && !terminal ? (
          <p className="case-detail__ownership"><LockKeyhole aria-hidden="true" size={16} /> Assigned to another analyst</p>
        ) : null}
        {terminal ? (
          <p className="case-detail__complete"><CheckCircle2 aria-hidden="true" size={17} /> Review complete</p>
        ) : null}
        <p className="case-detail__workflow-hint">
          <ShieldCheck aria-hidden="true" size={16} /> Evidence shown here follows the case's fixed study assignment.
        </p>
      </div>

      <div className="case-detail__content-grid">
        <section className="case-detail__main" aria-label="Case evidence and disposition">
          <TransactionDetails transactions={caseDetail.transactions} />
          {presentation.showScore ? (
            <RiskScoreCard
              combined={caseDetail.risk_score}
              supervised={prediction?.supervised_score}
              anomaly={prediction?.anomaly_score}
            />
          ) : null}
          {presentation.showExplanation ? (
            <ExplanationPanel
              explanation={explanation}
              reliabilityGated={presentation.applyReliabilityGate}
            />
          ) : null}
          {canDispose ? (
            <DispositionForm
              elapsedSeconds={elapsedSeconds}
              isSubmitting={actionPending}
              onSubmit={onDisposition}
            />
          ) : null}
        </section>
        <aside className="case-detail__aside" aria-label="Case workflow summary">
          <StatusTimeline caseDetail={caseDetail} />
          <CaseMetadata caseDetail={caseDetail} />
        </aside>
      </div>
    </article>
  );
}

function CaseMetadata({ caseDetail }: { caseDetail: CaseDetail }) {
  return (
    <section className="case-section case-metadata" aria-labelledby="case-metadata-heading">
      <h3 id="case-metadata-heading">Case record</h3>
      <dl>
        <div><dt>Review ID</dt><dd>{caseDetail.review_id}</dd></div>
        <div><dt>Queue rank</dt><dd>{caseDetail.queue_rank}</dd></div>
        <div><dt>Status</dt><dd>{caseStatusLabel(caseDetail.status)}</dd></div>
        <div><dt>Assignment</dt><dd>{caseDetail.assigned_to_id ? "Assigned" : "Unassigned"}</dd></div>
        <div><dt>Created</dt><dd>{formatDateTime(caseDetail.created_at)}</dd></div>
      </dl>
    </section>
  );
}

function DetailSkeleton() {
  return (
    <section className="case-detail case-detail--loading" aria-label="Loading case details" role="status">
      <span className="sr-only">Loading case details</span>
      <div className="case-detail__skeleton-title" aria-hidden="true" />
      <div className="case-detail__skeleton-grid" aria-hidden="true"><span /><span /></div>
    </section>
  );
}

function shortReviewId(reviewId: string) {
  return reviewId.length > 16 ? reviewId.slice(0, 12).toUpperCase() : reviewId;
}
