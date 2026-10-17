import {
  ChevronRight,
  CircleAlert,
  Inbox,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";

import { caseStatusLabel, formatDateTime, formatPercent, riskTone } from "../../lib/format";
import type { Case } from "../../lib/types";
import { getStudyArmPresentation } from "./studyArms";

interface CaseQueueProps {
  cases: Case[];
  selectedCaseId?: string;
  isLoading: boolean;
  isRefreshing: boolean;
  error?: string;
  lastUpdatedAt?: number;
  onSelect: (caseId: string) => void;
  onRefresh: () => void;
}

export function CaseQueue({
  cases,
  selectedCaseId,
  isLoading,
  isRefreshing,
  error,
  lastUpdatedAt,
  onSelect,
  onRefresh,
}: CaseQueueProps) {
  return (
    <section className="case-queue" aria-labelledby="case-queue-heading">
      <header className="case-queue__header">
        <div>
          <div className="case-queue__eyebrow">
            <span className="case-queue__live-dot" aria-hidden="true" />
            Live triage queue
          </div>
          <h2 id="case-queue-heading">Ranked cases</h2>
          <p>
            Highest-priority cases appear first. The queue refreshes every 30 seconds.
          </p>
        </div>
        <button
          className="case-queue__refresh"
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label={isRefreshing ? "Refreshing case queue" : "Refresh case queue"}
        >
          <RefreshCw className={isRefreshing ? "is-spinning" : undefined} aria-hidden="true" size={18} />
        </button>
      </header>

      {lastUpdatedAt ? (
        <p className="case-queue__updated" aria-live="polite">
          Last synced {formatDateTime(new Date(lastUpdatedAt).toISOString())}
        </p>
      ) : null}

      {isLoading ? <QueueSkeleton /> : null}

      {!isLoading && error ? (
        <div className="case-queue__state case-queue__state--error" role="alert">
          <CircleAlert aria-hidden="true" size={22} />
          <div>
            <strong>Queue unavailable</strong>
            <p>{error}</p>
            <button type="button" onClick={onRefresh}>Try again</button>
          </div>
        </div>
      ) : null}

      {!isLoading && !error && cases.length === 0 ? (
        <div className="case-queue__state">
          <Inbox aria-hidden="true" size={24} />
          <div>
            <strong>No matching cases</strong>
            <p>Adjust the filters or wait for the next queue refresh.</p>
          </div>
        </div>
      ) : null}

      {!isLoading && !error && cases.length > 0 ? (
        <ol className="case-queue__list" aria-label="Cases ordered by queue rank">
          {cases.map((item) => {
            const presentation = getStudyArmPresentation(item.study_arm);
            const selected = selectedCaseId === item.id;

            return (
              <li key={item.id} className="case-queue__item">
                <button
                  className={`case-queue-card${selected ? " case-queue-card--selected" : ""}`}
                  type="button"
                  aria-current={selected ? "true" : undefined}
                  onClick={() => onSelect(item.id)}
                >
                  <span className="case-queue-card__rank" aria-label={`Queue rank ${item.queue_rank}`}>
                    {item.queue_rank.toString().padStart(2, "0")}
                  </span>
                  <span className="case-queue-card__body">
                    <span className="case-queue-card__title-row">
                      <strong>{shortReviewId(item.review_id)}</strong>
                      <span className={`status-badge status-badge--${item.status}`}>
                        {caseStatusLabel(item.status)}
                      </span>
                    </span>
                    <span className="case-queue-card__meta">
                      <span>{formatDateTime(item.alert_hour)}</span>
                      <span aria-hidden="true">•</span>
                      <span>{presentation.label}</span>
                    </span>
                    <span className="case-queue-card__risk">
                      {presentation.showScore ? (
                        <>
                          <ShieldAlert aria-hidden="true" size={15} />
                          <span>Risk {formatPercent(item.risk_score)}</span>
                          <span className={`risk-dot risk-dot--${riskTone(item.risk_score)}`} aria-hidden="true" />
                        </>
                      ) : (
                        <span className="case-queue-card__protocol-note">Score withheld by review protocol</span>
                      )}
                    </span>
                  </span>
                  <ChevronRight className="case-queue-card__chevron" aria-hidden="true" size={18} />
                </button>
              </li>
            );
          })}
        </ol>
      ) : null}
    </section>
  );
}

function QueueSkeleton() {
  return (
    <div className="case-queue__skeleton" aria-label="Loading case queue" role="status">
      <span className="sr-only">Loading case queue</span>
      {[0, 1, 2, 3].map((index) => (
        <div className="case-queue__skeleton-row" key={index} aria-hidden="true">
          <span />
          <span />
        </div>
      ))}
    </div>
  );
}

function shortReviewId(reviewId: string) {
  if (reviewId.length <= 16) return reviewId;
  return `Review ${reviewId.slice(0, 8).toUpperCase()}`;
}

