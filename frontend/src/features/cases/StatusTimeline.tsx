import { Check, Circle, Clock3, Flag, UserRoundCheck } from "lucide-react";

import { formatDateTime } from "../../lib/format";
import type { CaseDetail } from "../../lib/types";

interface StatusTimelineProps {
  caseDetail: CaseDetail;
}

const ORDER = ["open", "claimed", "in_review", "resolved"] as const;

export function StatusTimeline({ caseDetail }: StatusTimelineProps) {
  const terminal = ["escalated", "dismissed", "more_info_requested", "closed"].includes(caseDetail.status);
  const activeIndex = terminal
    ? 3
    : caseDetail.status === "in_review"
      ? 2
      : caseDetail.status === "claimed"
        ? 1
        : 0;
  const disposition = caseDetail.dispositions.at(-1);
  const resolvedLabel = caseDetail.status === "more_info_requested"
    ? "More information requested"
    : caseDetail.status === "escalated"
      ? "Escalated"
      : caseDetail.status === "dismissed"
        ? "Dismissed"
        : "Resolution";

  const items = [
    { key: ORDER[0], label: "Case opened", date: caseDetail.created_at, icon: Flag },
    { key: ORDER[1], label: "Claimed", date: activeIndex >= 1 ? undefined : undefined, icon: UserRoundCheck },
    { key: ORDER[2], label: "Review started", date: activeIndex >= 2 ? undefined : undefined, icon: Clock3 },
    { key: ORDER[3], label: resolvedLabel, date: disposition?.created_at, icon: Check },
  ];

  return (
    <section className="case-section status-timeline" aria-labelledby="status-timeline-heading">
      <header className="case-section__header">
        <span className="case-section__icon"><Clock3 aria-hidden="true" size={19} /></span>
        <div>
          <p className="case-section__eyebrow">Workflow</p>
          <h3 id="status-timeline-heading">Case timeline</h3>
        </div>
      </header>
      <ol className="status-timeline__list">
        {items.map((item, index) => {
          const complete = index < activeIndex || (terminal && index === activeIndex);
          const current = !terminal && index === activeIndex;
          const Icon = complete || current ? item.icon : Circle;
          return (
            <li
              className={`status-timeline__item${complete ? " is-complete" : ""}${current ? " is-current" : ""}`}
              key={item.key}
              aria-current={current ? "step" : undefined}
            >
              <span className="status-timeline__marker"><Icon aria-hidden="true" size={15} /></span>
              <span>
                <strong>{item.label}</strong>
                <small>{item.date ? formatDateTime(item.date) : complete || current ? "Timestamp unavailable" : "Pending"}</small>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

