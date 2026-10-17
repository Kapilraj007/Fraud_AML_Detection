import { ListFilter, Search, UserRoundCheck, X } from "lucide-react";

import type { CaseStatus } from "../../lib/types";

export interface CaseFilterValue {
  search: string;
  status: CaseStatus | "all";
  assignedToMe: boolean;
}

interface CaseFiltersProps {
  value: CaseFilterValue;
  resultCount: number;
  onChange: (next: CaseFilterValue) => void;
}

const STATUS_OPTIONS: Array<{ value: CaseStatus | "all"; label: string }> = [
  { value: "all", label: "All statuses" },
  { value: "open", label: "Open" },
  { value: "claimed", label: "Claimed" },
  { value: "in_review", label: "In review" },
  { value: "more_info_requested", label: "More info requested" },
  { value: "escalated", label: "Escalated" },
  { value: "dismissed", label: "Dismissed" },
];

export function CaseFilters({ value, resultCount, onChange }: CaseFiltersProps) {
  const isFiltered = value.search !== "" || value.status !== "all" || value.assignedToMe;

  return (
    <section className="case-filters" aria-labelledby="case-filter-heading">
      <div className="case-filters__heading-row">
        <div className="case-filters__heading">
          <ListFilter aria-hidden="true" size={18} />
          <h2 id="case-filter-heading">Queue filters</h2>
        </div>
        <span className="case-filters__count" aria-live="polite">
          {resultCount} {resultCount === 1 ? "case" : "cases"}
        </span>
      </div>

      <div className="case-filters__controls">
        <label className="case-filters__search">
          <span className="sr-only">Search cases</span>
          <Search aria-hidden="true" size={17} />
          <input
            type="search"
            value={value.search}
            placeholder="Search review ID"
            autoComplete="off"
            onChange={(event) => onChange({ ...value, search: event.target.value })}
          />
        </label>

        <label className="case-filters__select">
          <span className="sr-only">Filter by status</span>
          <select
            value={value.status}
            onChange={(event) =>
              onChange({
                ...value,
                status: event.target.value as CaseStatus | "all",
              })
            }
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="case-filters__mine">
          <input
            type="checkbox"
            checked={value.assignedToMe}
            onChange={(event) =>
              onChange({ ...value, assignedToMe: event.target.checked })
            }
          />
          <UserRoundCheck aria-hidden="true" size={17} />
          Assigned to me
        </label>

        {isFiltered ? (
          <button
            className="case-filters__clear"
            type="button"
            onClick={() => onChange({ search: "", status: "all", assignedToMe: false })}
          >
            <X aria-hidden="true" size={16} />
            Clear
          </button>
        ) : null}
      </div>
    </section>
  );
}

