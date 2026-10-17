import { AlertTriangle, Inbox, LoaderCircle, RefreshCw } from "lucide-react";

type PageStateProps = {
  kind: "loading" | "error" | "empty";
  title?: string;
  message?: string;
  onRetry?: () => void;
};

export function PageState({ kind, title, message, onRetry }: PageStateProps) {
  const Icon = kind === "loading" ? LoaderCircle : kind === "error" ? AlertTriangle : Inbox;
  const fallbackTitle = kind === "loading" ? "Loading workspace" : kind === "error" ? "We could not load this view" : "Nothing to review";

  return (
    <div className={`page-state page-state-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span className="page-state-icon">
        <Icon className={kind === "loading" ? "spin" : ""} size={22} aria-hidden="true" />
      </span>
      <div>
        <h2>{title || fallbackTitle}</h2>
        {message ? <p>{message}</p> : null}
        {onRetry ? (
          <button className="button button-secondary button-compact" type="button" onClick={onRetry}>
            <RefreshCw size={15} aria-hidden="true" /> Retry
          </button>
        ) : null}
      </div>
    </div>
  );
}
