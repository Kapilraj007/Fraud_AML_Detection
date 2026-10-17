import { FormEvent, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Filter,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { PageState } from "../../components/ui/PageState";
import { api, normalizeApiError } from "../../lib/api";
import { formatCurrency, formatDateTime, humanize } from "../../lib/format";
import "../dashboard/operations-shared.css";
import "./transactions.css";

type DateFilter = {
  after: string;
  before: string;
};

function asIsoDate(value: string): string | undefined {
  if (!value) return undefined;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

export function TransactionsPage() {
  const [draftDates, setDraftDates] = useState<DateFilter>({ after: "", before: "" });
  const [appliedDates, setAppliedDates] = useState<DateFilter>({ after: "", before: "" });
  const [dateError, setDateError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [pageSearch, setPageSearch] = useState("");
  const offset = page * pageSize;

  const transactionsQuery = useQuery({
    queryKey: ["transactions", { appliedDates, page, pageSize }],
    queryFn: () =>
      api.transactions.list({
        limit: pageSize,
        offset,
        event_after: asIsoDate(appliedDates.after),
        event_before: asIsoDate(appliedDates.before),
      }),
    staleTime: 15_000,
  });

  const transactions = useMemo(() => transactionsQuery.data ?? [], [transactionsQuery.data]);
  const filteredTransactions = useMemo(() => {
    const needle = pageSearch.trim().toLocaleLowerCase();
    if (!needle) return transactions;
    return transactions.filter((transaction) =>
      [
        transaction.event_id,
        transaction.masked_source_account,
        transaction.masked_destination_account,
        transaction.transaction_type,
        transaction.currency,
        transaction.dataset,
      ]
        .filter(Boolean)
        .some((value) => String(value).toLocaleLowerCase().includes(needle)),
    );
  }, [pageSearch, transactions]);

  const applyDates = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const after = asIsoDate(draftDates.after);
    const before = asIsoDate(draftDates.before);
    if (after && before && Date.parse(after) > Date.parse(before)) {
      setDateError("The start of the event window must be before its end.");
      return;
    }
    setDateError(null);
    setAppliedDates(draftDates);
    setPage(0);
  };

  const clearDates = () => {
    const empty = { after: "", before: "" };
    setDraftDates(empty);
    setAppliedDates(empty);
    setDateError(null);
    setPage(0);
  };

  const hasDateFilters = Boolean(appliedDates.after || appliedDates.before);
  const hasNextPage = transactions.length === pageSize;
  const showingStart = transactions.length > 0 ? offset + 1 : 0;
  const showingEnd = offset + transactions.length;

  return (
    <div className="page transactions-page">
      <header className="page-header">
        <div>
          <p className="page-eyebrow">Governed activity</p>
          <h2 className="page-title">Transactions</h2>
          <p className="page-subtitle">
            Inspect masked transaction records in reverse chronological order without exposing account identifiers.
          </p>
        </div>
        <div className="page-header-actions">
          <span className="current-view-note">
            API rows {showingStart}–{showingEnd}
          </span>
          <button
            className="button button-secondary"
            type="button"
            onClick={() => void transactionsQuery.refetch()}
            disabled={transactionsQuery.isFetching}
          >
            <RefreshCw
              className={transactionsQuery.isFetching ? "spin" : ""}
              size={16}
              aria-hidden="true"
            />
            {transactionsQuery.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <section className="panel transaction-filter-panel" aria-labelledby="transaction-filters-title">
        <div className="panel-header filters-header">
          <div>
            <p className="panel-eyebrow">Server filters</p>
            <h3 id="transaction-filters-title">Event window</h3>
            <p>Date filters are sent to the transaction API. Times are interpreted in your local timezone.</p>
          </div>
          <Filter size={20} aria-hidden="true" />
        </div>
        <form className="filters transaction-filters" onSubmit={applyDates}>
          <div className="form-field">
            <label htmlFor="transaction-after">From</label>
            <input
              id="transaction-after"
              type="datetime-local"
              value={draftDates.after}
              onChange={(event) =>
                setDraftDates((current) => ({ ...current, after: event.target.value }))
              }
            />
          </div>
          <div className="form-field">
            <label htmlFor="transaction-before">To</label>
            <input
              id="transaction-before"
              type="datetime-local"
              value={draftDates.before}
              onChange={(event) =>
                setDraftDates((current) => ({ ...current, before: event.target.value }))
              }
            />
          </div>
          <button className="button button-primary" type="submit">Apply window</button>
          {hasDateFilters ? (
            <button className="button button-ghost" type="button" onClick={clearDates}>
              <X size={16} aria-hidden="true" /> Clear
            </button>
          ) : null}
        </form>
        {dateError ? <p className="field-error" role="alert">{dateError}</p> : null}
      </section>

      <section className="panel transaction-table-panel" aria-labelledby="transaction-table-title">
        <div className="panel-header transaction-table-header">
          <div>
            <p className="panel-eyebrow">Current API page</p>
            <h3 id="transaction-table-title">Transaction ledger</h3>
            <p>Page search narrows only the rows already loaded below.</p>
          </div>
          <div className="table-tools">
            <label className="table-search" htmlFor="transaction-page-search">
              <Search size={16} aria-hidden="true" />
              <span className="sr-only">Search the current transaction page</span>
              <input
                id="transaction-page-search"
                type="search"
                value={pageSearch}
                placeholder="Search this page"
                onChange={(event) => setPageSearch(event.target.value)}
              />
            </label>
            <label className="page-size-control" htmlFor="transaction-page-size">
              <span>Rows</span>
              <select
                id="transaction-page-size"
                value={pageSize}
                onChange={(event) => {
                  setPageSize(Number(event.target.value));
                  setPage(0);
                }}
              >
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </label>
          </div>
        </div>

        {transactionsQuery.isPending ? (
          <PageState
            kind="loading"
            title="Loading transactions"
            message="Retrieving the selected event window."
          />
        ) : transactionsQuery.isError ? (
          <PageState
            kind="error"
            title="Transactions unavailable"
            message={normalizeApiError(transactionsQuery.error)}
            onRetry={() => void transactionsQuery.refetch()}
          />
        ) : transactions.length === 0 ? (
          <PageState
            kind="empty"
            title={hasDateFilters ? "No transactions in this event window" : "No transactions available"}
            message={
              hasDateFilters
                ? "Clear or widen the server-side event window to continue."
                : "Transaction rows will appear after records are ingested."
            }
          />
        ) : filteredTransactions.length === 0 ? (
          <div className="search-empty-state">
            <PageState
              kind="empty"
              title="No matches on this page"
              message="The page search does not search rows that have not been loaded."
            />
            <button className="button button-secondary" type="button" onClick={() => setPageSearch("")}>
              Clear page search
            </button>
          </div>
        ) : (
          <div className="data-table-wrap">
            <table className="data-table transaction-data-table">
              <caption className="sr-only">
                Masked transactions {showingStart} through {showingEnd} from the API
              </caption>
              <thead>
                <tr>
                  <th scope="col">Event time</th>
                  <th scope="col">Event</th>
                  <th scope="col">Account route</th>
                  <th scope="col">Type</th>
                  <th scope="col">Amount</th>
                  <th scope="col">Dataset</th>
                </tr>
              </thead>
              <tbody>
                {filteredTransactions.map((transaction) => (
                  <tr key={transaction.id}>
                    <td>{formatDateTime(transaction.event_time)}</td>
                    <td><code className="event-id">{transaction.event_id}</code></td>
                    <td>
                      <span className="account-route">
                        <span>{transaction.masked_source_account ?? "Unavailable"}</span>
                        <span aria-hidden="true">→</span>
                        <span>{transaction.masked_destination_account ?? "Unavailable"}</span>
                      </span>
                    </td>
                    <td><span className="type-chip">{humanize(transaction.transaction_type)}</span></td>
                    <td><strong>{formatCurrency(transaction.amount, transaction.currency)}</strong></td>
                    <td>{humanize(transaction.dataset)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!transactionsQuery.isError && transactions.length > 0 ? (
          <nav className="pagination" aria-label="Transaction pages">
            <p>
              Showing API rows <strong>{showingStart}–{showingEnd}</strong>
              {pageSearch ? ` · ${filteredTransactions.length} match this page` : ""}
            </p>
            <div>
              <button
                className="button button-secondary button-compact"
                type="button"
                disabled={page === 0 || transactionsQuery.isFetching}
                onClick={() => setPage((current) => Math.max(0, current - 1))}
              >
                <ChevronLeft size={16} aria-hidden="true" /> Previous
              </button>
              <span aria-current="page">Page {page + 1}</span>
              <button
                className="button button-secondary button-compact"
                type="button"
                disabled={!hasNextPage || transactionsQuery.isFetching}
                onClick={() => setPage((current) => current + 1)}
              >
                Next <ChevronRight size={16} aria-hidden="true" />
              </button>
            </div>
          </nav>
        ) : null}
      </section>
    </div>
  );
}
