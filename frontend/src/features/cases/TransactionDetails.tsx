import { ArrowRight, CreditCard, Database, Landmark } from "lucide-react";

import { formatCurrency, formatDateTime, humanize } from "../../lib/format";
import type { Transaction } from "../../lib/types";

interface TransactionDetailsProps {
  transactions: Transaction[];
}

export function TransactionDetails({ transactions }: TransactionDetailsProps) {
  return (
    <section className="case-section transaction-details" aria-labelledby="transaction-heading">
      <header className="case-section__header">
        <span className="case-section__icon"><CreditCard aria-hidden="true" size={19} /></span>
        <div>
          <p className="case-section__eyebrow">Primary evidence</p>
          <h3 id="transaction-heading">
            {transactions.length === 1 ? "Transaction details" : "Linked transactions"}
          </h3>
        </div>
        <span className="case-section__count">{transactions.length}</span>
      </header>

      {transactions.length === 0 ? (
        <div className="case-empty-state">
          <Database aria-hidden="true" size={21} />
          <p>No transaction records are attached to this case.</p>
        </div>
      ) : (
        <div className="transaction-details__list">
          {transactions.map((transaction) => (
            <article className="transaction-card" key={transaction.id}>
              <div className="transaction-card__topline">
                <div>
                  <span className="transaction-card__event-label">Event ID</span>
                  <strong>{transaction.event_id}</strong>
                </div>
                <span className="transaction-card__type">{humanize(transaction.transaction_type)}</span>
              </div>

              <div className="transaction-card__amount">
                {formatCurrency(Number(transaction.amount), transaction.currency)}
              </div>

              <div className="transaction-card__route" aria-label="Masked account transfer route">
                <span>
                  <Landmark aria-hidden="true" size={16} />
                  <span>
                    <small>Source</small>
                    <strong>{transaction.masked_source_account ?? "Not available"}</strong>
                  </span>
                </span>
                <ArrowRight aria-hidden="true" size={18} />
                <span>
                  <Landmark aria-hidden="true" size={16} />
                  <span>
                    <small>Destination</small>
                    <strong>{transaction.masked_destination_account ?? "Not available"}</strong>
                  </span>
                </span>
              </div>

              <dl className="transaction-card__metadata">
                <div>
                  <dt>Occurred</dt>
                  <dd>{formatDateTime(transaction.event_time)}</dd>
                </div>
                <div>
                  <dt>Dataset</dt>
                  <dd>{transaction.dataset}</dd>
                </div>
                <div>
                  <dt>Currency</dt>
                  <dd>{transaction.currency}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

