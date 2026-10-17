import { Building2, LogOut, Menu } from "lucide-react";
import { useLocation } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { humanizeRole } from "../../lib/format";

type TopbarProps = {
  onOpenNavigation: () => void;
};

function pageTitle(pathname: string) {
  if (pathname.startsWith("/cases")) return "Investigation queue";
  if (pathname.startsWith("/monitoring")) return "Risk monitoring";
  if (pathname.startsWith("/transactions")) return "Transactions";
  if (pathname.startsWith("/governance")) return "Governance & audit";
  return "Command center";
}

export function Topbar({ onOpenNavigation }: TopbarProps) {
  const location = useLocation();
  const { user, logout } = useAuth();
  const displayName = user?.full_name || user?.email || "Analyst";
  const initials = displayName
    .split(/\s|@/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");

  return (
    <header className="topbar">
      <div className="topbar-heading">
        <button className="icon-button menu-button" type="button" onClick={onOpenNavigation}>
          <Menu size={21} />
          <span className="sr-only">Open navigation</span>
        </button>
        <div>
          <p>Financial crime operations</p>
          <h1>{pageTitle(location.pathname)}</h1>
        </div>
      </div>

      <div className="topbar-actions">
        <div className="topbar-context" aria-label="Single institution workspace">
          <Building2 size={16} aria-hidden="true" />
          <span>Single-institution workspace</span>
        </div>

        <div className="user-menu">
          <span className="avatar" aria-hidden="true">{initials || "A"}</span>
          <span className="user-copy">
            <strong>{displayName}</strong>
            <small>{humanizeRole(user?.role)}</small>
          </span>
          <button className="icon-button" type="button" onClick={() => void logout()}>
            <LogOut size={18} />
            <span className="sr-only">Sign out</span>
          </button>
        </div>
      </div>
    </header>
  );
}
