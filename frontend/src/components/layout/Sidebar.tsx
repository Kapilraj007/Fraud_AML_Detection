import { NavLink } from "react-router-dom";
import {
  Activity,
  BriefcaseBusiness,
  LayoutDashboard,
  ReceiptText,
  ShieldCheck,
  X,
} from "lucide-react";
import type { UserRole } from "../../lib/types";

type SidebarProps = {
  open: boolean;
  onClose: () => void;
  role?: UserRole;
};

const navigation = [
  { to: "/", label: "Command center", icon: LayoutDashboard, end: true },
  { to: "/cases", label: "Investigation queue", icon: BriefcaseBusiness },
  { to: "/monitoring", label: "Risk monitoring", icon: Activity },
  { to: "/transactions", label: "Transactions", icon: ReceiptText },
];

export function Sidebar({ open, onClose, role }: SidebarProps) {
  return (
    <>
      <button
        className={`sidebar-scrim${open ? " is-open" : ""}`}
        type="button"
        aria-label="Close navigation"
        onClick={onClose}
        tabIndex={open ? 0 : -1}
      />
      <aside className={`sidebar${open ? " is-open" : ""}`} aria-label="Primary navigation">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            <ShieldCheck size={22} strokeWidth={2.1} />
          </span>
          <span>
            <strong>Sentinel AML</strong>
            <small>Financial crime operations</small>
          </span>
          <button className="icon-button sidebar-close" type="button" onClick={onClose}>
            <X size={20} />
            <span className="sr-only">Close navigation</span>
          </button>
        </div>

        <div className="workspace-label">
          <span className="workspace-dot" aria-hidden="true" />
          <span>
            <small>Workspace</small>
            <strong>AML operations</strong>
          </span>
        </div>

        <nav className="sidebar-nav">
          <p className="sidebar-kicker">Operations</p>
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) => `nav-link${isActive ? " is-active" : ""}`}
            >
              <Icon size={19} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}

          {role === "admin" ? (
            <>
              <p className="sidebar-kicker sidebar-kicker-spaced">Control</p>
              <NavLink
                to="/governance"
                onClick={onClose}
                className={({ isActive }) => `nav-link${isActive ? " is-active" : ""}`}
              >
                <ShieldCheck size={19} strokeWidth={1.8} aria-hidden="true" />
                <span>Governance & audit</span>
              </NavLink>
            </>
          ) : null}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-footer-icon" aria-hidden="true">
            <ShieldCheck size={17} />
          </span>
          <span>
            <strong>Decision support only</strong>
            <small>Human approval is always required</small>
          </span>
        </div>
      </aside>
    </>
  );
}
