import { useEffect, useRef, useState } from "react";
import { Outlet } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

export function AppShell() {
  const { user } = useAuth();
  const [navigationOpen, setNavigationOpen] = useState(false);
  const openerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!navigationOpen) return undefined;
    openerRef.current = document.activeElement as HTMLElement;
    const sidebar = document.querySelector<HTMLElement>(".sidebar.is-open");
    const focusable = sidebar
      ? Array.from(
          sidebar.querySelectorAll<HTMLElement>(
            'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
          ),
        )
      : [];
    focusable[0]?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setNavigationOpen(false);
        return;
      }
      if (event.key !== "Tab" || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    document.body.classList.add("nav-open");
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("nav-open");
      openerRef.current?.focus();
    };
  }, [navigationOpen]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Sidebar open={navigationOpen} onClose={() => setNavigationOpen(false)} role={user?.role} />
      <div className="app-frame">
        <Topbar onOpenNavigation={() => setNavigationOpen(true)} />
        <main id="main-content" className="main-content" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
