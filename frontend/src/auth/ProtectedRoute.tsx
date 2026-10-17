import type { ReactNode } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import type { UserRole } from "../lib/types";
import { useAuth } from "./AuthContext";

export interface ProtectedRouteProps {
  children?: ReactNode;
  allowedRoles?: readonly UserRole[];
}

export function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="route-loader" role="status" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        <span>Restoring secure session…</span>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    return <Navigate to="/" replace />;
  }

  return children ?? <Outlet />;
}
