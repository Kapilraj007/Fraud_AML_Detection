import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  api,
  AUTH_SESSION_EXPIRED_EVENT,
  tokenStore,
} from "../lib/api";
import type { User } from "../lib/types";

export interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<User>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(() => tokenStore.get() !== null);
  const operation = useRef(0);

  useEffect(() => {
    let active = true;
    const currentOperation = ++operation.current;

    const restoreSession = async () => {
      if (!tokenStore.get()) {
        if (active) setIsLoading(false);
        return;
      }

      try {
        const currentUser = await api.getCurrentUser();
        if (active && operation.current === currentOperation) setUser(currentUser);
      } catch {
        tokenStore.clear();
        if (active && operation.current === currentOperation) setUser(null);
      } finally {
        if (active && operation.current === currentOperation) setIsLoading(false);
      }
    };

    const handleExpiredSession = () => {
      ++operation.current;
      setUser(null);
      setIsLoading(false);
    };

    window.addEventListener(AUTH_SESSION_EXPIRED_EVENT, handleExpiredSession);
    void restoreSession();

    return () => {
      active = false;
      window.removeEventListener(AUTH_SESSION_EXPIRED_EVENT, handleExpiredSession);
    };
  }, []);

  const login = useCallback(async (email: string, password: string): Promise<User> => {
    const currentOperation = ++operation.current;
    setIsLoading(true);
    try {
      await api.login({ email: email.trim(), password });
      const currentUser = await api.getCurrentUser();
      if (operation.current === currentOperation) setUser(currentUser);
      return currentUser;
    } catch (error) {
      tokenStore.clear();
      if (operation.current === currentOperation) setUser(null);
      throw error;
    } finally {
      if (operation.current === currentOperation) setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async (): Promise<void> => {
    ++operation.current;
    setUser(null);
    setIsLoading(false);
    try {
      await api.logout();
    } catch {
      // The API helper always clears local credentials; remote revocation is best effort.
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: user !== null,
      isLoading,
      login,
      logout,
    }),
    [isLoading, login, logout, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// The provider and its companion hook intentionally share one module.
// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
}
