"use client";

import * as React from "react";
import { createContext, useContext, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { X, CheckCircle, XCircle, AlertTriangle, Info, Loader2 } from "lucide-react";

// Toast types
export type ToastType = "success" | "error" | "warning" | "info" | "loading";

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number;
  dismissible?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, "id">) => string;
  removeToast: (id: string) => void;
  updateToast: (id: string, toast: Partial<Toast>) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}

// Helper functions
export function toast(options: Omit<Toast, "id">): string {
  // This will be set by the provider
  if (typeof window !== "undefined" && (window as any).__toastAdd) {
    return (window as any).__toastAdd(options);
  }
  console.warn("Toast provider not initialized");
  return "";
}

toast.success = (title: string, description?: string) =>
  toast({ type: "success", title, description, duration: 4000 });

toast.error = (title: string, description?: string) =>
  toast({ type: "error", title, description, duration: 6000 });

toast.warning = (title: string, description?: string) =>
  toast({ type: "warning", title, description, duration: 5000 });

toast.info = (title: string, description?: string) =>
  toast({ type: "info", title, description, duration: 4000 });

toast.loading = (title: string, description?: string) =>
  toast({ type: "loading", title, description, duration: 0, dismissible: false });

toast.dismiss = (id: string) => {
  if (typeof window !== "undefined" && (window as any).__toastRemove) {
    (window as any).__toastRemove(id);
  }
};

toast.update = (id: string, options: Partial<Toast>) => {
  if (typeof window !== "undefined" && (window as any).__toastUpdate) {
    (window as any).__toastUpdate(id, options);
  }
};

// Provider component
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((toast: Omit<Toast, "id">) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const newToast: Toast = {
      ...toast,
      id,
      dismissible: toast.dismissible ?? true,
      duration: toast.duration ?? 4000,
    };

    setToasts((prev) => [...prev, newToast]);

    // Auto dismiss if duration > 0
    if (newToast.duration && newToast.duration > 0) {
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, newToast.duration);
    }

    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const updateToast = useCallback((id: string, updates: Partial<Toast>) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t))
    );
  }, []);

  // Expose to global toast function
  React.useEffect(() => {
    if (typeof window !== "undefined") {
      (window as any).__toastAdd = addToast;
      (window as any).__toastRemove = removeToast;
      (window as any).__toastUpdate = updateToast;
    }
    return () => {
      if (typeof window !== "undefined") {
        delete (window as any).__toastAdd;
        delete (window as any).__toastRemove;
        delete (window as any).__toastUpdate;
      }
    };
  }, [addToast, removeToast, updateToast]);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, updateToast }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}

// Toast container component
function ToastContainer({
  toasts,
  removeToast,
}: {
  toasts: Toast[];
  removeToast: (id: string) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-md w-full pointer-events-none">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={() => removeToast(toast.id)} />
      ))}
    </div>
  );
}

// Individual toast component
function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: () => void;
}) {
  const icons = {
    success: CheckCircle,
    error: XCircle,
    warning: AlertTriangle,
    info: Info,
    loading: Loader2,
  };

  const colors = {
    success: "bg-green-50 dark:bg-green-950/50 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200",
    error: "bg-red-50 dark:bg-red-950/50 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200",
    warning: "bg-yellow-50 dark:bg-yellow-950/50 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-200",
    info: "bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-200",
    loading: "bg-slate-50 dark:bg-slate-950/50 border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200",
  };

  const iconColors = {
    success: "text-green-500",
    error: "text-red-500",
    warning: "text-yellow-500",
    info: "text-blue-500",
    loading: "text-slate-500",
  };

  const Icon = icons[toast.type];

  return (
    <div
      className={cn(
        "pointer-events-auto flex items-start gap-3 p-4 rounded-lg border shadow-lg backdrop-blur-sm animate-in slide-in-from-right-full duration-300",
        colors[toast.type]
      )}
    >
      <Icon
        className={cn(
          "w-5 h-5 mt-0.5 shrink-0",
          iconColors[toast.type],
          toast.type === "loading" && "animate-spin"
        )}
      />
      <div className="flex-1 min-w-0">
        <p className="font-medium text-sm">{toast.title}</p>
        {toast.description && (
          <p className="text-xs mt-0.5 opacity-80">{toast.description}</p>
        )}
        {toast.action && (
          <button
            onClick={toast.action.onClick}
            className="mt-2 text-xs font-medium underline underline-offset-2 hover:no-underline"
          >
            {toast.action.label}
          </button>
        )}
      </div>
      {toast.dismissible && (
        <button
          onClick={onDismiss}
          className="shrink-0 opacity-50 hover:opacity-100 transition-opacity"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
