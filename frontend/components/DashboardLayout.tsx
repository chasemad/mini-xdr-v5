"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Home,
  AlertTriangle,
  Activity,
  Brain,
  Search,
  BarChart,
  Workflow,
  Zap,
  Settings,
  Menu,
  X,
  LogOut,
  User,
  ChevronRight,
  Shield,
  Bot,
  Bell
} from "lucide-react";
import { useAuth } from "../app/contexts/AuthContext";
import { CopilotSidebar } from "./layout/CopilotSidebar";
import { useDashboard } from "../app/contexts/DashboardContext";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ElementType;
  roles?: string[]; // If undefined, available to all roles
}

const navigation: NavigationItem[] = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Incidents", href: "/incidents", icon: AlertTriangle },
  { name: "Honeypot", href: "/honeypot", icon: Shield, roles: ["analyst", "soc_lead", "admin"] },
  { name: "Agents", href: "/agents", icon: Activity, roles: ["analyst", "soc_lead", "admin"] },
  { name: "Threat Intel", href: "/intelligence", icon: Brain, roles: ["analyst", "soc_lead", "admin"] },
  { name: "Investigations", href: "/investigations", icon: Search, roles: ["analyst", "soc_lead", "admin"] },
  { name: "Analytics", href: "/analytics", icon: BarChart },
  { name: "Workflows", href: "/workflows", icon: Workflow, roles: ["soc_lead", "admin"] },
  { name: "Automations", href: "/automations", icon: Zap, roles: ["admin"] },
  { name: "Settings", href: "/settings", icon: Settings, roles: ["admin"] },
];

interface DashboardLayoutProps {
  children: React.ReactNode;
  breadcrumbs?: { label: string; href?: string }[];
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  breadcrumbs,
}) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const { user, organization, logout } = useAuth();
  const { isCopilotOpen, toggleCopilot, copilotContext } = useDashboard();
  const [telemetry, setTelemetry] = useState<{ hasLogs: boolean } | null>(null);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
        if (!token) return;
        const res = await fetch(`${API_BASE}/api/telemetry/status`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setTelemetry({ hasLogs: !!data.hasLogs });
        }
      } catch (e) {
        // Best-effort; ignore
      }
    };
    fetchTelemetry();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (userDropdownOpen && !target.closest('[data-user-dropdown]')) {
        setUserDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [userDropdownOpen]);

  const roleHierarchy: Record<string, number> = {
    viewer: 1,
    analyst: 2,
    soc_lead: 3,
    admin: 4,
  };

  const userRoleLevel = roleHierarchy[user?.role || "viewer"] || 1;

  const filteredNavigation = navigation.filter((item) => {
    if (!item.roles) return true;
    const minRequiredLevel = Math.min(
      ...item.roles.map((role) => roleHierarchy[role] || 99)
    );
    return userRoleLevel >= minRequiredLevel;
  });

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/80 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border overflow-y-auto",
          "transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:h-screen",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center gap-3 px-6 h-16 border-b border-border">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-bold tracking-tight">Mini-XDR</span>
            <button
              onClick={() => setSidebarOpen(false)}
              className="ml-auto lg:hidden text-muted-foreground hover:text-foreground"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Organization info */}
          {organization && (
            <div className="px-6 py-4">
              <div className="text-xs text-muted-foreground uppercase tracking-wider font-medium mb-1">
                Organization
              </div>
              <div className="text-sm font-medium truncate">
                {organization.name}
              </div>
            </div>
          )}

          {/* Navigation */}
          <nav className="flex-1 px-3 py-2 space-y-1 overflow-y-auto">
            {filteredNavigation.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || pathname.startsWith(item.href + "/");

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ease-in-out border border-transparent",
                    isActive
                      ? "bg-primary text-primary-foreground shadow-md shadow-primary/20"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-primary hover:border-primary/10 hover:shadow-lg hover:shadow-primary/10 hover:-translate-y-0.5"
                  )}
                  onClick={() => setSidebarOpen(false)}
                >
                  <Icon className="w-4 h-4" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* User menu */}
          <div className="border-t border-border p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                <User className="w-4 h-4 text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">
                  {user?.full_name || user?.email}
                </div>
                <div className="text-xs text-muted-foreground capitalize">{user?.role}</div>
              </div>
            </div>
            <Button
              variant="outline"
              className="w-full justify-start gap-2 text-muted-foreground"
              onClick={handleLogout}
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 bg-muted/10">
        {/* Top bar */}
        <header className="sticky top-0 z-30 h-16 bg-background/80 backdrop-blur-md border-b border-border px-4 lg:px-8 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden text-muted-foreground hover:text-foreground"
            >
              <Menu className="w-6 h-6" />
            </button>

            {breadcrumbs && breadcrumbs.length > 0 && (
              <nav className="hidden md:flex items-center gap-2 text-sm">
                {breadcrumbs.map((crumb, index) => (
                  <React.Fragment key={index}>
                    {index > 0 && (
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    )}
                    {crumb.href ? (
                      <Link
                        href={crumb.href}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {crumb.label}
                      </Link>
                    ) : (
                      <span className="font-medium text-foreground">
                        {crumb.label}
                      </span>
                    )}
                  </React.Fragment>
                ))}
              </nav>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="text-muted-foreground">
              <Bell className="w-5 h-5" />
            </Button>

            <Button
              variant="outline"
              size="sm"
              className={cn("gap-2", isCopilotOpen && "bg-accent text-accent-foreground")}
              onClick={toggleCopilot}
            >
              <Bot className="w-4 h-4" />
              <span className="hidden sm:inline">Copilot</span>
            </Button>

            {organization && organization.onboarding_status !== 'completed' && (
              <Button asChild variant="default" size="sm">
                <Link href="/onboarding">
                  Start setup
                </Link>
              </Button>
            )}
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-4 lg:p-8 overflow-y-auto">
          <div className="max-w-7xl mx-auto space-y-6">
             {children}
          </div>
        </main>
      </div>

      {/* Copilot Sidebar */}
      <CopilotSidebar
        isOpen={isCopilotOpen}
        onClose={toggleCopilot}
        selectedIncidentId={copilotContext?.incidentId}
        incidentData={copilotContext?.incidentData}
      />
    </div>
  );
};
