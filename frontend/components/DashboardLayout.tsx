"use client";

import React, { useEffect, useState } from "react";
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
} from "lucide-react";
import { useAuth } from "../app/contexts/AuthContext";

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
  const pathname = usePathname();
  const router = useRouter();
  const { user, organization, logout } = useAuth();
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

  // Filter navigation based on user role
  const roleHierarchy: Record<string, number> = {
    viewer: 1,
    analyst: 2,
    soc_lead: 3,
    admin: 4,
  };

  const userRoleLevel = roleHierarchy[user?.role || "viewer"] || 1;

  const filteredNavigation = navigation.filter((item) => {
    if (!item.roles) return true; // Available to all
    
    // Check if user's role meets the minimum requirement
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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-64 bg-gray-900/95 backdrop-blur-sm border-r border-gray-800
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-gray-800">
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">Mini-XDR</span>
            </Link>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Organization info */}
          {organization && (
            <div className="px-6 py-4 border-b border-gray-800">
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
                Organization
              </div>
              <div className="text-sm font-medium text-gray-200 truncate">
                {organization.name}
              </div>
            </div>
          )}

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {filteredNavigation.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || pathname.startsWith(item.href + "/");

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`
                    flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                    transition-colors duration-150
                    ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : "text-gray-400 hover:text-white hover:bg-gray-800"
                    }
                  `}
                  onClick={() => setSidebarOpen(false)}
                >
                  <Icon className="w-5 h-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* User menu */}
          <div className="border-t border-gray-800 p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-white truncate">
                  {user?.full_name || user?.email}
                </div>
                <div className="text-xs text-gray-400 capitalize">{user?.role}</div>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <header className="sticky top-0 z-30 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800">
          <div className="flex items-center justify-between px-4 py-4 lg:px-6">
            <div className="flex items-center gap-4">
              {/* Mobile menu button */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden text-gray-400 hover:text-white"
              >
                <Menu className="w-6 h-6" />
              </button>

              {/* Breadcrumbs */}
              {breadcrumbs && breadcrumbs.length > 0 && (
                <nav className="flex items-center gap-2 text-sm">
                  {breadcrumbs.map((crumb, index) => (
                    <React.Fragment key={index}>
                      {index > 0 && (
                        <ChevronRight className="w-4 h-4 text-gray-600" />
                      )}
                      {crumb.href ? (
                        <Link
                          href={crumb.href}
                          className="text-gray-400 hover:text-white transition-colors"
                        >
                          {crumb.label}
                        </Link>
                      ) : (
                        <span className="text-gray-200 font-medium">
                          {crumb.label}
                        </span>
                      )}
                    </React.Fragment>
                  ))}
                </nav>
              )}
            </div>

            {/* Top bar actions */}
            <div className="flex items-center gap-3">
              {organization && organization.onboarding_status !== 'completed' && (
                <Link
                  href="/onboarding"
                  className="inline-flex items-center rounded-lg px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-sm font-medium"
                >
                  Start setup
                </Link>
              )}
              {organization && organization.onboarding_status === 'completed' && telemetry && !telemetry.hasLogs && (
                <button
                  className="inline-flex items-center rounded-lg px-3 py-2 bg-gray-700 text-gray-200 cursor-default text-sm"
                  title="Waiting for first events"
                >
                  Awaiting data
                </button>
              )}
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6">{children}</main>
      </div>
    </div>
  );
};

function Shield(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}


