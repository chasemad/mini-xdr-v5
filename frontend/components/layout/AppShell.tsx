'use client'

import React, { useState } from 'react'
import {
  Menu, Search, PanelRight, Shield,
  LayoutDashboard, AlertTriangle, Brain, Target, Bot,
  BarChart3, Globe, Settings, Search as SearchIcon,
  Zap, Workflow, Plug
} from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { useFeature } from '@/lib/flags'
import { Button } from '@/components/ui/button'
import { CopilotDock } from '@/components/copilot/CopilotDock'
import { CommandPalette } from '@/components/system/CommandPalette'

interface AppShellProps {
  children: React.ReactNode
  breadcrumbs?: { label: string; href?: string }[]
}

function AppShell({ children, breadcrumbs }: AppShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [dockOpen, setDockOpen] = useState(true)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)

  // Check if V2 features are enabled
  const showCopilotDock = useFeature('COPILOT_DOCK')
  const showCommandPalette = useFeature('COMMAND_PALETTE')

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K / Ctrl+K for command palette
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCommandPaletteOpen(true)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [])

  return (
    <div className="bg-bg text-text flex min-h-screen">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        "fixed inset-y-0 left-0 z-50 w-64 shrink-0 border-r border-border bg-surface-0 shadow-lg transform transition-transform duration-200 ease-in-out",
        sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
      )}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-between px-6 py-6 border-b border-border">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 bg-gradient-to-br from-primary to-primary/80 rounded-xl flex items-center justify-center shadow-lg">
                <Shield className="w-5 h-5 text-bg" />
              </div>
              <span className="text-xl font-bold text-text group-hover:text-primary transition-colors">
                Mini-XDR
              </span>
            </Link>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden w-8 h-8 rounded-lg flex items-center justify-center text-text-muted hover:text-text hover:bg-surface-1 transition-colors"
            >
              <Menu className="w-4 h-4 rotate-90" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 overflow-y-auto">
            <div className="space-y-1">
              {[
                { href: '/', label: 'Dashboard', icon: LayoutDashboard },
                { href: '/incidents', label: 'Incidents', icon: AlertTriangle },
                { href: '/intelligence', label: 'Intelligence', icon: Brain },
                { href: '/hunt', label: 'Threat Hunting', icon: Target },
                { href: '/agents', label: 'AI Agents', icon: Bot },
                { href: '/analytics', label: 'Analytics', icon: BarChart3 },
                { href: '/visualizations', label: 'Visualizations', icon: Globe },
                { href: '/investigations', label: 'Investigations', icon: SearchIcon },
                { href: '/automations', label: 'Automations', icon: Zap },
                { href: '/workflows', label: 'Workflows', icon: Workflow },
                { href: '/integrations', label: 'Integrations', icon: Plug },
                { href: '/settings', label: 'Settings', icon: Settings },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-lg transition-colors",
                      "hover:bg-surface-1 hover:text-text",
                      "text-text-muted"
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </nav>

          {/* User menu */}
          <div className="border-t border-border p-4 bg-surface-1/50">
            <div className="flex items-center gap-3 px-3 py-2">
              <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center">
                <span className="text-sm font-medium text-primary">U</span>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-text truncate">User</div>
                <div className="text-xs text-text-muted truncate">user@company.com</div>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col lg:ml-64">
        {/* Top bar */}
        <header className="h-14 border-b border-border flex items-center gap-3 px-4 bg-surface-0 shadow-sm">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden inline-flex items-center justify-center w-10 h-10 rounded-lg text-text-muted hover:text-text hover:bg-surface-1 transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="font-semibold text-text">Mini-XDR</div>

          {/* Breadcrumbs */}
          {breadcrumbs && breadcrumbs.length > 0 && (
            <nav className="flex items-center gap-2 text-sm ml-6">
              {breadcrumbs.map((crumb, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <span className="text-text-muted">/</span>}
                  {crumb.href ? (
                    <Link
                      href={crumb.href}
                      className="text-text-muted hover:text-text transition-colors"
                    >
                      {crumb.label}
                    </Link>
                  ) : (
                    <span className="text-text font-medium">{crumb.label}</span>
                  )}
                </React.Fragment>
              ))}
            </nav>
          )}

          <div className="ml-auto flex items-center gap-2">
            {showCommandPalette && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCommandPaletteOpen(true)}
                className="inline-flex items-center gap-2 text-text-muted hover:text-text"
              >
                <Search className="w-4 h-4" />
                <span className="hidden sm:inline">Search</span>
              </Button>
            )}

            {showCopilotDock && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setDockOpen(!dockOpen)}
                className={cn(
                  "text-text-muted hover:text-text",
                  dockOpen && "text-primary"
                )}
              >
                <PanelRight className="w-5 h-5" />
              </Button>
            )}
          </div>
        </header>

        {/* Content area */}
        <div className="flex-1 grid grid-cols-12">
          <main className={cn(
            "col-span-12 transition-all duration-200",
            dockOpen && showCopilotDock && "xl:col-span-9"
          )}>
            {children}
          </main>

          {/* Copilot Dock */}
          {showCopilotDock && dockOpen && (
            <aside className="hidden xl:block col-span-3 border-l border-border bg-surface-0">
              <CopilotDock />
            </aside>
          )}
        </div>
      </div>

      {/* Command Palette */}
      {showCommandPalette && commandPaletteOpen && (
        <CommandPalette onClose={() => setCommandPaletteOpen(false)} />
      )}
    </div>
  )
}

export default AppShell
