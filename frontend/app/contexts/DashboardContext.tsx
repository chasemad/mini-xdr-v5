"use client";

import React, { createContext, useContext, useState, ReactNode } from 'react';

interface DashboardContextType {
  isCopilotOpen: boolean;
  toggleCopilot: () => void;
  setCopilotContext: (context: any) => void;
  copilotContext: any;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [copilotContext, setCopilotContext] = useState<any>(null);

  const toggleCopilot = () => setIsCopilotOpen(prev => !prev);

  return (
    <DashboardContext.Provider value={{ isCopilotOpen, toggleCopilot, setCopilotContext, copilotContext }}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
}
