"use client";

import React from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { BrainCircuit, Shield, Users, Activity } from "lucide-react";

import { OverviewTab } from "../incident-v2/OverviewTab";
import { CouncilAnalysisTab } from "../incident-v2/CouncilAnalysisTab";
import { AIAgentsTab } from "../incident-v2/AIAgentsTab";
import { ComprehensiveAnalysisTab } from "../incident-v2/ComprehensiveAnalysisTab";

interface GateResult {
  gate: string;
  verdict: string;
  reason: string;
  confidence_modifier?: number;
  processing_time_ms?: number;
  details?: Record<string, any>;
}

interface DeepAnalysisSheetProps {
  isOpen: boolean;
  onClose: () => void;
  incident: any;
  coordination: any;
  coordinationLoading: boolean;
  onRefreshIncident?: () => void;
  // Detection metadata props
  gateResults?: GateResult[];
  escalationReasons?: string[];
  detectionMethod?: string;
  // LangChain orchestration props
  langchainVerdict?: string;
  langchainReasoning?: string;
  langchainActions?: any[];
  langchainTrace?: string;
}

export default function DeepAnalysisSheet({
  isOpen,
  onClose,
  incident,
  coordination,
  coordinationLoading,
  onRefreshIncident,
  gateResults,
  escalationReasons,
  detectionMethod,
  langchainVerdict,
  langchainReasoning,
  langchainActions,
  langchainTrace
}: DeepAnalysisSheetProps) {
  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent className="w-[900px] sm:max-w-[900px] p-0 flex flex-col bg-background border-l">
        <SheetHeader className="p-6 border-b bg-muted/5 shrink-0">
          <SheetTitle className="flex items-center gap-2 text-xl">
            <BrainCircuit className="w-6 h-6 text-primary" />
            Deep Threat Analysis
          </SheetTitle>
          <SheetDescription>
            Comprehensive analysis from ML ensembles, Council of Models, and autonomous AI agents.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 overflow-hidden bg-background/50">
          <Tabs defaultValue="comprehensive" className="h-full flex flex-col">
            <div className="px-6 pt-4 border-b bg-background">
              <TabsList className="w-full justify-start h-12 bg-transparent p-0 space-x-6">
                <TabsTrigger
                  value="comprehensive"
                  className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none h-full px-0 font-medium"
                >
                  <BrainCircuit className="w-4 h-4 mr-2" />
                  Complete Analysis
                </TabsTrigger>
                <TabsTrigger
                  value="overview"
                  className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none h-full px-0 font-medium"
                >
                  <Activity className="w-4 h-4 mr-2" />
                  ML Details
                </TabsTrigger>
                <TabsTrigger
                  value="council"
                  className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none h-full px-0 font-medium"
                >
                  <Shield className="w-4 h-4 mr-2" />
                  Council Verdict
                </TabsTrigger>
                <TabsTrigger
                  value="agents"
                  className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none h-full px-0 font-medium"
                >
                  <Users className="w-4 h-4 mr-2" />
                  Agent Actions
                </TabsTrigger>
              </TabsList>
            </div>

            <ScrollArea className="flex-1 p-6">
              <TabsContent value="comprehensive" className="m-0 space-y-6">
                <ComprehensiveAnalysisTab incident={incident} />
              </TabsContent>

              <TabsContent value="overview" className="m-0 space-y-6">
                <OverviewTab
                  incident={incident}
                  gateResults={gateResults}
                  escalationReasons={escalationReasons}
                  detectionMethod={detectionMethod}
                />
              </TabsContent>

              <TabsContent value="council" className="m-0 space-y-6">
                <CouncilAnalysisTab
                  incident={incident}
                  onRefresh={onRefreshIncident}
                  langchainVerdict={langchainVerdict}
                  langchainReasoning={langchainReasoning}
                  langchainActions={langchainActions}
                  langchainTrace={langchainTrace}
                />
              </TabsContent>

              <TabsContent value="agents" className="m-0 space-y-6">
                <AIAgentsTab coordination={coordination} loading={coordinationLoading} />
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </div>
      </SheetContent>
    </Sheet>
  );
}
