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

interface DeepAnalysisSheetProps {
  isOpen: boolean;
  onClose: () => void;
  incident: any;
  coordination: any;
  coordinationLoading: boolean;
  onRefreshIncident?: () => void;
}

export default function DeepAnalysisSheet({
  isOpen,
  onClose,
  incident,
  coordination,
  coordinationLoading,
  onRefreshIncident
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
          <Tabs defaultValue="overview" className="h-full flex flex-col">
            <div className="px-6 pt-4 border-b bg-background">
              <TabsList className="w-full justify-start h-12 bg-transparent p-0 space-x-6">
                <TabsTrigger
                  value="overview"
                  className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none h-full px-0 font-medium"
                >
                  <Activity className="w-4 h-4 mr-2" />
                  ML Overview
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
                  Agent Coordination
                </TabsTrigger>
              </TabsList>
            </div>

            <ScrollArea className="flex-1 p-6">
              <TabsContent value="overview" className="m-0 space-y-6">
                <OverviewTab incident={incident} />
              </TabsContent>

              <TabsContent value="council" className="m-0 space-y-6">
                <CouncilAnalysisTab incident={incident} onRefresh={onRefreshIncident} />
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
