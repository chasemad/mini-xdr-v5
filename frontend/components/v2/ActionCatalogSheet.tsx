"use client";

import React from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Shield, Search, Lock, Activity, Terminal, FileSearch,
  Network, UserX, Database, Bug, Skull, Play, Zap
} from "lucide-react";
import { cn } from "@/lib/utils";

interface ActionCatalogProps {
  isOpen: boolean;
  onClose: () => void;
  onExecuteAction: (actionId: string, params?: any) => void;
}

type ActionCategory = "containment" | "investigation" | "forensics" | "recovery" | "deception";

interface CatalogAction {
  id: string;
  title: string;
  description: string;
  category: ActionCategory;
  riskLevel: "low" | "medium" | "high" | "critical";
  agent: string;
  icon: React.ElementType;
  requiresApproval?: boolean;
}

const catalogActions: CatalogAction[] = [
  // Containment
  {
    id: "block_ip_firewall",
    title: "Block IP (Firewall)",
    description: "Add a deny rule to the perimeter firewall for the source IP.",
    category: "containment",
    riskLevel: "low",
    agent: "Containment Agent",
    icon: Shield
  },
  {
    id: "isolate_host",
    title: "Isolate Host",
    description: "Quarantine the host from the network, allowing only management connectivity.",
    category: "containment",
    riskLevel: "high",
    agent: "EDR Agent",
    icon: Network
  },
  {
    id: "disable_user",
    title: "Disable User Account",
    description: "Temporarily disable the compromised user account in AD/IdP.",
    category: "containment",
    riskLevel: "medium",
    agent: "IAM Agent",
    icon: UserX
  },
  {
    id: "kill_process",
    title: "Kill Malicious Process",
    description: "Terminate the suspicious process tree on the endpoint.",
    category: "containment",
    riskLevel: "medium",
    agent: "EDR Agent",
    icon: Zap
  },

  // Investigation & Forensics
  {
    id: "memory_dump",
    title: "Capture Memory Dump",
    description: "Trigger a full memory acquisition for Volatility analysis.",
    category: "forensics",
    riskLevel: "low",
    agent: "Forensics Agent",
    icon: Database
  },
  {
    id: "yara_scan",
    title: "Run YARA Scan",
    description: "Scan the host filesystem against known malware signatures.",
    category: "forensics",
    riskLevel: "low",
    agent: "Forensics Agent",
    icon: FileSearch
  },
  {
    id: "deep_packet_inspection",
    title: "Deep Packet Inspection",
    description: "Analyze PCAP data for C2 beacons and exfiltration signatures.",
    category: "investigation",
    riskLevel: "low",
    agent: "Network Agent",
    icon: Activity
  },
  {
    id: "threat_intel_enrichment",
    title: "Deep Threat Intel Lookup",
    description: "Query premium feeds (VirusTotal, AlienVault) for IOC context.",
    category: "investigation",
    riskLevel: "low",
    agent: "Attribution Agent",
    icon: Search
  },

  // Deception
  {
    id: "deploy_decoy_creds",
    title: "Deploy Decoy Credentials",
    description: "Inject honey-tokens into LSASS to detect lateral movement.",
    category: "deception",
    riskLevel: "low",
    agent: "Deception Agent",
    icon: Bug
  },

  // Recovery
  {
    id: "restore_snapshot",
    title: "Restore VM Snapshot",
    description: "Revert the virtual machine to the last known good state.",
    category: "recovery",
    riskLevel: "critical",
    agent: "Infrastructure Agent",
    icon: RotateCcw,
    requiresApproval: true
  }
];

import { RotateCcw } from "lucide-react";

export default function ActionCatalogSheet({ isOpen, onClose, onExecuteAction }: ActionCatalogProps) {
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedCategory, setSelectedCategory] = React.useState<ActionCategory | "all">("all");

  const filteredActions = catalogActions.filter(action => {
    const matchesSearch = action.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          action.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === "all" || action.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const categories = [
    { id: "all", label: "All Capabilities" },
    { id: "containment", label: "Containment" },
    { id: "investigation", label: "Investigation" },
    { id: "forensics", label: "Forensics" },
    { id: "deception", label: "Deception" },
    { id: "recovery", label: "Recovery" },
  ];

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent className="w-[600px] sm:max-w-[600px] p-0 flex flex-col bg-background border-l border-border" side="right">
        <SheetHeader className="p-6 pb-4 border-b shrink-0 bg-card">
          <div className="flex items-center justify-between mb-2">
             <SheetTitle className="text-lg font-bold flex items-center gap-2">
               <Terminal className="w-5 h-5 text-primary" />
               Agent Capabilities Catalog
             </SheetTitle>
          </div>
          <SheetDescription className="text-xs">
            Browse and execute advanced autonomous capabilities across your security fabric.
          </SheetDescription>

          <div className="mt-4 relative">
            <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search capabilities..."
              className="pl-8 bg-background"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div className="flex gap-2 mt-4 overflow-x-auto pb-2 scrollbar-hide">
            {categories.map(cat => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id as any)}
                className={cn(
                  "text-xs px-3 py-1.5 rounded-full whitespace-nowrap transition-colors border",
                  selectedCategory === cat.id
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-muted/50 hover:bg-muted text-muted-foreground border-transparent"
                )}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </SheetHeader>

        <ScrollArea className="flex-1 bg-muted/5">
          <div className="p-6 grid gap-4">
            {filteredActions.map((action) => (
              <div
                key={action.id}
                className="group bg-card border hover:border-primary/50 rounded-lg p-4 transition-all shadow-sm hover:shadow-md flex gap-4 items-start"
              >
                <div className="p-2 rounded-md bg-primary/10 text-primary mt-1">
                  <action.icon className="w-5 h-5" />
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold text-sm text-foreground group-hover:text-primary transition-colors">
                        {action.title}
                      </h3>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline" className="text-[10px] font-normal opacity-80">
                          {action.agent}
                        </Badge>
                        {action.riskLevel === "high" || action.riskLevel === "critical" ? (
                           <Badge variant="destructive" className="text-[10px] h-4 px-1.5">
                              High Risk
                           </Badge>
                        ) : null}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      className="h-8 text-xs gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={() => onExecuteAction(action.id)}
                    >
                      <Play className="w-3 h-3" /> Run
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                    {action.description}
                  </p>
                </div>
              </div>
            ))}
            {filteredActions.length === 0 && (
              <div className="text-center py-12 text-muted-foreground">
                <Skull className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p>No capabilities found matching your criteria.</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
