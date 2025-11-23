"use client";

import React, { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import {
  Shield, Network, Terminal, Database, Mail, Cloud,
  FileSearch, Users, Lock, AlertOctagon, Activity,
  Search, Play, Info, Zap, AlertTriangle
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface AgentCapabilitiesSheetProps {
  isOpen: boolean;
  onClose: () => void;
  onExecute: (action: string) => void;
}

// Comprehensive catalog from codebase analysis
const AGENT_CAPABILITIES = {
  network: {
    label: "Network & Firewall",
    icon: Network,
    color: "blue",
    actions: [
      { id: "block_ip_advanced", name: "Block IP (Advanced)", description: "Block IP with adaptive duration and threat scoring. Prevents all traffic from target.", requiresApproval: false, impact: "Target IP blocked. Active connections dropped." },
      { id: "deploy_firewall_rules", name: "Deploy Firewall Rules", description: "Push custom firewall rules to network perimeter.", requiresApproval: true, impact: "Global firewall policy updated." },
      { id: "dns_sinkhole", name: "DNS Sinkhole", description: "Redirect malicious domains to sinkhole server.", requiresApproval: false, impact: "Domains unresolvable for all users." },
      { id: "traffic_redirection", name: "Traffic Redirection", description: "Redirect suspicious traffic for analysis.", requiresApproval: true, impact: "Traffic latency may increase." },
      { id: "network_segmentation", name: "Network Segmentation", description: "Isolate network segments to contain lateral movement.", requiresApproval: true, impact: "Inter-VLAN traffic blocked." },
      { id: "capture_network_traffic", name: "Capture Traffic (PCAP)", description: "Full packet capture for forensic analysis.", requiresApproval: false, impact: "High storage usage." },
      { id: "deploy_waf_rules", name: "Deploy WAF Rules", description: "Update Web Application Firewall rules.", requiresApproval: true, impact: "WAF config reload." },
    ]
  },
  endpoint: {
    label: "Endpoint & Host",
    icon: Terminal,
    color: "purple",
    actions: [
      { id: "isolate_host_advanced", name: "Isolate Host", description: "Complete network isolation with rollback capability.", requiresApproval: false, impact: "Host offline. Only admin access allowed." },
      { id: "memory_dump_collection", name: "Memory Dump", description: "Capture RAM snapshot for malware analysis.", requiresApproval: false, impact: "System freeze during dump (~30s)." },
      { id: "process_termination", name: "Kill Process", description: "Terminate malicious process by PID or name.", requiresApproval: false, impact: "Process stopped immediately." },
      { id: "registry_hardening", name: "Registry Hardening", description: "Apply security hardening to Windows Registry.", requiresApproval: true, impact: "System restart may be required." },
      { id: "system_recovery", name: "System Recovery", description: "Restore system to clean checkpoint.", requiresApproval: true, impact: "Data since last backup lost." },
      { id: "malware_removal", name: "Malware Removal", description: "Automated malware cleanup and remediation.", requiresApproval: false, impact: "File deletion." },
      { id: "scan_endpoint", name: "Endpoint Scan", description: "Full antivirus/EDR scan of endpoint.", requiresApproval: false, impact: "High CPU usage." },
    ]
  },
  forensics: {
    label: "Investigation & Forensics",
    icon: FileSearch,
    color: "orange",
    actions: [
      { id: "investigate_behavior", name: "Behavior Analysis", description: "Deep dive into attack patterns and TTPs.", requiresApproval: false, impact: "Read-only analysis." },
      { id: "hunt_similar_attacks", name: "Threat Hunting", description: "Proactive search for IoCs across environment.", requiresApproval: false, impact: "Read-only search." },
      { id: "threat_intel_lookup", name: "Threat Intel Lookup", description: "Query external threat intelligence feeds.", requiresApproval: false, impact: "API quota usage." },
      { id: "collect_evidence", name: "Evidence Collection", description: "Gather and preserve forensic artifacts.", requiresApproval: false, impact: "Read-only collection." },
      { id: "analyze_logs", name: "Log Analysis", description: "Correlate and analyze security logs.", requiresApproval: false, impact: "Heavy query load." },
      { id: "attribution_analysis", name: "Attribution Analysis", description: "Identify threat actor using ML and OSINT.", requiresApproval: false, impact: "Read-only analysis." },
    ]
  },
  identity: {
    label: "Identity & Access",
    icon: Users,
    color: "green",
    actions: [
      { id: "reset_passwords", name: "Reset Passwords (Bulk)", description: "Force password reset for compromised accounts.", requiresApproval: true, impact: "Users forced to relogin." },
      { id: "revoke_user_sessions", name: "Revoke Sessions", description: "Terminate all active user sessions.", requiresApproval: false, impact: "Immediate logout." },
      { id: "disable_user_account", name: "Disable Account", description: "Immediately disable user account.", requiresApproval: false, impact: "User lockout." },
      { id: "enforce_mfa", name: "Enforce MFA", description: "Require multi-factor authentication.", requiresApproval: true, impact: "Login flow change." },
      { id: "privileged_access_review", name: "Privilege Review", description: "Audit and restrict privileged access.", requiresApproval: false, impact: "Read-only audit." },
    ]
  },
  data: {
    label: "Data Protection",
    icon: Database,
    color: "cyan",
    actions: [
      { id: "check_database_integrity", name: "Database Integrity Check", description: "Verify database for tampering.", requiresApproval: false, impact: "Database load increase." },
      { id: "backup_critical_data", name: "Emergency Backup", description: "Create immutable backup of critical data.", requiresApproval: false, impact: "High bandwidth usage." },
      { id: "encrypt_sensitive_data", name: "Data Encryption", description: "Apply encryption to sensitive data at rest.", requiresApproval: true, impact: "Data temporarily unavailable." },
      { id: "enable_dlp", name: "Enable DLP", description: "Activate Data Loss Prevention policies.", requiresApproval: true, impact: "Policy enforcement enabled." },
    ]
  },
  communication: {
    label: "Alerting & Notification",
    icon: AlertOctagon,
    color: "red",
    actions: [
      { id: "alert_security_analysts", name: "Alert Analysts", description: "Send urgent notification to SOC team.", requiresApproval: false, impact: "Notifications sent." },
      { id: "create_incident_case", name: "Create Case", description: "Generate incident case in ticketing system.", requiresApproval: false, impact: "Ticket created." },
      { id: "stakeholder_notification", name: "Notify Stakeholders", description: "Alert executive leadership.", requiresApproval: true, impact: "High-priority alert sent." },
    ]
  }
};

export default function AgentCapabilitiesSheet({ isOpen, onClose, onExecute }: AgentCapabilitiesSheetProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [confirmAction, setConfirmAction] = useState<any>(null);

  const filteredCategories = Object.entries(AGENT_CAPABILITIES).filter(([key, category]) => {
    if (selectedCategory !== "all" && selectedCategory !== key) return false;

    if (!searchQuery) return true;

    const query = searchQuery.toLowerCase();
    return category.label.toLowerCase().includes(query) ||
           category.actions.some(a =>
             a.name.toLowerCase().includes(query) ||
             a.description.toLowerCase().includes(query)
           );
  });

  const handleActionClick = (action: any) => {
    setConfirmAction(action);
  };

  const handleConfirmExecute = () => {
    if (confirmAction) {
      onExecute(confirmAction.id);
      setConfirmAction(null);
      onClose();
    }
  };

  return (
    <>
      <Sheet open={isOpen} onOpenChange={onClose}>
        <SheetContent className="w-[900px] sm:max-w-[900px] p-0 flex flex-col bg-background border-l border-border">
          <SheetHeader className="p-6 pb-4 border-b border-border bg-card/50">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Zap className="w-5 h-5 text-primary" />
              </div>
              <div>
                <SheetTitle className="text-xl font-bold">AI Agent Capabilities</SheetTitle>
                <SheetDescription className="text-xs mt-1 text-muted-foreground">
                  Execute advanced containment, investigation, and remediation actions
                </SheetDescription>
              </div>
            </div>

            <div className="relative mt-4">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search capabilities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background border-input"
              />
            </div>
          </SheetHeader>

          <Tabs value={selectedCategory} onValueChange={setSelectedCategory} className="flex-1 flex flex-col min-h-0 bg-background">
            <div className="px-6 pt-4 bg-background">
              <TabsList className="w-full grid grid-cols-7 gap-1 h-auto bg-muted/50 p-1">
                <TabsTrigger value="all" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">All</TabsTrigger>
                <TabsTrigger value="network" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Network</TabsTrigger>
                <TabsTrigger value="endpoint" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Endpoint</TabsTrigger>
                <TabsTrigger value="forensics" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Forensics</TabsTrigger>
                <TabsTrigger value="identity" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Identity</TabsTrigger>
                <TabsTrigger value="data" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Data</TabsTrigger>
                <TabsTrigger value="communication" className="text-xs data-[state=active]:bg-background data-[state=active]:text-foreground">Alerts</TabsTrigger>
              </TabsList>
            </div>

            <ScrollArea className="flex-1 px-6 py-4 bg-background">
              <div className="space-y-6">
                {filteredCategories.map(([key, category]) => (
                  <div key={key} className="space-y-3">
                    <div className="flex items-center gap-2 sticky top-0 bg-background py-2 border-b border-border z-10">
                      <category.icon className={cn("w-4 h-4", `text-${category.color}-500`)} />
                      <h3 className="font-semibold text-sm text-foreground">{category.label}</h3>
                      <Badge variant="outline" className="ml-auto text-[10px] border-border text-muted-foreground">
                        {category.actions.length} actions
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      {category.actions.map((action) => (
                        <div
                          key={action.id}
                          className="p-3 border border-border rounded-lg bg-card hover:border-primary/50 hover:bg-accent/50 transition-all group cursor-pointer flex flex-col justify-between"
                          onClick={() => handleActionClick(action)}
                        >
                          <div>
                            <div className="flex items-start justify-between mb-2">
                              <h4 className="font-medium text-sm text-foreground group-hover:text-primary transition-colors">
                                {action.name}
                              </h4>
                              {action.requiresApproval && (
                                <Badge variant="outline" className="text-[8px] h-4 px-1 bg-yellow-500/10 text-yellow-600 border-yellow-500/30">
                                  Approval
                                </Badge>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground leading-relaxed mb-3">
                              {action.description}
                            </p>
                          </div>
                          <Button
                            size="sm"
                            className="w-full h-7 text-[10px] gap-1 bg-secondary text-secondary-foreground hover:bg-primary hover:text-primary-foreground transition-colors mt-auto"
                            variant="ghost"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleActionClick(action);
                            }}
                          >
                            <Play className="w-3 h-3" />
                            Execute Action
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}

                {filteredCategories.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground">
                    <Activity className="w-12 h-12 mx-auto mb-3 opacity-20" />
                    <p className="text-sm">No capabilities match your search</p>
                  </div>
                )}
              </div>
            </ScrollArea>

            <div className="border-t border-border p-4 bg-card/50 flex items-center justify-between shrink-0">
              <div className="text-xs text-muted-foreground">
                <Info className="w-3 h-3 inline mr-1" />
                Actions marked with "Approval" require manual confirmation
              </div>
              <Button variant="outline" size="sm" onClick={onClose} className="bg-background hover:bg-accent">
                Close
              </Button>
            </div>
          </Tabs>
        </SheetContent>
      </Sheet>

      {/* Confirmation Dialog */}
      <Dialog open={!!confirmAction} onOpenChange={(open) => !open && setConfirmAction(null)}>
        <DialogContent className="bg-background border-border sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              <AlertTriangle className="w-5 h-5 text-warning" />
              Confirm Action Execution
            </DialogTitle>
            <DialogDescription>
              Are you sure you want to execute this action? This may affect system operations.
            </DialogDescription>
          </DialogHeader>

          {confirmAction && (
            <div className="grid gap-4 py-4">
              <div className="p-4 rounded-md bg-muted/50 border border-border">
                <div className="font-semibold text-sm text-foreground mb-1">{confirmAction.name}</div>
                <div className="text-xs text-muted-foreground mb-3">{confirmAction.description}</div>

                <div className="text-xs font-medium text-foreground mb-1">Expected Impact:</div>
                <div className="text-xs text-muted-foreground bg-background p-2 rounded border border-border">
                  {confirmAction.impact}
                </div>
              </div>

              {confirmAction.requiresApproval && (
                <div className="flex items-center gap-2 text-xs text-yellow-600 bg-yellow-500/10 p-2 rounded border border-yellow-500/20">
                  <Info className="w-4 h-4" />
                  This action requires secondary approval which will be requested after initiation.
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmAction(null)} className="bg-background text-foreground border-input hover:bg-accent hover:text-accent-foreground">
              Cancel
            </Button>
            <Button onClick={handleConfirmExecute} className="bg-primary text-primary-foreground hover:bg-primary/90">
              Confirm Execution
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
