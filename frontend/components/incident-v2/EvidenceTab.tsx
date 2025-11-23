"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Copy, ExternalLink, Shield, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { IncidentCoordination } from "@/lib/agent-api";

interface EvidenceTabProps {
  incident: any;
  coordination: IncidentCoordination | null;
}

export function EvidenceTab({ incident, coordination }: EvidenceTabProps) {
  const forensics = coordination?.agent_decisions?.forensics;
  const attribution = coordination?.agent_decisions?.attribution;

  // Combine IOCs
  const iocs = [
    ...(incident.iocs?.ip_addresses || []).map((val: string) => ({ type: "IP", value: val, source: "Detection" })),
    ...(incident.iocs?.domains || []).map((val: string) => ({ type: "Domain", value: val, source: "Detection" })),
    ...(incident.iocs?.hashes || []).map((val: string) => ({ type: "Hash", value: val, source: "Detection" })),
    ...(attribution?.iocs_identified || []).map((ioc: any) => ({ type: ioc.type || "Unknown", value: ioc.value, source: "Attribution Agent" }))
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-amber-500" />
            Indicators of Compromise (IOCs)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Type</TableHead>
                <TableHead>Value</TableHead>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {iocs.map((ioc, idx) => (
                <TableRow key={idx}>
                  <TableCell><Badge variant="outline">{ioc.type}</Badge></TableCell>
                  <TableCell className="font-mono text-xs">{ioc.value}</TableCell>
                  <TableCell className="text-muted-foreground text-sm">{ioc.source}</TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon" onClick={() => navigator.clipboard.writeText(ioc.value)}>
                      <Copy className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {iocs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-muted-foreground py-8">
                    No IOCs detected
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Forensic Artifacts</CardTitle>
          </CardHeader>
          <CardContent>
            {forensics?.evidence_collected?.length > 0 ? (
              <ul className="space-y-2">
                {forensics.evidence_collected.map((item: string, i: number) => (
                  <li key={i} className="flex items-center gap-2 text-sm p-2 bg-muted rounded">
                    <Shield className="h-4 w-4 text-primary" />
                    {item}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground text-sm">No forensic artifacts collected yet.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Network Analysis</CardTitle>
          </CardHeader>
          <CardContent>
             <div className="space-y-4">
                <div className="flex justify-between text-sm border-b pb-2">
                    <span className="text-muted-foreground">Source IP</span>
                    <span className="font-mono">{incident.src_ip}</span>
                </div>
                <div className="flex justify-between text-sm border-b pb-2">
                    <span className="text-muted-foreground">Unique Ports</span>
                    <span className="font-mono">{incident.ml_features?.unique_ports || "N/A"}</span>
                </div>
                 <div className="flex justify-between text-sm border-b pb-2">
                    <span className="text-muted-foreground">Flow Count</span>
                    <span className="font-mono">{incident.ml_features?.flow_count || "N/A"}</span>
                </div>
             </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
