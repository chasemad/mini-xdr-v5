"use client";

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button, type ButtonProps } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Play, Loader2, AlertOctagon } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface ActionCardProps {
  title: string;
  description: string;
  confidence?: number;
  impact: "Low" | "Medium" | "High";
  impactDescription?: string;
  onExecute: () => Promise<void>;
  badgeText?: string;
  badgeTone?: "info" | "success" | "warning" | "danger" | "muted";
  buttonLabel?: string;
  buttonVariant?: ButtonProps["variant"];
  requiresConfirm?: boolean;
  confirmTitle?: string;
  confirmDescription?: string;
  confirmActionLabel?: string;
  disabled?: boolean;
  loading?: boolean;
}

export default function ActionCard({
  title,
  description,
  confidence,
  impact,
  impactDescription,
  onExecute,
  badgeText,
  badgeTone = "info",
  buttonLabel,
  buttonVariant,
  requiresConfirm = false,
  confirmTitle,
  confirmDescription,
  confirmActionLabel,
  disabled = false,
  loading = false
}: ActionCardProps) {
  const [executing, setExecuting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const badgeToneClasses = {
    info: "bg-primary/10 text-primary border-primary/20",
    success: "bg-green-500/10 text-green-400 border-green-500/20",
    warning: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    danger: "bg-red-500/10 text-red-400 border-red-500/20",
    muted: "bg-muted text-muted-foreground border-muted"
  } as const;

  const handleExecute = async () => {
    setExecuting(true);
    try {
      await onExecute();
    } finally {
      setExecuting(false);
    }
  };

  const handleClick = () => {
    if (requiresConfirm) {
      setConfirmOpen(true);
      return;
    }
    void handleExecute();
  };

  const isBusy = executing || loading;
  const buttonText = isBusy ? 'Executing...' : (buttonLabel || 'Execute Action');

  return (
    <Card className="border-primary/20 shadow-md bg-card/95 hover:border-primary/50 transition-colors group">
      <CardHeader className="p-3 pb-1">
        <CardTitle className="text-sm flex justify-between items-center">
          <span className="font-semibold text-foreground">{title}</span>
          <div className="flex items-center gap-2">
            {badgeText && (
              <Badge className={`text-[10px] ${badgeToneClasses[badgeTone]}`}>
                {badgeText}
              </Badge>
            )}
            {confidence && (
              <Badge className="bg-primary/10 text-primary hover:bg-primary/20 border-primary/20 text-[10px]">
                {confidence}% Conf.
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3 pt-2">
        <p className="text-xs text-muted-foreground mb-3 min-h-[2.5em]">
          {description}
        </p>

        <TooltipProvider>
           <Tooltip delayDuration={200}>
              <TooltipTrigger asChild>
                 <div className="w-full">
                    <Button
                       className="w-full h-8 text-xs flex items-center gap-2"
                       size="sm"
                       onClick={handleClick}
                       disabled={isBusy || disabled}
                       variant={buttonVariant || (impact === 'High' ? 'destructive' : 'default')}
                    >
                       {isBusy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                       {buttonText}
                    </Button>
                 </div>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="w-64 p-0 border-muted">
                 <div className="p-3 bg-card rounded-md border">
                    <div className="flex items-center gap-2 mb-2 pb-2 border-b">
                       <AlertOctagon className="w-4 h-4 text-orange-500" />
                       <span className="font-semibold text-xs">Predicted Impact Simulation</span>
                    </div>
                    <div className="space-y-2 text-xs">
                       <div className="flex justify-between">
                          <span className="text-muted-foreground">Business Impact:</span>
                          <Badge variant="outline" className={
                             impact === 'High' ? 'text-red-500 border-red-500/30' :
                             impact === 'Medium' ? 'text-orange-500 border-orange-500/30' :
                             'text-green-500 border-green-500/30'
                          }>{impact}</Badge>
                       </div>
                       {impactDescription && (
                          <p className="text-muted-foreground bg-muted/50 p-2 rounded">
                             {impactDescription}
                          </p>
                       )}
                       <div className="flex justify-between text-[10px] text-muted-foreground pt-1">
                          <span>Confidence: 98%</span>
                          <span>Source: AI Simulation</span>
                       </div>
                    </div>
                 </div>
             </TooltipContent>
           </Tooltip>
        </TooltipProvider>

        {requiresConfirm && (
          <Dialog open={confirmOpen} onOpenChange={(open) => !isBusy && setConfirmOpen(open)}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>{confirmTitle || `Confirm ${buttonLabel || 'action'}`}</DialogTitle>
                {confirmDescription && (
                  <DialogDescription className="text-sm text-muted-foreground">
                    {confirmDescription}
                  </DialogDescription>
                )}
              </DialogHeader>
              <DialogFooter className="gap-2 sm:gap-0">
                <Button variant="ghost" onClick={() => setConfirmOpen(false)} disabled={isBusy}>
                  Cancel
                </Button>
                <Button
                  onClick={async () => {
                    await handleExecute();
                    setConfirmOpen(false);
                  }}
                  variant={buttonVariant || 'default'}
                  disabled={isBusy}
                  className="gap-2"
                >
                  {isBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                  {confirmActionLabel || 'Confirm'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </CardContent>
    </Card>
  );
}
