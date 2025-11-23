"use client";

import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertTriangle, Play, ShieldAlert, Info } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

interface ActionConfirmationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  actionId: string;
  actionName: string;
  description?: string;
  isDangerous?: boolean;
  requiresApproval?: boolean;
}

export default function ActionConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  actionId,
  actionName,
  description,
  isDangerous = false,
  requiresApproval = false
}: ActionConfirmationModalProps) {
  const [isConfirmed, setIsConfirmed] = React.useState(false);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className={`p-2 rounded-full ${isDangerous ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>
              {isDangerous ? <AlertTriangle className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
            </div>
            <DialogTitle>Confirm Action Execution</DialogTitle>
          </div>
          <DialogDescription className="pt-2">
            You are about to execute the following AI Agent action:
          </DialogDescription>
        </DialogHeader>

        <div className="bg-muted/30 border rounded-lg p-4 my-2">
          <h3 className="font-semibold text-base mb-1">{actionName}</h3>
          <p className="text-sm text-muted-foreground">{description || "No description available."}</p>

          <div className="mt-3 flex gap-2 text-xs">
            <span className="px-2 py-1 bg-background border rounded font-mono text-muted-foreground">ID: {actionId}</span>
            {requiresApproval && (
              <span className="px-2 py-1 bg-yellow-100 text-yellow-700 border border-yellow-200 rounded font-medium">
                Requires Approval
              </span>
            )}
          </div>
        </div>

        {isDangerous && (
          <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-100 rounded-md text-red-800 text-sm mb-4">
            <Info className="w-4 h-4 mt-0.5 shrink-0" />
            <p>This action may have significant impact on the target system or network availability. Please proceed with caution.</p>
          </div>
        )}

        {isDangerous && (
          <div className="flex items-center space-x-2 py-2">
            <Checkbox
              id="confirm-hazardous"
              checked={isConfirmed}
              onCheckedChange={(c) => setIsConfirmed(!!c)}
            />
            <Label htmlFor="confirm-hazardous" className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              I understand the risks and want to proceed
            </Label>
          </div>
        )}

        <DialogFooter className="gap-2 sm:gap-0">
          <DialogClose asChild>
            <Button variant="outline" onClick={onClose}>Cancel</Button>
          </DialogClose>
          <Button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            disabled={isDangerous && !isConfirmed}
            variant={isDangerous ? "destructive" : "default"}
            className="gap-2"
          >
            <Play className="w-4 h-4" />
            Execute Action
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
