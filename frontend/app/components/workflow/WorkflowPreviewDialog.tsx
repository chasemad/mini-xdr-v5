'use client'

import React from 'react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
    Workflow as WorkflowIcon,
    Zap,
    User,
    Clock,
    Activity,
    Target,
    GitBranch,
    Play,
    X
} from 'lucide-react';
import { Workflow } from './workflow-client';

interface WorkflowPreviewDialogProps {
    workflow: Workflow | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onLoad: () => void;
}

export function WorkflowPreviewDialog({ workflow, open, onOpenChange, onLoad }: WorkflowPreviewDialogProps) {
    if (!workflow) return null;

    const nodeCount = workflow.graph?.nodes?.length || 0;
    const edgeCount = workflow.graph?.edges?.length || 0;

    // Count node types
    const triggerNodes = workflow.graph?.nodes?.filter((n: any) => n.type === 'triggerNode').length || 0;
    const actionNodes = workflow.graph?.nodes?.filter((n: any) => n.type === 'actionNode').length || 0;
    const conditionNodes = workflow.graph?.nodes?.filter((n: any) => n.type === 'conditionNode').length || 0;

    const handleLoad = () => {
        onLoad();
        onOpenChange(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-3xl max-h-[85vh] bg-slate-950 border-white/10 text-slate-200 flex flex-col p-0">
                {/* Header */}
                <DialogHeader className="px-6 pt-6 pb-4 space-y-3">
                    <div className="flex items-start justify-between">
                        <div className="flex-1">
                            <DialogTitle className="text-xl font-semibold text-white">
                                {workflow.name}
                            </DialogTitle>
                            <DialogDescription className="text-sm text-slate-400 mt-1">
                                Preview workflow structure and details before loading
                            </DialogDescription>
                        </div>
                        <Badge
                            variant="outline"
                            className={`ml-4 ${workflow.incident_id
                                ? 'border-blue-500/30 text-blue-400 bg-blue-950/10'
                                : 'border-amber-500/30 text-amber-400 bg-amber-950/10'
                                }`}
                        >
                            {workflow.incident_id ? (
                                <><User size={12} className="mr-1" /> Manual</>
                            ) : (
                                <><Zap size={12} className="mr-1" /> System</>
                            )}
                        </Badge>
                    </div>
                </DialogHeader>

                {/* Content */}
                <div className="flex-1 overflow-hidden px-6">
                    <div className="grid grid-cols-3 gap-4 mb-6">
                        {/* Stats Cards */}
                        <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <WorkflowIcon size={14} />
                                <span className="text-xs font-medium">Total Nodes</span>
                            </div>
                            <div className="text-2xl font-bold text-white">{nodeCount}</div>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <GitBranch size={14} />
                                <span className="text-xs font-medium">Connections</span>
                            </div>
                            <div className="text-2xl font-bold text-white">{edgeCount}</div>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-lg p-4">
                            <div className="flex items-center gap-2 text-slate-400 mb-1">
                                <Clock size={14} />
                                <span className="text-xs font-medium">Created</span>
                            </div>
                            <div className="text-sm font-medium text-white">
                                {new Date(workflow.created_at).toLocaleDateString()}
                            </div>
                        </div>
                    </div>

                    {/* Node Breakdown */}
                    <div className="space-y-4">
                        <div>
                            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                                Node Breakdown
                            </h3>
                            <div className="grid grid-cols-3 gap-3">
                                <div className="flex items-center gap-2 px-3 py-2 bg-yellow-500/5 border border-yellow-500/20 rounded-lg">
                                    <Zap size={14} className="text-yellow-400" />
                                    <div>
                                        <div className="text-xs text-slate-400">Triggers</div>
                                        <div className="text-sm font-semibold text-yellow-400">{triggerNodes}</div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 px-3 py-2 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                                    <Activity size={14} className="text-blue-400" />
                                    <div>
                                        <div className="text-xs text-slate-400">Actions</div>
                                        <div className="text-sm font-semibold text-blue-400">{actionNodes}</div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 px-3 py-2 bg-purple-500/5 border border-purple-500/20 rounded-lg">
                                    <Target size={14} className="text-purple-400" />
                                    <div>
                                        <div className="text-xs text-slate-400">Conditions</div>
                                        <div className="text-sm font-semibold text-purple-400">{conditionNodes}</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <Separator className="bg-white/5" />

                        {/* Node List */}
                        <div>
                            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                                Workflow Nodes ({nodeCount})
                            </h3>
                            <ScrollArea className="h-48 pr-4">
                                <div className="space-y-2">
                                    {workflow.graph?.nodes?.map((node: any, index: number) => (
                                        <div
                                            key={node.id || index}
                                            className="flex items-center gap-3 px-3 py-2 bg-white/5 border border-white/5 rounded-lg hover:border-white/10 transition-colors"
                                        >
                                            <div className={`p-1.5 rounded ${node.type === 'triggerNode' ? 'bg-yellow-500/10 text-yellow-400' :
                                                    node.type === 'conditionNode' ? 'bg-purple-500/10 text-purple-400' :
                                                        'bg-blue-500/10 text-blue-400'
                                                }`}>
                                                {node.type === 'triggerNode' ? <Zap size={12} /> :
                                                    node.type === 'conditionNode' ? <Target size={12} /> :
                                                        <Activity size={12} />}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <div className="text-xs font-medium text-slate-200 truncate">
                                                    {node.data?.label || 'Unnamed Node'}
                                                </div>
                                                {node.data?.description && (
                                                    <div className="text-[10px] text-slate-500 truncate">
                                                        {node.data.description}
                                                    </div>
                                                )}
                                            </div>
                                            <Badge variant="outline" className="text-[9px] border-white/10 text-slate-400 bg-white/5">
                                                {node.type === 'triggerNode' ? 'Trigger' :
                                                    node.type === 'conditionNode' ? 'Condition' :
                                                        node.data?.category || 'Action'}
                                            </Badge>
                                        </div>
                                    )) || (
                                            <div className="text-center py-8 text-slate-500 text-xs">
                                                No nodes in this workflow
                                            </div>
                                        )}
                                </div>
                            </ScrollArea>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <DialogFooter className="px-6 py-4 border-t border-white/5">
                    <Button
                        variant="outline"
                        onClick={() => onOpenChange(false)}
                        className="bg-slate-900/50 border-white/10 text-slate-300 hover:bg-white/10 hover:text-white"
                    >
                        <X size={14} className="mr-2" />
                        Cancel
                    </Button>
                    <Button
                        onClick={handleLoad}
                        className="bg-emerald-500 hover:bg-emerald-400 text-black font-semibold shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                    >
                        <Play size={14} className="mr-2" />
                        Load Workflow
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
