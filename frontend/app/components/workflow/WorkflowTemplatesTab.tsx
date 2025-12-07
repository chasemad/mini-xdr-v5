'use client'

import React, { useEffect, useState } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
    Search,
    Loader2,
    FileText,
    Workflow as WorkflowIcon,
    Trash2,
    Edit2,
    Check,
    X,
    AlertTriangle,
    Clock,
    Zap,
    User
} from 'lucide-react';
import { getWorkflows, deleteWorkflow, updateWorkflow, Workflow } from './workflow-client';
import { toast } from "@/components/ui/toast";
import { WorkflowPreviewDialog } from './WorkflowPreviewDialog';

interface WorkflowTemplatesTabProps {
    onLoadWorkflow: (workflow: Workflow) => void;
}

export function WorkflowTemplatesTab({ onLoadWorkflow }: WorkflowTemplatesTabProps) {
    const [workflows, setWorkflows] = useState<Workflow[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editingName, setEditingName] = useState('');
    const [previewWorkflow, setPreviewWorkflow] = useState<Workflow | null>(null);
    const [previewOpen, setPreviewOpen] = useState(false);

    // Load workflows on mount
    useEffect(() => {
        loadWorkflows();
    }, []);

    const loadWorkflows = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getWorkflows();
            setWorkflows(data);
        } catch (err: any) {
            console.error('Failed to load workflows:', err);
            setError(err.message || 'Failed to load workflows');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: number, name: string) => {
        if (!confirm(`Delete workflow "${name}"? This cannot be undone.`)) {
            return;
        }

        try {
            await deleteWorkflow(id);
            setWorkflows(prev => prev.filter(w => w.id !== id));
            toast.success("Workflow Deleted", `"${name}" has been removed.`);
        } catch (err: any) {
            console.error('Failed to delete workflow:', err);
            toast.error("Delete Failed", err.message || 'Could not delete workflow');
        }
    };

    const handleEditStart = (workflow: Workflow) => {
        setEditingId(workflow.id);
        setEditingName(workflow.name);
    };

    const handleEditSave = async (id: number) => {
        if (!editingName.trim()) {
            toast.error("Invalid Name", "Workflow name cannot be empty");
            return;
        }

        try {
            const updated = await updateWorkflow(id, { name: editingName });
            setWorkflows(prev => prev.map(w => w.id === id ? { ...w, name: updated.name } : w));
            setEditingId(null);
            toast.success("Workflow Updated", "Name changed successfully");
        } catch (err: any) {
            console.error('Failed to update workflow:', err);
            toast.error("Update Failed", err.message || 'Could not update workflow');
        }
    };

    const handleEditCancel = () => {
        setEditingId(null);
        setEditingName('');
    };

    const handlePreview = (workflow: Workflow) => {
        setPreviewWorkflow(workflow);
        setPreviewOpen(true);
    };

    const handleLoadFromPreview = () => {
        if (previewWorkflow) {
            onLoadWorkflow(previewWorkflow);
        }
    };

    // Filter workflows by search term
    const filteredWorkflows = workflows.filter(wf =>
        wf.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Get node count from graph
    const getNodeCount = (workflow: Workflow): number => {
        return workflow.graph?.nodes?.length || 0;
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full text-slate-500">
                <Loader2 className="animate-spin mr-2" size={18} />
                <span className="text-xs">Loading templates...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-red-400 p-4">
                <AlertTriangle size={32} className="mb-3 opacity-50" />
                <p className="text-xs font-medium">Failed to load workflows</p>
                <p className="text-[10px] text-slate-500 mt-1 text-center">{error}</p>
                <Button
                    size="sm"
                    variant="outline"
                    onClick={loadWorkflows}
                    className="mt-4 text-xs h-7 border-white/10 bg-[#111] hover:bg-[#1a1a1a]"
                >
                    Retry
                </Button>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col">
            {/* Search Bar */}
            <div className="p-3 border-b border-white/5">
                <div className="relative">
                    <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-500" />
                    <Input
                        type="text"
                        placeholder="Search templates..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-8 h-9 bg-[#111] border-white/10 text-xs text-slate-200 focus:border-blue-500/50"
                    />
                </div>
            </div>

            {/* Workflow List */}
            <ScrollArea className="flex-1">
                <div className="p-3 space-y-2">
                    {filteredWorkflows.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                            <FileText size={32} className="mb-3 opacity-20" />
                            <p className="text-xs">
                                {searchTerm ? 'No matching templates' : 'No saved templates'}
                            </p>
                        </div>
                    ) : (
                        filteredWorkflows.map((workflow) => (
                            <div
                                key={workflow.id}
                                className="p-3 rounded-lg border border-white/5 bg-[#111] hover:border-blue-500/30 hover:bg-[#161616] transition-all group"
                            >
                                {/* Header - Name & Edit */}
                                <div className="flex items-start justify-between gap-2 mb-2">
                                    {editingId === workflow.id ? (
                                        <div className="flex-1 flex items-center gap-1">
                                            <Input
                                                value={editingName}
                                                onChange={(e) => setEditingName(e.target.value)}
                                                className="h-7 text-xs bg-[#0A0A0A] border-blue-500/50"
                                                autoFocus
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') handleEditSave(workflow.id);
                                                    if (e.key === 'Escape') handleEditCancel();
                                                }}
                                            />
                                            <Button
                                                size="icon"
                                                variant="ghost"
                                                className="h-7 w-7 text-green-400 hover:text-green-300 hover:bg-green-950/20"
                                                onClick={() => handleEditSave(workflow.id)}
                                            >
                                                <Check size={12} />
                                            </Button>
                                            <Button
                                                size="icon"
                                                variant="ghost"
                                                className="h-7 w-7 text-slate-400 hover:text-white hover:bg-white/5"
                                                onClick={handleEditCancel}
                                            >
                                                <X size={12} />
                                            </Button>
                                        </div>
                                    ) : (
                                        <>
                                            <h4 className="text-xs font-medium text-slate-200 flex-1 truncate">
                                                {workflow.name}
                                            </h4>
                                            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <Button
                                                    size="icon"
                                                    variant="ghost"
                                                    className="h-6 w-6 text-slate-400 hover:text-blue-400 hover:bg-blue-950/20"
                                                    onClick={() => handleEditStart(workflow)}
                                                >
                                                    <Edit2 size={12} />
                                                </Button>
                                                <Button
                                                    size="icon"
                                                    variant="ghost"
                                                    className="h-6 w-6 text-slate-400 hover:text-red-400 hover:bg-red-950/20"
                                                    onClick={() => handleDelete(workflow.id, workflow.name)}
                                                >
                                                    <Trash2 size={12} />
                                                </Button>
                                            </div>
                                        </>
                                    )}
                                </div>

                                {/* Metadata */}
                                <div className="flex items-center gap-2 mb-2 flex-wrap">
                                    <Badge
                                        variant="outline"
                                        className={`text-[9px] px-1.5 py-0 h-4 ${workflow.incident_id
                                            ? 'border-blue-500/30 text-blue-400 bg-blue-950/10'
                                            : 'border-amber-500/30 text-amber-400 bg-amber-950/10'
                                            }`}
                                    >
                                        {workflow.incident_id ? (
                                            <><User size={8} className="mr-0.5" /> Manual</>
                                        ) : (
                                            <><Zap size={8} className="mr-0.5" /> System</>
                                        )}
                                    </Badge>
                                    <span className="text-[9px] text-slate-500 flex items-center gap-1">
                                        <WorkflowIcon size={8} />
                                        {getNodeCount(workflow)} nodes
                                    </span>
                                    <span className="text-[9px] text-slate-500 flex items-center gap-1">
                                        <Clock size={8} />
                                        {new Date(workflow.created_at).toLocaleDateString()}
                                    </span>
                                </div>

                                {/* Load Button */}
                                <Button
                                    size="sm"
                                    className="w-full h-7 text-xs bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500/50"
                                    onClick={() => handlePreview(workflow)}
                                >
                                    Preview Workflow
                                </Button>
                            </div>
                        ))
                    )}
                </div>
            </ScrollArea>

            {/* Workflow Preview Dialog */}
            <WorkflowPreviewDialog
                workflow={previewWorkflow}
                open={previewOpen}
                onOpenChange={setPreviewOpen}
                onLoad={handleLoadFromPreview}
            />
        </div>
    );
}
