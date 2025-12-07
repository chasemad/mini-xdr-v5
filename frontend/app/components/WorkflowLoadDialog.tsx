import React, { useEffect, useState } from 'react';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { getWorkflows, Workflow } from './workflow/workflow-client';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { FolderOpen, Loader2, FileText, Clock, AlertTriangle } from 'lucide-react';
import { Badge } from "@/components/ui/badge";

interface WorkflowLoadDialogProps {
    onLoad: (workflow: Workflow) => void;
}

export function WorkflowLoadDialog({ onLoad }: WorkflowLoadDialogProps) {
    const [open, setOpen] = useState(false);
    const [workflows, setWorkflows] = useState<Workflow[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (open) {
            setLoading(true);
            setError(null);
            getWorkflows()
                .then(setWorkflows)
                .catch((err) => {
                    console.error('Failed to load workflows:', err);
                    setError(err.message || 'Failed to load workflows');
                })
                .finally(() => setLoading(false));
        }
    }, [open]);

    const handleLoad = (wf: Workflow) => {
        onLoad(wf);
        setOpen(false);
    };

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="bg-slate-900/50 border-white/10 text-slate-300 hover:bg-white/10 hover:text-white backdrop-blur-md">
                    <FolderOpen size={16} className="mr-2" /> Load
                </Button>
            </DialogTrigger>
            <DialogContent className="bg-slate-950 border-white/10 text-slate-200 max-w-3xl max-h-[80vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle>Load Workflow</DialogTitle>
                </DialogHeader>

                <div className="flex-1 overflow-auto min-h-[300px]">
                    {loading ? (
                        <div className="flex items-center justify-center h-full text-slate-500">
                            <Loader2 className="animate-spin mr-2" /> Loading workflows...
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center h-full text-red-400">
                            <AlertTriangle size={48} className="mb-4 opacity-50" />
                            <p className="text-sm font-medium">Failed to load workflows</p>
                            <p className="text-xs text-slate-500 mt-1">{error}</p>
                        </div>
                    ) : workflows.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-slate-500">
                            <FileText size={48} className="mb-4 opacity-20" />
                            <p>No saved workflows found.</p>
                        </div>
                    ) : (
                        <Table>
                            <TableHeader>
                                <TableRow className="border-white/10 hover:bg-white/5">
                                    <TableHead className="text-slate-400">Name</TableHead>
                                    <TableHead className="text-slate-400">Status</TableHead>
                                    <TableHead className="text-slate-400">Created</TableHead>
                                    <TableHead className="text-right text-slate-400">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {workflows.map((wf) => (
                                    <TableRow key={wf.id} className="border-white/10 hover:bg-white/5">
                                        <TableCell className="font-medium text-slate-200">
                                            {wf.name}
                                            {wf.incident_id && (
                                                <span className="ml-2 text-xs text-slate-500">(Incident #{wf.incident_id})</span>
                                            )}
                                        </TableCell>
                                        <TableCell>
                                            <Badge variant="outline" className="bg-slate-900/50 border-white/10 text-slate-400">
                                                {wf.status}
                                            </Badge>
                                        </TableCell>
                                        <TableCell className="text-slate-400 text-xs">
                                            <div className="flex items-center gap-1">
                                                <Clock size={12} />
                                                {new Date(wf.created_at).toLocaleDateString()}
                                            </div>
                                        </TableCell>
                                        <TableCell className="text-right">
                                            <Button size="sm" variant="ghost" className="hover:bg-cyan-500/20 hover:text-cyan-400" onClick={() => handleLoad(wf)}>
                                                Load
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
}
