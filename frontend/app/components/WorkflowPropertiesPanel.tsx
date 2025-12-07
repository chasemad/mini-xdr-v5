import React from 'react';
import { Node } from '@xyflow/react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Trash2, Copy, Activity, LayoutGrid } from 'lucide-react';
import ActionNodeLibrary from './ActionNodeLibrary';
import { WorkflowTemplatesTab } from './workflow/WorkflowTemplatesTab';
import { Workflow } from './workflow/workflow-client';

interface WorkflowPropertiesPanelProps {
    selectedNode: Node | null;
    workflowMetadata: { name: string; id?: string; type?: 'incident' | 'system'; triggerEvent?: string; triggerThreshold?: string };
    onMetadataChange: (key: string, value: string) => void;
    onNodeUpdate: (nodeId: string, data: Record<string, unknown>) => void;
    onDeleteNode: (nodeId: string) => void;
    onLoadWorkflow?: (workflow: Workflow) => void;
    availableActions?: Record<string, any>;
}

export function WorkflowPropertiesPanel({
    selectedNode,
    workflowMetadata,
    onMetadataChange,
    onNodeUpdate,
    onDeleteNode,
    onLoadWorkflow,
    availableActions = {}
}: WorkflowPropertiesPanelProps) {
    return (
        <div className="w-80 h-full bg-[#0A0A0A] border-l border-white/5 flex flex-col z-20">
            <Tabs className="w-full h-full flex flex-col">
                <div className="p-4 border-b border-white/5">
                    <TabsList className="w-full grid grid-cols-4 bg-[#111] p-1 h-9">
                        <TabsTrigger
                            value="library"
                            className="text-xs data-[state=active]:bg-[#222] data-[state=active]:text-white text-slate-500"
                        >
                            Library
                        </TabsTrigger>
                        <TabsTrigger
                            value="properties"
                            className="text-xs data-[state=active]:bg-[#222] data-[state=active]:text-white text-slate-500"
                        >
                            Props
                        </TabsTrigger>
                        <TabsTrigger
                            value="code"
                            className="text-xs data-[state=active]:bg-[#222] data-[state=active]:text-white text-slate-500"
                        >
                            Code
                        </TabsTrigger>
                        <TabsTrigger
                            value="templates"
                            className="text-xs data-[state=active]:bg-[#222] data-[state=active]:text-white text-slate-500"
                        >
                            Templates
                        </TabsTrigger>
                    </TabsList>
                </div>

                <div className="flex-1 overflow-hidden">
                    <TabsContent value="library" className="mt-0 h-full p-4">
                        <ActionNodeLibrary actions={availableActions} isEmbedded={true} />
                    </TabsContent>

                    <TabsContent value="properties" className="mt-0 h-full">
                        <ScrollArea className="h-full">
                            <div className="p-4 space-y-6">
                                {selectedNode ? (
                                    <div className="space-y-6">
                                        <div className="flex items-center justify-between">
                                            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Node Configuration</h3>
                                            <Badge variant="outline" className="text-[10px] border-cyan-500/30 text-cyan-400 bg-cyan-950/10 px-2 py-0.5 h-auto rounded-full">
                                                {selectedNode.type}
                                            </Badge>
                                        </div>

                                        {/* Trigger Node Specific Configuration */}
                                        {selectedNode.type === 'triggerNode' && (
                                            <div className="space-y-4 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
                                                <h4 className="text-xs font-medium text-yellow-500 uppercase tracking-wider flex items-center gap-2">
                                                    <Activity size={12} /> Trigger Logic
                                                </h4>

                                                <div className="space-y-3">
                                                    <div className="space-y-1.5">
                                                        <Label className="text-xs text-slate-400 font-normal">Trigger Type</Label>
                                                        <div className="grid grid-cols-2 gap-2">
                                                            <Button
                                                                variant="outline"
                                                                onClick={() => onMetadataChange('type', 'incident')}
                                                                className={`h-8 text-xs border-white/5 ${workflowMetadata.type === 'incident' ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30' : 'bg-[#111] text-slate-400 hover:bg-[#1a1a1a]'}`}
                                                            >
                                                                Manual / Incident
                                                            </Button>
                                                            <Button
                                                                variant="outline"
                                                                onClick={() => onMetadataChange('type', 'system')}
                                                                className={`h-8 text-xs border-white/5 ${workflowMetadata.type === 'system' ? 'bg-purple-500/10 text-purple-400 border-purple-500/30' : 'bg-[#111] text-slate-400 hover:bg-[#1a1a1a]'}`}
                                                            >
                                                                System Event
                                                            </Button>
                                                        </div>
                                                    </div>

                                                    {workflowMetadata.type === 'system' && (
                                                        <>
                                                            <div className="space-y-1.5">
                                                                <Label className="text-xs text-slate-400 font-normal">Event Type</Label>
                                                                <Input
                                                                    placeholder="e.g. cowrie.login.failed"
                                                                    value={workflowMetadata.triggerEvent || ''}
                                                                    onChange={(e) => onMetadataChange('triggerEvent', e.target.value)}
                                                                    className="bg-[#111] border-white/5 text-slate-200 text-xs h-9 focus:border-yellow-500/50"
                                                                />
                                                            </div>

                                                            <div className="space-y-1.5">
                                                                <Label className="text-xs text-slate-400 font-normal">Threshold</Label>
                                                                <Input
                                                                    type="number"
                                                                    placeholder="e.g. 5"
                                                                    value={workflowMetadata.triggerThreshold || ''}
                                                                    onChange={(e) => onMetadataChange('triggerThreshold', e.target.value)}
                                                                    className="bg-[#111] border-white/5 text-slate-200 text-xs h-9 focus:border-yellow-500/50"
                                                                />
                                                            </div>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        <div className="space-y-3">
                                            <div className="space-y-1.5">
                                                <Label className="text-xs text-slate-400 font-normal">Label</Label>
                                                <Input
                                                    value={selectedNode.data.label as string || ''}
                                                    onChange={(e) => onNodeUpdate(selectedNode.id, { ...selectedNode.data, label: e.target.value })}
                                                    className="bg-[#111] border-white/5 text-sm text-slate-200 focus:border-cyan-500/50 h-9"
                                                />
                                            </div>

                                            <div className="space-y-1.5">
                                                <Label className="text-xs text-slate-400 font-normal">Description</Label>
                                                <Textarea
                                                    value={selectedNode.data.description as string || ''}
                                                    onChange={(e) => onNodeUpdate(selectedNode.id, { ...selectedNode.data, description: e.target.value })}
                                                    className="bg-[#111] border-white/5 min-h-[100px] text-sm text-slate-200 focus:border-cyan-500/50 resize-none"
                                                />
                                            </div>

                                            {selectedNode.type === 'actionNode' && (
                                                <div className="space-y-1.5">
                                                    <Label className="text-xs text-slate-400 font-normal">Category</Label>
                                                    <Input
                                                        value={selectedNode.data.category as string || ''}
                                                        onChange={(e) => onNodeUpdate(selectedNode.id, { ...selectedNode.data, category: e.target.value })}
                                                        className="bg-[#111] border-white/5 text-sm text-slate-200 focus:border-cyan-500/50 h-9"
                                                    />
                                                </div>
                                            )}
                                        </div>

                                        <Separator className="bg-white/5" />

                                        <Button
                                            variant="ghost"
                                            className="w-full text-red-400 hover:text-red-300 hover:bg-red-950/20 h-9 text-xs"
                                            onClick={() => onDeleteNode(selectedNode.id)}
                                        >
                                            <Trash2 size={14} className="mr-2" /> Delete Node
                                        </Button>
                                    </div>
                                ) : (
                                    <div className="space-y-6">
                                        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Workflow Settings</h3>

                                        <div className="space-y-3">
                                            <div className="space-y-1.5">
                                                <Label className="text-xs text-slate-400 font-normal">Workflow Name</Label>
                                                <Input
                                                    value={workflowMetadata.name}
                                                    onChange={(e) => onMetadataChange('name', e.target.value)}
                                                    className="bg-[#111] border-white/5 text-sm text-slate-200 focus:border-cyan-500/50 h-9"
                                                />
                                            </div>

                                            <div className="space-y-1.5">
                                                <Label className="text-xs text-slate-400 font-normal">Workflow ID</Label>
                                                <div className="flex gap-2">
                                                    <Input
                                                        value={workflowMetadata.id || 'New Workflow'}
                                                        readOnly
                                                        className="bg-[#111] border-white/5 font-mono text-xs text-slate-500 h-9"
                                                    />
                                                    <Button size="icon" variant="outline" className="shrink-0 w-9 h-9 border-white/5 bg-[#111] hover:bg-[#1a1a1a] text-slate-400">
                                                        <Copy size={14} />
                                                    </Button>
                                                </div>
                                            </div>

                                            <div className="p-3 rounded bg-white/5 border border-white/5 text-xs text-slate-400">
                                                <p>Select the <span className="text-yellow-500 font-bold">Start Node</span> to configure trigger settings.</p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </ScrollArea>
                    </TabsContent>

                    <TabsContent value="code" className="mt-0 h-full">
                        <ScrollArea className="h-full">
                            <div className="p-4">
                                <div className="rounded-lg bg-[#111] border border-white/5 p-4 font-mono text-[10px] text-slate-400 overflow-auto">
                                    <pre>
                                        {JSON.stringify(selectedNode ? selectedNode : workflowMetadata, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        </ScrollArea>
                    </TabsContent>

                    <TabsContent value="templates" className="mt-0 h-full">
                        {onLoadWorkflow ? (
                            <WorkflowTemplatesTab onLoadWorkflow={onLoadWorkflow} />
                        ) : (
                            <div className="text-center py-12 text-slate-600 text-xs">
                                <Activity className="mx-auto mb-3 opacity-20" size={24} />
                                Template loading not available.
                            </div>
                        )}
                    </TabsContent>
                </div>
            </Tabs>
        </div>
    );
}
