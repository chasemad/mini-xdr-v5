'use client'

import React, { useCallback, useState, useMemo, useRef, useEffect } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    Edge,
    Node,
    BackgroundVariant,
    Panel,
    useReactFlow,
    ReactFlowProvider
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ActionNode, TriggerNode, ConditionNode } from './workflow/CustomNodes';
import { FloatingCommandBar } from './FloatingCommandBar';
import { Button } from "@/components/ui/button";
import { Play, Save, Settings, Maximize2, Plus, LayoutGrid } from 'lucide-react';
import { parseNlpWorkflow, saveWorkflow, runWorkflow } from '@/app/lib/api';
import { toast } from "@/components/ui/toast";
import { WorkflowPropertiesPanel } from './WorkflowPropertiesPanel';
import { WorkflowLoadDialog } from './WorkflowLoadDialog';
import { getWorkflow, Workflow } from './workflow/workflow-client';
import ActionNodeLibrary from './ActionNodeLibrary';
import { WORKFLOW_ACTIONS } from './workflow/workflow-actions';

// --- Types ---
const nodeTypes = {
    actionNode: ActionNode,
    triggerNode: TriggerNode,
    conditionNode: ConditionNode,
};

const defaultViewport = { x: 0, y: 0, zoom: 1 };

// Actions are now imported from WORKFLOW_ACTIONS

interface WorkflowCanvasProps {
    incidentId?: number;
}

function WorkflowCanvasContent({ incidentId }: WorkflowCanvasProps) {
    // State
    const [nodes, setNodesInternal, onNodesChange] = useNodesState<Node>([]);

    // Wrapper to ensure all nodes have valid positions
    const setNodes = useCallback((nodesOrUpdater: Node[] | ((nodes: Node[]) => Node[])) => {
        setNodesInternal((currentNodes) => {
            const newNodes = typeof nodesOrUpdater === 'function'
                ? nodesOrUpdater(currentNodes)
                : nodesOrUpdater;

            // Validate and fix any nodes with invalid positions
            return newNodes.map((node, index) => ({
                ...node,
                position: node.position && typeof node.position.x === 'number' && typeof node.position.y === 'number'
                    ? node.position
                    : { x: 100, y: 100 + (index * 150) } // Default fallback position
            }));
        });
    }, [setNodesInternal]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [showLibrary, setShowLibrary] = useState(false);

    const [selectedNode, setSelectedNode] = useState<Node | null>(null);
    const [workflowMetadata, setWorkflowMetadata] = useState<{ name: string; id?: string; type?: 'incident' | 'system'; triggerEvent?: string; triggerThreshold?: string }>({ name: 'New Workflow', id: '', type: 'incident' });

    const reactFlowWrapper = useRef<HTMLDivElement>(null);
    const reactFlowInstance = useReactFlow();

    // Safe access to screenToFlowPosition
    const screenToFlowPosition = reactFlowInstance?.screenToFlowPosition || (() => ({ x: 0, y: 0 }));

    // Handlers
    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#22d3ee', strokeWidth: 2 } } as Edge, eds)),
        [setEdges],
    );

    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    }, []);

    const onPaneClick = useCallback(() => {
        setSelectedNode(null);
    }, []);

    const handleUpdateNode = (nodeId: string, data: Record<string, unknown>) => {
        setNodes((nds) => nds.map((node) => {
            if (node.id === nodeId) {
                return { ...node, data: { ...node.data, ...data } };
            }
            return node;
        }));
        setSelectedNode((prev) => prev && prev.id === nodeId ? { ...prev, data: { ...prev.data, ...data } } : prev);
    };

    const handleDeleteNode = (nodeId: string) => {
        setNodes((nds) => nds.filter((n) => n.id !== nodeId));
        setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
        setSelectedNode(null);
    };

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            // Delete or Backspace to remove selected node
            if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNode) {
                // Prevent deletion if user is typing in an input field
                if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
                    return;
                }
                event.preventDefault();
                handleDeleteNode(selectedNode.id);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedNode]);

    // Drag and Drop Handlers
    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow');

            // check if the dropped element is valid
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const actionData = WORKFLOW_ACTIONS[type as keyof typeof WORKFLOW_ACTIONS];
            if (!actionData) return;

            // Safety check for screenToFlowPosition
            if (!reactFlowWrapper.current) {
                console.warn('ReactFlow wrapper not ready');
                return;
            }

            const position = screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            // Additional safety check for position
            if (!position || typeof position.x === 'undefined') {
                console.warn('Invalid position from screenToFlowPosition');
                return;
            }

            const newNode: Node = {
                id: `node-${Date.now()}`,
                type: actionData.category === 'trigger' ? 'triggerNode' : 'actionNode',
                position,
                data: {
                    label: actionData.name,
                    category: actionData.category,
                    description: actionData.description,
                    status: 'pending'
                },
            };

            setNodes((nds) => nds.concat(newNode));
        },
        [screenToFlowPosition, setNodes, reactFlowWrapper],
    );

    // NLP Handler
    const handleCommandSubmit = async (command: string) => {
        setIsProcessing(true);
        try {
            // 1. Call API to parse intent
            const result = await parseNlpWorkflow({
                text: command,
                auto_execute: false
            });

            if (result) {
                const newNodes: Node[] = [];
                const newEdges: Edge[] = [];

                // Check for visual graph from backend (Graph Mode)
                if (result.visual_graph && result.visual_graph.nodes) {
                    // Map backend nodes to React Flow nodes
                    result.visual_graph.nodes.forEach((node: Record<string, any>) => {
                        let type = 'actionNode';
                        if (node.type === 'condition') type = 'conditionNode';
                        if (node.type === 'trigger') type = 'triggerNode';

                        newNodes.push({
                            id: node.id,
                            type,
                            position: { x: 0, y: 0 }, // Layout will need to be handled or auto-layouted
                            data: {
                                label: node.label,
                                category: node.category,
                                description: node.description,
                                status: 'pending'
                            }
                        });
                    });

                    // Map backend edges
                    result.visual_graph.edges.forEach((edge: Record<string, any>) => {
                        newEdges.push({
                            id: `edge-${edge.source}-${edge.target}`,
                            source: edge.source,
                            target: edge.target,
                            label: edge.label,
                            animated: true,
                            style: { stroke: '#22d3ee', strokeWidth: 2 }
                        });
                    });

                    // Simple auto-layout (vertical stack for now, can be improved with dagre)
                    let yOffset = 50;
                    newNodes.forEach((node) => {
                        node.position = { x: 400, y: yOffset };
                        yOffset += 150;
                    });

                } else if (result.actions) {
                    // Fallback: Linear Mode from actions list
                    let yOffset = 100;

                    // Add Start Node if empty
                    if (nodes.length === 0) {
                        newNodes.push({
                            id: 'start',
                            type: 'triggerNode',
                            position: { x: 400, y: 50 },
                            data: { label: 'Manual Trigger' }
                        });
                        yOffset += 150;
                    }

                    // Add Action Nodes
                    result.actions.forEach((action: Record<string, any>, index: number) => {
                        const nodeId = `node-${Date.now()}-${index}`;
                        newNodes.push({
                            id: nodeId,
                            type: 'actionNode',
                            position: { x: 400, y: yOffset },
                            data: {
                                label: action.description || action.action_type,
                                category: action.category,
                                description: action.description,
                                status: 'pending'
                            }
                        });

                        // Connect to previous
                        const prevNodeId = index === 0 ? (nodes.length > 0 ? nodes[nodes.length - 1].id : 'start') : `node-${Date.now()}-${index - 1}`;
                        newEdges.push({
                            id: `edge-${Date.now()}-${index}`,
                            source: prevNodeId,
                            target: nodeId,
                            animated: true,
                            style: { stroke: '#22d3ee', strokeWidth: 2 }
                        });

                        yOffset += 150;
                    });
                }

                // Update Graph
                setNodes((nds) => [...nds, ...newNodes]);
                setEdges((eds) => [...eds, ...newEdges]);
            }
        } catch (error) {
            console.error("Failed to parse workflow:", error);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSave = async () => {
        try {
            const payload: any = {
                incident_id: incidentId || null,
                name: workflowMetadata.name,
                graph: { nodes, edges }
            };

            if (workflowMetadata.type === 'system') {
                payload.trigger_config = {
                    event_type: workflowMetadata.triggerEvent,
                    threshold: parseInt(workflowMetadata.triggerThreshold || '1'),
                    window_seconds: 60 // Default
                };
            }

            await saveWorkflow(payload);
            toast.success("Workflow Saved", "Your visual workflow has been saved successfully.");
        } catch (error) {
            console.error("Save failed:", error);
            toast.error("Save Failed", "Could not save the workflow.");
        }
    };

    const handleRun = async () => {
        if (!incidentId && workflowMetadata.type !== 'system') {
            toast.error("Execution Context Missing", "Cannot run incident workflow without an active incident context.");
            return;
        }

        setIsProcessing(true);
        try {
            await runWorkflow({
                incident_id: incidentId || null, // Can be null for system workflows
                graph: { nodes, edges }
            });
            toast.success("Workflow Started", "Execution has started in the background.");
        } catch (error) {
            console.error("Run failed:", error);
            toast.error("Execution Failed", "Could not start the workflow.");
        } finally {
            setIsProcessing(false);
        }
    };

    const handleLoadWorkflow = async (wf: Workflow) => {
        try {
            const fullWf = await getWorkflow(wf.id);
            if (fullWf.graph) {
                // Ensure all nodes have valid position data
                const validatedNodes = (fullWf.graph.nodes || []).map((node: Node, index: number) => ({
                    ...node,
                    position: node.position && typeof node.position.x === 'number' && typeof node.position.y === 'number'
                        ? node.position
                        : { x: 100, y: 100 + (index * 150) } // Default position if missing
                }));

                setNodes(validatedNodes);
                setEdges(fullWf.graph.edges || []);
                setWorkflowMetadata({ name: fullWf.name, id: fullWf.id });
                toast.success("Workflow Loaded", `Loaded ${fullWf.name}`);
            }
        } catch (e) {
            console.error(e);
            toast.error("Load Failed", "Could not load workflow details.");
        }
    };

    return (
        <div className="w-full h-screen bg-[#0A0A0A] flex overflow-hidden font-sans">
            {/* Main Canvas Area */}
            <div className="flex-1 relative h-full flex flex-col overflow-hidden">

                {/* Top Header Bar */}
                <div className="h-14 border-b border-white/5 bg-[#0A0A0A] flex items-center justify-between px-4 z-20 shrink-0">
                    <div className="flex items-center gap-4">
                        <div className="text-sm font-medium text-slate-400">
                            Workflow / <span className="text-slate-200">{workflowMetadata.name}</span>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        <WorkflowLoadDialog onLoad={handleLoadWorkflow} />
                        <Button
                            variant="ghost"
                            size="sm"
                            className="text-slate-400 hover:text-white hover:bg-white/5"
                            onClick={handleSave}
                        >
                            <Save size={16} className="mr-2" /> Save
                        </Button>
                        <Button
                            size="sm"
                            className="bg-cyan-500 hover:bg-cyan-400 text-black font-semibold shadow-[0_0_15px_rgba(6,182,212,0.3)]"
                            onClick={handleRun}
                        >
                            <Play size={16} className="mr-2 fill-current" /> Run
                        </Button>
                    </div>
                </div>

                <div className="flex-1 relative w-full h-full" ref={reactFlowWrapper}>
                    {/* Floating Command Bar - Moved to Top */}
                    <div className="absolute top-6 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-30">
                        <FloatingCommandBar
                            onCommandSubmit={handleCommandSubmit}
                            isProcessing={isProcessing}
                        />
                    </div>

                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        onNodeClick={onNodeClick}
                        onPaneClick={onPaneClick}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                        nodeTypes={nodeTypes}
                        defaultViewport={defaultViewport}
                        minZoom={0.5}
                        maxZoom={2}
                        fitView
                        className="bg-[#0A0A0A]"
                    >
                        <Background
                            variant={BackgroundVariant.Dots}
                            gap={24}
                            size={1}
                            color="#222"
                        />

                        <Controls className="bg-[#111] border-white/5 text-slate-400 fill-slate-400" />

                        <MiniMap
                            className="bg-[#111] border-white/5"
                            nodeColor={(n) => {
                                if (n.type === 'triggerNode') return '#eab308';
                                if (n.type === 'actionNode') return '#06b6d4';
                                return '#333';
                            }}
                        />
                    </ReactFlow>
                </div>
            </div>

            {/* Right Sidebar */}
            <WorkflowPropertiesPanel
                selectedNode={selectedNode}
                workflowMetadata={workflowMetadata}
                onMetadataChange={(k: string, v: string) => setWorkflowMetadata(prev => ({ ...prev, [k]: v }))}
                onNodeUpdate={handleUpdateNode}
                onDeleteNode={handleDeleteNode}
                onLoadWorkflow={handleLoadWorkflow}
                availableActions={WORKFLOW_ACTIONS}
            />
        </div>
    );
}

export default function WorkflowCanvas(props: WorkflowCanvasProps) {
    return (
        <ReactFlowProvider>
            <WorkflowCanvasContent {...props} />
        </ReactFlowProvider>
    );
}
