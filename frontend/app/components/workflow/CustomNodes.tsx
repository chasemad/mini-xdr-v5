import React, { memo } from 'react';
import { Handle, Position, NodeProps, Node, useReactFlow } from '@xyflow/react';
import {
    Shield,
    Zap,
    Activity,
    Server,
    Globe,
    Mail,
    Database,
    Lock,
    AlertTriangle,
    CheckCircle2,
    Play,
    HelpCircle,
    GitBranch,
    Terminal,
    Cpu,
    X,
    Info
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

// --- Icons Mapping ---
const iconMap: Record<string, any> = {
    network: Globe,
    endpoint: Server,
    email: Mail,
    data: Database,
    identity: Lock,
    security: Shield,
    trigger: Zap,
    condition: GitBranch,
    analysis: Cpu,
    response: Terminal,
    default: Activity
};

const getIcon = (category: string) => {
    return iconMap[category?.toLowerCase()] || iconMap.default;
};

// --- Color Mapping (Neon/Dark) ---
const colorMap: Record<string, { border: string, glow: string, icon: string, bg: string, accent: string }> = {
    network: { border: 'border-cyan-500/30', glow: 'shadow-cyan-500/10', icon: 'text-cyan-400', bg: 'bg-[#111]', accent: 'bg-cyan-500' },
    endpoint: { border: 'border-blue-500/30', glow: 'shadow-blue-500/10', icon: 'text-blue-400', bg: 'bg-[#111]', accent: 'bg-blue-500' },
    email: { border: 'border-purple-500/30', glow: 'shadow-purple-500/10', icon: 'text-purple-400', bg: 'bg-[#111]', accent: 'bg-purple-500' },
    security: { border: 'border-red-500/30', glow: 'shadow-red-500/10', icon: 'text-red-400', bg: 'bg-[#111]', accent: 'bg-red-500' },
    trigger: { border: 'border-yellow-500/30', glow: 'shadow-yellow-500/10', icon: 'text-yellow-400', bg: 'bg-[#111]', accent: 'bg-yellow-500' },
    condition: { border: 'border-orange-500/30', glow: 'shadow-orange-500/10', icon: 'text-orange-400', bg: 'bg-[#111]', accent: 'bg-orange-500' },
    analysis: { border: 'border-pink-500/30', glow: 'shadow-pink-500/10', icon: 'text-pink-400', bg: 'bg-[#111]', accent: 'bg-pink-500' },
    default: { border: 'border-slate-500/30', glow: 'shadow-slate-500/10', icon: 'text-slate-400', bg: 'bg-[#111]', accent: 'bg-slate-500' }
};

const getColor = (category: string) => {
    return colorMap[category?.toLowerCase()] || colorMap.default;
};

interface ActionNodeData extends Record<string, unknown> {
    label: string;
    category: string;
    description?: string;
    status?: string;
}

interface TriggerNodeData extends Record<string, unknown> {
    label: string;
}

interface ConditionNodeData extends Record<string, unknown> {
    label: string;
}

// --- Action Node (Compact Card) ---
export const ActionNode = memo(({ data, selected, id }: NodeProps<Node<ActionNodeData>>) => {
    const colors = getColor(data.category as string);
    const Icon = getIcon(data.category as string);
    const { deleteElements } = useReactFlow();

    const handleDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        deleteElements({ nodes: [{ id }] });
    };

    return (
        <TooltipProvider>
            <Tooltip delayDuration={300}>
                <TooltipTrigger asChild>
                    <div className={`
                        relative min-w-[180px] rounded-lg border bg-[#0A0A0A] transition-all duration-200 group
                        ${selected ? `border-white/20 ring-1 ring-white/20 shadow-lg ${colors.glow}` : 'border-white/5 hover:border-white/10'}
                    `}>
                        {/* Accent Line */}
                        <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg ${colors.accent}`} />

                        {/* Delete Button */}
                        <button
                            onClick={handleDelete}
                            className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-red-500/90 hover:bg-red-500 text-white opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center shadow-lg z-10"
                        >
                            <X size={12} />
                        </button>

                        {/* Input Handle */}
                        <Handle
                            type="target"
                            position={Position.Top}
                            className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] group-hover:!bg-white transition-colors"
                        />

                        <div className="p-3 pl-4 flex items-start gap-3">
                            <div className={`mt-0.5 ${colors.icon}`}>
                                <Icon size={16} />
                            </div>
                            <div>
                                <div className="text-xs font-semibold text-slate-200 leading-tight">
                                    {data.label as string}
                                </div>
                                <div className="text-[10px] uppercase tracking-wider font-medium text-slate-500 mt-0.5">
                                    {(data.category as string) || 'Action'}
                                </div>
                            </div>
                        </div>

                        {/* Status Indicator */}
                        {data.status && (
                            <div className="absolute -top-1 -right-1">
                                {data.status === 'running' && <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]" />}
                                {data.status === 'completed' && <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />}
                                {data.status === 'failed' && <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]" />}
                            </div>
                        )}

                        {/* Output Handle */}
                        <Handle
                            type="source"
                            position={Position.Bottom}
                            className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] group-hover:!bg-white transition-colors"
                        />
                    </div>
                </TooltipTrigger>
                <TooltipContent side="right" className="bg-slate-900 border-white/10 text-slate-200 max-w-xs">
                    <div className="space-y-1">
                        <div className="font-semibold text-xs">{data.label as string}</div>
                        {data.description && (
                            <div className="text-[10px] text-slate-400">{data.description as string}</div>
                        )}
                        <div className="text-[10px] text-slate-500 italic">Click to configure • Hover Delete to remove</div>
                    </div>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
});

// --- Trigger Node (Minimal Pill) ---
export const TriggerNode = memo(({ data, selected }: NodeProps<Node<TriggerNodeData>>) => {
    return (
        <div className={`
            relative px-4 py-2 rounded-full border bg-[#0A0A0A] transition-all duration-200 flex items-center gap-2 group
            ${selected ? 'border-yellow-500/50 shadow-[0_0_15px_rgba(234,179,8,0.2)]' : 'border-white/10 hover:border-yellow-500/30'}
        `}>
            <div className="text-yellow-500">
                <Zap size={14} fill="currentColor" />
            </div>
            <div className="text-xs font-bold text-slate-200 tracking-wide">
                {(data.label as string) || 'START'}
            </div>

            <Handle
                type="source"
                position={Position.Bottom}
                className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] group-hover:!bg-yellow-500 transition-colors"
            />
        </div>
    );
});

// --- Condition Node (Sharp Diamond) ---
export const ConditionNode = memo(({ data, selected }: NodeProps<Node<ConditionNodeData>>) => {
    return (
        <div className="relative w-20 h-20 flex items-center justify-center group">
            <div className={`
                absolute w-14 h-14 rotate-45 border bg-[#0A0A0A] transition-all duration-200
                ${selected ? 'border-orange-500/50 shadow-[0_0_15px_rgba(249,115,22,0.2)]' : 'border-white/10 hover:border-orange-500/30'}
            `}>
                <div className="-rotate-45 flex items-center justify-center h-full w-full">
                    <GitBranch size={16} className="text-orange-500" />
                </div>
            </div>

            <div className="absolute -bottom-6 text-[10px] font-medium text-slate-500 uppercase tracking-wider text-center w-32">
                {(data.label as string) || 'Condition'}
            </div>

            <Handle type="target" position={Position.Top} className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] !-mt-8 group-hover:!bg-white" />

            {/* True/False Handles */}
            <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-600">NO</div>
            <Handle id="false" type="source" position={Position.Left} className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] !-ml-8 group-hover:!bg-red-500" />

            <div className="absolute -right-8 top-1/2 -translate-y-1/2 text-[9px] font-bold text-slate-600">YES</div>
            <Handle id="true" type="source" position={Position.Right} className="!bg-[#333] !w-2 !h-2 !border-2 !border-[#0A0A0A] !-mr-8 group-hover:!bg-emerald-500" />
        </div>
    );
});
