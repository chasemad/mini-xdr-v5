export interface Action {
    id: string;
    name: string;
    description: string;
    scope: {
        type: 'entity' | 'segment' | 'policy' | 'global';
        target: string;
    };
    parameters?: Record<string, any>;
    expectedImpact: string;
    riskLevel: 'read-only' | 'low' | 'medium' | 'high' | 'critical';
    requiresApproval: boolean;
    agent: string; // e.g., "Hunter", "Containment"
    estimatedDuration: string;
    rollbackPath?: string;
    status: 'pending' | 'awaiting_approval' | 'executing' | 'completed' | 'failed' | 'rolled_back';
    timestamp: string;
    outputs?: {
        destination: string;
        link?: string;
    }[];
    result_data?: any;
    error_details?: any;
    // Legacy/Compatibility fields (optional)
    action?: string; // mapping for legacy 'action' name
    action_name?: string; // mapping for legacy 'action_name'
    created_at?: string; // mapping for legacy timestamp
    detail?: string; // mapping for legacy detail
    agent_type?: string; // mapping for legacy agent type
    execution_method?: string;
    confidence_score?: number;
    rollback_id?: string;
}
