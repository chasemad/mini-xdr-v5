const API_BASE = 'http://localhost:8000/api/workflows';

export interface Workflow {
    id: number;
    name: string;
    incident_id: number | null;
    created_at: string;
    status: string;
    graph?: { nodes: Record<string, unknown>[], edges: Record<string, unknown>[] };
}

export const getWorkflows = async (incidentId?: number): Promise<Workflow[]> => {
    const params = new URLSearchParams();
    if (incidentId) params.append('incident_id', incidentId.toString());

    const response = await fetch(`${API_BASE}/?${params.toString()}`);
    if (!response.ok) throw new Error('Failed to fetch workflows');
    return response.json();
};

export const getWorkflow = async (id: number) => {
    const response = await fetch(`${API_BASE}/${id}`);
    if (!response.ok) throw new Error('Failed to fetch workflow');
    return response.json();
};

export const deleteWorkflow = async (id: number): Promise<void> => {
    const response = await fetch(`${API_BASE}/${id}`, {
        method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete workflow');
};

export const updateWorkflow = async (id: number, data: Partial<Workflow>): Promise<Workflow> => {
    const response = await fetch(`${API_BASE}/${id}`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    if (!response.ok) throw new Error('Failed to update workflow');
    return response.json();
};
