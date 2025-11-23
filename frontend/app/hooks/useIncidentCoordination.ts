"use client";

import { useState, useEffect } from 'react';
import { apiUrl } from '@/app/utils/api';

export interface IncidentCoordination {
  incident_id: number;
  coordination_status: string;
  participating_agents: string[];
  agent_decisions: {
    attribution?: {
      threat_actor: string;
      confidence: number;
      tactics: string[];
      iocs_identified: number;
    };
    containment?: {
      actions_taken: string[];
      effectiveness: number;
      status: string;
    };
    forensics?: {
      evidence_collected: string[];
      timeline_events: number;
      suspicious_processes: number;
    };
  };
  coordination_timeline: Array<{
    timestamp: string;
    event: string;
    details: string;
    verdict?: string;
    agents?: string[];
  }>;
  recommendations: string[];
}

export function useIncidentCoordination(incidentId: number) {
  const [coordination, setCoordination] = useState<IncidentCoordination | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCoordination = async () => {
      try {
        setLoading(true);
        const response = await fetch(apiUrl(`/api/agents/incidents/${incidentId}/coordination`));
        if (response.ok) {
          const data = await response.json();
          setCoordination(data);
        } else {
          // For now, don't set error - we'll use mock data
          console.warn('Failed to fetch coordination data:', response.status);
        }
      } catch (err) {
        console.warn('Error fetching coordination data:', err);
        // Don't set error - we'll use mock data
      } finally {
        setLoading(false);
      }
    };

    if (incidentId) {
      fetchCoordination();
    }
  }, [incidentId]);

  const refreshCoordination = async () => {
    // Re-fetch coordination data
    const response = await fetch(apiUrl(`/api/agents/incidents/${incidentId}/coordination`));
    if (response.ok) {
      const data = await response.json();
      setCoordination(data);
    }
  };

  return {
    coordination,
    loading,
    error,
    refreshCoordination
  };
}
