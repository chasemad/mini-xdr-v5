/**
 * API client and TypeScript models for seamless onboarding v2 endpoints.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "${API_BASE}";

// ---------------------------------------------------------------------------
// Shared Types
// ---------------------------------------------------------------------------

export interface CloudCredentials {
  role_arn?: string;
  external_id?: string;
  aws_access_key_id?: string;
  aws_secret_access_key?: string;
  aws_session_token?: string;
  client_id?: string;
  client_secret?: string;
  tenant_id?: string;
  service_account_key?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface QuickStartRequest {
  provider: string;
  credentials: CloudCredentials;
}

export interface QuickStartResponse {
  status: "initiated" | "error";
  message: string;
  provider: string;
  organization_id: number;
}

export interface ProgressResponse {
  discovery: {
    status: "pending" | "running" | "completed" | "error";
    progress: number;
    assets_found: number;
    message: string;
  };
  deployment: {
    status: "pending" | "running" | "completed" | "error";
    progress: number;
    agents_deployed: number;
    total_assets: number;
    message: string;
  };
  validation: {
    status: "pending" | "running" | "completed" | "error";
    progress: number;
    checks_passed: number;
    total_checks: number;
    message: string;
  };
  overall_status: "pending" | "running" | "completed" | "error";
  estimated_completion_time?: string;
}

export interface CloudAsset {
  id: string;
  provider: string;
  asset_type: string;
  asset_id: string;
  region: string | null;
  asset_data: Record<string, unknown>;
  tags?: Record<string, string>;
  discovered_at?: string | null;
  last_seen_at?: string | null;
  agent_deployed: boolean;
  agent_status: string;
  priority?: string;
  last_heartbeat_at?: string | null;
}

export interface AssetsResponse {
  total: number;
  assets: CloudAsset[];
  provider?: string;
}

export interface DeploymentSummary {
  total_assets: number;
  agent_deployed: number;
  deployment_pending: number;
  deployment_failed: number;
  by_provider: Record<string, number>;
  by_status: Record<string, number>;
  by_priority?: Record<string, number>;
}

export interface ValidationCheck {
  check_name: string;
  status: "pass" | "fail" | "pending";
  message: string;
  details?: unknown;
}

export interface ValidationSummary {
  checks: ValidationCheck[];
  overall_status: "pending" | "running" | "completed" | "error";
  timestamp: string;
}

export interface IntegrationListItem {
  provider: string;
  status: string;
  credential_type: string | null;
  configured_at: string | null;
  last_used_at: string | null;
  expires_at?: string | null;
  last_tested_at: string | null;
  last_test_status: string | null;
  last_test_latency_ms: number | null;
  health_summary: Record<string, unknown> | null;
  permission_summary: Record<string, unknown> | null;
  assets_count: number;
  is_stale: boolean;
}

export interface IntegrationListResponse {
  total: number;
  integrations: IntegrationListItem[];
}

export interface IntegrationDetail extends IntegrationListItem {
  last_test_error?: string | null;
  settings: Record<string, unknown>;
}

export interface IntegrationTestResult {
  status: string;
  latency_ms: number;
  error?: string | null;
  permission_summary?: Record<string, unknown> | null;
  health_summary?: Record<string, unknown> | null;
  tested_at: string;
}

export interface AssetSummaryResponse {
  total_assets: number;
  agent_deployed: number;
  deployment_pending: number;
  deployment_failed: number;
  incompatible: number;
  coverage_percent: number;
  by_provider: Record<string, number>;
  by_status: Record<string, number>;
  by_priority: Record<string, number>;
  last_discovery_run?: string | null;
}

export interface AssetActivityEntry {
  asset_id: string;
  provider: string;
  region?: string | null;
  priority?: string | null;
  agent_status?: string | null;
  agent_deployed: boolean;
  last_heartbeat_at?: string | null;
  heartbeat_age_seconds?: number | null;
  events_last_24h: number;
}

export interface AssetActivityResponse {
  total: number;
  page: number;
  page_size: number;
  items: AssetActivityEntry[];
}

export interface AssetMapBucket {
  provider: string;
  region: string | null;
  asset_count: number;
}

export interface AssetMapResponse {
  total_regions: number;
  buckets: AssetMapBucket[];
}

interface UpdateIntegrationPayload {
  credentials?: CloudCredentials;
  settings?: Record<string, unknown>;
  agentPublicBaseUrl?: string;
}

class OnboardingV2API {
  private getAuthHeaders(): HeadersInit {
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
    return {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...this.getAuthHeaders(),
        ...(init.headers || {}),
      },
    });

    if (!response.ok) {
      const errorPayload = await response
        .json()
        .catch(() => ({ message: "Network error" }));
      const message = (errorPayload as Record<string, unknown>).message ?? `HTTP ${response.status}`;
      throw new Error(String(message));
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return (await response.json()) as T;
  }

  // -------------------------------------------------------------------------
  // Onboarding lifecycle endpoints
  // -------------------------------------------------------------------------

  async quickStart(data: QuickStartRequest): Promise<QuickStartResponse> {
    return this.request<QuickStartResponse>("/api/onboarding/v2/quick-start", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getProgress(): Promise<ProgressResponse> {
    return this.request<ProgressResponse>("/api/onboarding/v2/progress");
  }

  async getAssets(provider?: string): Promise<AssetsResponse> {
    const url = new URL(`/api/onboarding/v2/assets`, API_BASE);
    if (provider) {
      url.searchParams.set("provider", provider);
    }
    return this.request<AssetsResponse>(url.pathname + url.search);
  }

  async refreshAssets(provider: string): Promise<{ status: string; provider: string; message: string }> {
    return this.request<{ status: string; provider: string; message: string }>(
      "/api/onboarding/v2/assets/refresh",
      {
        method: "POST",
        body: JSON.stringify({ provider }),
      }
    );
  }

  async getAssetSummary(): Promise<AssetSummaryResponse> {
    return this.request<AssetSummaryResponse>("/api/onboarding/v2/assets/summary");
  }

  async getAssetActivity(page = 1, pageSize = 25, provider?: string): Promise<AssetActivityResponse> {
    const url = new URL(`/api/onboarding/v2/assets/activity`, API_BASE);
    url.searchParams.set("page", page.toString());
    url.searchParams.set("page_size", pageSize.toString());
    if (provider) {
      url.searchParams.set("provider", provider);
    }
    return this.request<AssetActivityResponse>(url.pathname + url.search);
  }

  async getAssetMap(): Promise<AssetMapResponse> {
    return this.request<AssetMapResponse>("/api/onboarding/v2/assets/map");
  }

  async getDeploymentSummary(): Promise<DeploymentSummary> {
    return this.request<DeploymentSummary>("/api/onboarding/v2/deployment/summary");
  }

  async retryDeployment(provider: string): Promise<{ status: string; provider: string; message: string }> {
    return this.request<{ status: string; provider: string; message: string }>(
      "/api/onboarding/v2/deployment/retry",
      {
        method: "POST",
        body: JSON.stringify({ provider }),
      }
    );
  }

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.request<{ status: string; timestamp: string }>("/api/onboarding/v2/deployment/health");
  }

  async getValidationSummary(): Promise<ValidationSummary> {
    return this.request<ValidationSummary>("/api/onboarding/v2/validation/summary");
  }

  // -------------------------------------------------------------------------
  // Integration management endpoints
  // -------------------------------------------------------------------------

  async getIntegrations(): Promise<IntegrationListResponse> {
    return this.request<IntegrationListResponse>("/api/onboarding/v2/integrations");
  }

  async getIntegrationDetail(provider: string): Promise<IntegrationDetail> {
    return this.request<IntegrationDetail>(`/api/onboarding/v2/integrations/${provider}`);
  }

  async setupIntegration(provider: string, credentials: CloudCredentials): Promise<{ status: string; provider: string; message: string }> {
    return this.request<{ status: string; provider: string; message: string }>(
      "/api/onboarding/v2/integrations/setup",
      {
        method: "POST",
        body: JSON.stringify({ provider, credentials }),
      }
    );
  }

  async testIntegration(provider: string, credentials?: CloudCredentials): Promise<IntegrationTestResult> {
    return this.request<IntegrationTestResult>(`/api/onboarding/v2/integrations/${provider}/test`, {
      method: "POST",
      body: JSON.stringify({ credentials }),
    });
  }

  async updateIntegration(provider: string, payload: UpdateIntegrationPayload): Promise<IntegrationDetail> {
    const { credentials, settings, agentPublicBaseUrl } = payload;
    const body: Record<string, unknown> = {};
    if (credentials) {
      body.credentials = credentials;
    }
    if (settings) {
      body.settings = settings;
    }
    if (agentPublicBaseUrl) {
      body.agent_public_base_url = agentPublicBaseUrl;
    }

    return this.request<IntegrationDetail>(`/api/onboarding/v2/integrations/${provider}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
  }

  async setAgentPublicUrl(provider: string, url: string): Promise<IntegrationDetail> {
    return this.updateIntegration(provider, { agentPublicBaseUrl: url });
  }

  async removeIntegration(provider: string): Promise<{ status: string; message: string; provider: string }> {
    return this.request<{ status: string; message: string; provider: string }>(
      `/api/onboarding/v2/integrations/${provider}`,
      {
        method: "DELETE",
      }
    );
  }
}

export const onboardingV2API = new OnboardingV2API();
