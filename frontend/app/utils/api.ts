/**
 * Centralized API configuration and utilities
 * Ensures consistent API URL usage across the application
 */

// Normalize environment variables while preserving intentional empty strings
const sanitizeEnvValue = (value: string | undefined): string | undefined => {
  if (value === undefined) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed;
};

/**
 * Resolve the raw API base URL from environment variables.
 * Returns the value as-is (including empty string) when explicitly provided.
 */
export const resolveApiBaseUrl = (): string | null => {
  const apiUrl = sanitizeEnvValue(process.env.NEXT_PUBLIC_API_URL);
  if (apiUrl !== undefined) {
    return apiUrl;
  }

  const apiBase = sanitizeEnvValue(process.env.NEXT_PUBLIC_API_BASE);
  if (apiBase !== undefined) {
    return apiBase;
  }

  return null;
};

// Get API base URL from environment variable with fallback
export const getApiBaseUrl = (): string => {
  const resolved = resolveApiBaseUrl();
  if (resolved !== null) {
    return resolved;
  }

  // Default to local backend for developer convenience
  return 'http://localhost:8000';
};

// API base URL (cached)
export const API_BASE_URL = getApiBaseUrl();

/**
 * Create a full API endpoint URL
 */
export const apiUrl = (endpoint: string, baseUrl: string = API_BASE_URL): string => {
  // Remove leading slash if present (we'll add it)
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;

  // If endpoint already includes the full URL, return as-is
  if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
    return endpoint;
  }

  // Remove trailing slash from base URL if present
  const normalizedBase = (baseUrl || '').replace(/\/+$/, '');

  // Empty base means intentionally use relative URL
  if (!normalizedBase) {
    return cleanEndpoint;
  }

  return `${normalizedBase}${cleanEndpoint}`;
};

/**
 * Get API key from environment
 */
export const getApiKey = (): string => {
  // Fallback to demo key so background probes (e.g., block-status) don't 401 in dev/demo
  return process.env.NEXT_PUBLIC_API_KEY || 'demo-minixdr-api-key';
};

/**
 * Create fetch options with default headers
 */
export const createFetchOptions = (options: RequestInit = {}): RequestInit => {
  const apiKey = getApiKey();

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  if (apiKey) {
    headers['x-api-key'] = apiKey;
  }

  return {
    ...options,
    headers,
  };
};
