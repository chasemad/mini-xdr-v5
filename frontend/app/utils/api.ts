/**
 * Centralized API configuration and utilities
 * Ensures consistent API URL usage across the application
 */

// Get API base URL from environment variable with fallback
export const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    // Client-side: use environment variable or current origin
    return process.env.NEXT_PUBLIC_API_URL ||
           process.env.NEXT_PUBLIC_API_BASE ||
           (typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:8000` : 'http://localhost:8000');
  }

  // Server-side: use environment variable or default
  return process.env.NEXT_PUBLIC_API_URL ||
         process.env.NEXT_PUBLIC_API_BASE ||
         'http://mini-xdr-backend-service:8000';
};

// API base URL (cached)
export const API_BASE_URL = getApiBaseUrl();

/**
 * Create a full API endpoint URL
 */
export const apiUrl = (endpoint: string): string => {
  // Remove leading slash if present (we'll add it)
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;

  // If endpoint already includes the full URL, return as-is
  if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
    return endpoint;
  }

  // Remove trailing slash from base URL if present
  const base = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL;

  return `${base}${cleanEndpoint}`;
};

/**
 * Get API key from environment
 */
export const getApiKey = (): string => {
  return process.env.NEXT_PUBLIC_API_KEY || '';
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
