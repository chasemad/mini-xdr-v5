import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standard Next.js app configuration for dynamic applications
  // Removed static export since this is a real-time XDR system
  
  // Keep image optimization enabled for better performance - SECURITY HARDENED
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'cdn.jsdelivr.net', // Specific CDN for chart libraries
      },
      {
        protocol: 'https',
        hostname: 'api.github.com', // GitHub API for threat intel
      }
    ],
  },
  
  // Security headers for XDR protection
  async headers() {
    const isDevelopment = process.env.NODE_ENV === 'development'
    
    // Enhanced Development CSP - Secure while allowing Next.js development features
    const devCSP = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https: http://localhost:8000; connect-src 'self' http://localhost:8000 http://54.237.168.3:8000 ws://localhost:8000 ws://54.237.168.3:8000 wss://localhost:8000 wss://54.237.168.3:8000 https://api.github.com https://cdn.jsdelivr.net; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none';"

    // Production CSP - Maximum security for enterprise deployment
    const prodCSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https:; connect-src 'self' https://54.237.168.3:8000 wss://54.237.168.3:8000 https://api.github.com https://cdn.jsdelivr.net; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; upgrade-insecure-requests;"

    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Content-Security-Policy',
            value: isDevelopment ? devCSP : prodCSP
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=(), payment=()'
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=31536000; includeSubDomains; preload'
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp'
          },
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin'
          },
          {
            key: 'Cross-Origin-Resource-Policy',
            value: 'cross-origin'
          },
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'off'
          }
        ]
      }
    ]
  },

  // Environment-specific configuration
  async rewrites() {
    return [];
  },
};

export default nextConfig;
