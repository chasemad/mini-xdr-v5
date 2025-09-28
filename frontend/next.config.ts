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
    
    // Development CSP allows unsafe directives for Next.js hot reloading
    const devCSP = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' http://localhost:8000 http://54.237.168.3:8000 ws://localhost:8000 ws://54.237.168.3:8000 wss://localhost:8000 wss://54.237.168.3:8000;"
    
    // Production CSP is more restrictive
    const prodCSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' http://localhost:8000 http://54.237.168.3:8000 ws://localhost:8000 ws://54.237.168.3:8000 wss://localhost:8000 wss://54.237.168.3:8000;"

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
            value: 'max-age=31536000; includeSubDomains'
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
