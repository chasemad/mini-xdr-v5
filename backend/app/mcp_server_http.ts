#!/usr/bin/env node

/**
 * Mini-XDR MCP Server - Production HTTP Wrapper
 *
 * Supports both stdio (for local Claude Code integration) and HTTP transports
 * for Kubernetes or remote deployments and remote AI assistant access.
 *
 * Usage:
 *   - stdio mode: npm run mcp (default)
 *   - HTTP mode:  TRANSPORT=http PORT=3001 npm run mcp:http
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import http from "http";
import fetch from "node-fetch";

// Import the main XDRMCPServer class (we'll need to refactor slightly)
// For now, inline the configuration

const TRANSPORT = process.env.TRANSPORT || "stdio"; // "stdio" | "http" | "sse"
const PORT = parseInt(process.env.PORT || "3001");
const API_BASE = process.env.API_BASE || "http://localhost:8000";
const API_KEY = process.env.API_KEY || "";

console.error(`🚀 Starting Mini-XDR MCP Server`);
console.error(`   Transport: ${TRANSPORT}`);
console.error(`   API Base: ${API_BASE}`);
if (TRANSPORT !== "stdio") {
  console.error(`   Port: ${PORT}`);
}

/**
 * HTTP Server for MCP over HTTP/SSE
 */
if (TRANSPORT === "http" || TRANSPORT === "sse") {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || "/", `http://${req.headers.host}`);

    // Health check endpoint
    if (url.pathname === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        status: "healthy",
        service: "mini-xdr-mcp-server",
        transport: TRANSPORT,
        api_base: API_BASE,
        timestamp: new Date().toISOString()
      }));
      return;
    }

    // MCP endpoint
    if (url.pathname === "/mcp") {
      if (TRANSPORT === "sse") {
        // SSE transport for streaming
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        });

        const transport = new SSEServerTransport("/mcp", res);
        const mcpServer = createMCPServer();
        await mcpServer.connect(transport);

        req.on("close", () => {
          console.error("Client disconnected from SSE");
        });
      } else {
        // HTTP POST for request/response
        let body = "";
        req.on("data", (chunk) => {
          body += chunk.toString();
        });

        req.on("end", async () => {
          try {
            const request = JSON.parse(body);
            // Handle MCP request
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
              message: "HTTP MCP handler not yet implemented",
              request: request
            }));
          } catch (error) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "Invalid JSON" }));
          }
        });
      }
      return;
    }

    // 404 for other paths
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not found" }));
  });

  server.listen(PORT, () => {
    console.error(`✅ Mini-XDR MCP Server listening on port ${PORT}`);
    console.error(`   Health check: http://localhost:${PORT}/health`);
    console.error(`   MCP endpoint: http://localhost:${PORT}/mcp`);
  });

} else {
  // Stdio mode (default for Claude Code)
  console.error("Starting in stdio mode...");
  // Re-execute the original mcp_server.ts for stdio
  const { spawn } = require("child_process");
  const path = require("path");

  const child = spawn("ts-node", [path.join(__dirname, "mcp_server.ts")], {
    stdio: "inherit",
    env: { ...process.env, API_BASE, API_KEY }
  });

  child.on("error", (error) => {
    console.error("Failed to start MCP server:", error);
    process.exit(1);
  });

  child.on("exit", (code) => {
    process.exit(code || 0);
  });
}

/**
 * Create MCP Server instance (for HTTP/SSE modes)
 */
function createMCPServer() {
  const server = new Server(
    {
      name: "mini-xdr",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // TODO: Setup tool handlers here
  // For now, return basic server

  return server;
}
