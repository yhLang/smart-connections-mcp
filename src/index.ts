#!/usr/bin/env node
/**
 * Smart Connections MCP Server
 *
 * A security-hardened MCP server for semantic search of Obsidian vaults.
 * Uses Smart Connections embeddings for indexed notes, and can compute
 * embeddings locally for freeform text queries.
 *
 * Security principles:
 * - Read-only: No write operations
 * - Path confined: All access validated against vault root
 * - Fail closed: Errors deny access
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { validateConfig, log, Config } from './security.js';
import { loadSmartConnectionsData } from './data.js';
import { toolDefinitions, handleToolCall, ToolContext } from './tools.js';
import { createEmbedder, Embedder } from './embeddings.js';

const VERSION = '0.2.0';

/**
 * Main entry point.
 */
async function main(): Promise<void> {
  // Parse command line arguments
  const vaultPath = getVaultPath();

  // SECURITY: Validate configuration at startup (fail fast)
  let config: Config;
  try {
    config = validateConfig(vaultPath);
    log('INFO', 'config_validated', { vaultPath: config.vaultPath });
  } catch (e) {
    log('ERROR', 'config_validation_failed', { error: String(e) });
    process.exit(1);
  }

  // PATCH (2026-05-14): Lazy data loading.
  // Originally loadSmartConnectionsData + createEmbedder ran sync before server.connect,
  // causing 3-minute startup that exceeded Claude Desktop / Claude Code attach timeouts.
  // Now: server.connect happens immediately, data loads in background.
  // Tool calls await dataReady (with 30s timeout per call) — first call after spawn may
  // wait up to 30s, subsequent calls are fast (data cached).
  let ctx: ToolContext | undefined;
  let dataLoadError: string | undefined;
  const dataReady: Promise<void> = (async () => {
    try {
      const data = loadSmartConnectionsData(config);
      let embedder: Embedder | undefined;
      try {
        embedder = await createEmbedder(
          data.modelInfo.modelKey,
          data.modelInfo.dimensions
        );
      } catch (e) {
        log('WARN', 'embedder_init_failed', { error: String(e) });
        log('WARN', 'text_search_disabled', { reason: 'embedder initialization failed' });
      }
      ctx = { config, data, embedder };
      log('INFO', 'data_ready', {
        indexedNotes: data.entries.size,
        modelKey: data.modelInfo.modelKey,
        textSearchEnabled: embedder?.isReady() ?? false,
      });
    } catch (e) {
      dataLoadError = String(e);
      log('ERROR', 'data_load_failed', { error: dataLoadError });
    }
  })();

  // Create MCP server
  const server = new Server(
    {
      name: 'smart-connections-mcp',
      version: VERSION,
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register tool listing handler — always available (returns static tool defs)
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: toolDefinitions,
    };
  });

  // Register tool call handler — wait for data if not ready (with timeout)
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (!ctx) {
      log('INFO', 'tool_call_waiting_for_data', { tool: name });
      const TIMEOUT_MS = 30000;
      const waitResult = await Promise.race([
        dataReady.then(() => ({ ready: true as const })),
        new Promise<{ ready: false }>((resolve) =>
          setTimeout(() => resolve({ ready: false }), TIMEOUT_MS)
        ),
      ]);

      if (!waitResult.ready) {
        return {
          content: [
            {
              type: 'text',
              text: 'Smart Connections data still loading (large vault). Try again in ~1-2 minutes.',
            },
          ],
          isError: true,
        };
      }

      if (dataLoadError) {
        return {
          content: [
            {
              type: 'text',
              text: `Smart Connections data load failed: ${dataLoadError}`,
            },
          ],
          isError: true,
        };
      }

      if (!ctx) {
        return {
          content: [
            { type: 'text', text: 'Server data not ready' },
          ],
          isError: true,
        };
      }
    }

    const result = await handleToolCall(name, args, ctx);

    return result;
  });

  // Start server with stdio transport — listen IMMEDIATELY (no await on data load)
  const transport = new StdioServerTransport();

  log('INFO', 'server_starting', {
    version: VERSION,
    vaultPath: config.vaultPath,
    note: 'data loading in background; first tool call may wait up to 30s',
  });

  await server.connect(transport);

  log('INFO', 'server_connected');

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    log('INFO', 'server_shutdown', { reason: 'SIGINT' });
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    log('INFO', 'server_shutdown', { reason: 'SIGTERM' });
    process.exit(0);
  });
}

/**
 * Get vault path from CLI args or environment variable.
 */
function getVaultPath(): string | undefined {
  // Check CLI args: --vault /path/to/vault
  const args = process.argv.slice(2);
  const vaultIdx = args.indexOf('--vault');
  if (vaultIdx !== -1 && args[vaultIdx + 1]) {
    return args[vaultIdx + 1];
  }

  // Fall back to environment variable
  return process.env.VAULT_PATH;
}

// Run the server
main().catch((e) => {
  log('ERROR', 'server_fatal', { error: String(e) });
  process.exit(1);
});
