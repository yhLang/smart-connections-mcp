/**
 * Data loading module for Smart Connections embeddings.
 *
 * Reads pre-computed embeddings from Smart Connections plugin's
 * .smart-env directory. Does NOT compute new embeddings.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Config, log } from './security.js';

export interface EmbeddingEntry {
  path: string;
  embedding: number[];
  blocks?: Record<string, BlockInfo>;
}

export interface BlockInfo {
  hash?: string;
  size?: number;
  lines?: [number, number];
}

export interface ModelInfo {
  modelKey: string;
  dimensions: number;
  adapter: string;
}

export interface SmartConnectionsData {
  entries: Map<string, EmbeddingEntry>;
  modelInfo: ModelInfo;
}

/**
 * Detect the actual active embedding model by counting which model key
 * has the most entries across all .ajson files. This is more reliable than
 * trusting smart_env.json.model_key, which Smart Connections plugin can
 * overwrite when saving other settings (e.g. folder exclusions).
 *
 * Patch added 2026-05-15: plugin overwrites model_key on every settings save,
 * so we detect from .ajson directly to ensure we always use the latest reindex.
 */
function detectActiveModelKey(smartEnvPath: string): string | null {
  const multiPath = path.join(smartEnvPath, 'multi');
  if (!fs.existsSync(multiPath)) return null;

  const files = fs.readdirSync(multiPath).filter(f => f.endsWith('.ajson'));
  const keyCounts: Record<string, number> = {};

  for (const file of files) {
    const filePath = path.join(multiPath, file);
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      // Scan for "modelKey":{"vec" patterns without full JSON parse (faster)
      const matches = content.match(/"([^"]{5,80})":\{"vec"/g) || [];
      for (const match of matches) {
        const key = match.replace(/^"|":\{"vec"$/, '').replace(/^"/, '');
        keyCounts[key] = (keyCounts[key] || 0) + 1;
      }
    } catch { /* skip unparseable files */ }
  }

  if (Object.keys(keyCounts).length === 0) return null;

  // Pick the model key with the most entries (= most recently reindexed)
  return Object.entries(keyCounts).sort(([, a], [, b]) => b - a)[0][0];
}

/**
 * Load Smart Connections data from the vault's .smart-env directory.
 */
export function loadSmartConnectionsData(config: Config): SmartConnectionsData {
  const smartEnvPath = path.join(config.resolvedVaultPath, '.smart-env');

  // Load model info from smart_env.json (base config)
  let modelInfo = loadModelInfo(smartEnvPath);

  // Detect actual active model from .ajson entry counts —
  // overrides smart_env.json when plugin has overwritten it with a stale value.
  const detectedKey = detectActiveModelKey(smartEnvPath);
  if (detectedKey && detectedKey !== modelInfo.modelKey) {
    log('INFO', 'model_key_auto_detected', {
      fromConfig: modelInfo.modelKey,
      fromAjson: detectedKey,
      reason: '.ajson has more entries for detected key — using it',
    });
    const dimensions = getModelDimensions(detectedKey);
    modelInfo = { modelKey: detectedKey, dimensions, adapter: 'transformers' };
  }

  log('INFO', 'model_loaded', { modelKey: modelInfo.modelKey, dimensions: modelInfo.dimensions });

  // Load embeddings from multi/*.ajson files
  const entries = loadEmbeddings(smartEnvPath, modelInfo.modelKey);
  log('INFO', 'embeddings_loaded', { count: entries.size });

  return { entries, modelInfo };
}

/**
 * Load model configuration from smart_env.json
 */
function loadModelInfo(smartEnvPath: string): ModelInfo {
  const configPath = path.join(smartEnvPath, 'smart_env.json');

  if (!fs.existsSync(configPath)) {
    throw new Error(`Smart env config not found: ${configPath}`);
  }

  let config: unknown;
  try {
    const content = fs.readFileSync(configPath, 'utf-8');
    config = JSON.parse(content);
  } catch (e) {
    throw new Error(`Failed to parse smart_env.json: ${e}`);
  }

  // Navigate to model key - handle nested structure
  const modelKey = extractModelKey(config);
  if (!modelKey) {
    throw new Error('Could not determine embedding model from smart_env.json');
  }

  // Determine dimensions based on known models
  const dimensions = getModelDimensions(modelKey);

  return {
    modelKey,
    dimensions,
    adapter: 'transformers',
  };
}

/**
 * Extract model key from smart_env.json config structure.
 */
function extractModelKey(config: unknown): string | null {
  if (!config || typeof config !== 'object') return null;

  const c = config as Record<string, unknown>;

  // Try smart_sources.embed_model.transformers.model_key
  const smartSources = c.smart_sources as Record<string, unknown> | undefined;
  if (smartSources?.embed_model) {
    const embedModel = smartSources.embed_model as Record<string, unknown>;
    if (embedModel.transformers) {
      const transformers = embedModel.transformers as Record<string, unknown>;
      if (typeof transformers.model_key === 'string') {
        return transformers.model_key;
      }
    }
  }

  return null;
}

/**
 * Get embedding dimensions for known models.
 */
function getModelDimensions(modelKey: string): number {
  const knownModels: Record<string, number> = {
    'TaylorAI/bge-micro-v2': 384,
    'sentence-transformers/all-MiniLM-L6-v2': 384,
    'BAAI/bge-small-en-v1.5': 384,
    'BAAI/bge-base-en-v1.5': 768,
    'BAAI/bge-large-en-v1.5': 1024,
    // Multilingual / Chinese (added 2026-05-14 for cross-lingual vault)
    'BAAI/bge-m3': 1024,
    'intfloat/multilingual-e5-small': 384,
    'intfloat/multilingual-e5-base': 768,
    'intfloat/multilingual-e5-large': 1024,
    // Xenova ONNX forks (transformers.js 用的, Smart Connections plugin 实际 load 这些)
    'Xenova/multilingual-e5-small': 384,
    'Xenova/multilingual-e5-base': 768,
    'Xenova/bge-small-en-v1.5': 384,
    'Xenova/bge-base-en-v1.5': 768,
  };

  const dimensions = knownModels[modelKey];
  if (dimensions) {
    return dimensions;
  }

  // Default to 384 (most common for small models)
  log('WARN', 'unknown_model_dimensions', { modelKey, defaulting: 384 });
  return 384;
}

/**
 * Load embeddings from .ajson files in the multi/ directory.
 */
function loadEmbeddings(smartEnvPath: string, modelKey: string): Map<string, EmbeddingEntry> {
  const multiPath = path.join(smartEnvPath, 'multi');
  const entries = new Map<string, EmbeddingEntry>();

  if (!fs.existsSync(multiPath)) {
    log('WARN', 'no_multi_directory', { path: multiPath });
    return entries;
  }

  const files = fs.readdirSync(multiPath).filter(f => f.endsWith('.ajson'));

  for (const file of files) {
    const filePath = path.join(multiPath, file);
    try {
      const fileEntries = parseAjsonFile(filePath, modelKey);
      for (const [key, entry] of fileEntries) {
        entries.set(key, entry);
      }
    } catch (e) {
      log('WARN', 'ajson_parse_error', { file, error: String(e) });
      // Continue with other files
    }
  }

  return entries;
}

/**
 * Parse a single .ajson file.
 *
 * Smart Connections uses an "append JSON" format where each file contains
 * entries like: "key": {value}, one per line. We wrap in braces and parse
 * as a single JSON object.
 */
function parseAjsonFile(filePath: string, modelKey: string): Map<string, EmbeddingEntry> {
  const entries = new Map<string, EmbeddingEntry>();
  const content = fs.readFileSync(filePath, 'utf-8').trim();

  if (!content) return entries;

  // The file content is a series of "key": {...}, entries
  // Wrap in braces to make it valid JSON, removing trailing comma
  let jsonContent = content;

  // Remove trailing comma if present (before we wrap)
  if (jsonContent.endsWith(',')) {
    jsonContent = jsonContent.slice(0, -1);
  }

  // Wrap in braces to create valid JSON object
  const wrappedContent = '{' + jsonContent + '}';

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(wrappedContent);
  } catch (e) {
    // If parsing fails, log and return empty
    log('WARN', 'ajson_json_parse_failed', {
      file: path.basename(filePath),
      error: String(e).slice(0, 100)
    });
    return entries;
  }

  // Iterate over all entries in the parsed object
  for (const [fullKey, value] of Object.entries(parsed)) {
    // Only process smart_sources entries (file-level), skip smart_blocks
    if (!fullKey.startsWith('smart_sources:')) continue;

    const notePath = fullKey.replace(/^smart_sources:/, '');

    // Skip if it's actually a block (has # in path)
    if (notePath.includes('#')) continue;

    const data = value as Record<string, unknown>;

    // Extract embedding vector
    const embeddings = data.embeddings as Record<string, { vec?: number[] }> | undefined;
    if (!embeddings) continue;

    const modelData = embeddings[modelKey];
    if (!modelData?.vec || !Array.isArray(modelData.vec)) continue;

    // Extract block info if present
    const blocks = data.blocks as Record<string, BlockInfo> | undefined;

    entries.set(notePath, {
      path: notePath,
      embedding: modelData.vec,
      blocks,
    });
  }

  return entries;
}

/**
 * Get the note path relative to vault root, suitable for display.
 */
export function normalizeNotePath(notePath: string): string {
  // Remove leading slashes if present
  return notePath.replace(/^\/+/, '');
}

/**
 * Extract a title from a note path (filename without extension).
 */
export function extractTitle(notePath: string): string {
  const filename = path.basename(notePath, '.md');
  // Replace underscores with spaces for readability
  return filename.replace(/_/g, ' ');
}
