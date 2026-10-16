import express, { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
const FLASK_PORT = 5005;
const FLASK_URL = `http://127.0.0.1:${FLASK_PORT}`;

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Ensure Python ML Flask backend is running on FLASK_PORT
let pythonProcess: any = null;

function startFlaskBackend() {
  console.log(`[SentimentIQ Server] Starting Python ML Flask service on port ${FLASK_PORT}...`);
  pythonProcess = spawn('python3', ['app.py'], {
    cwd: __dirname,
    env: { ...process.env, FLASK_PORT: String(FLASK_PORT) },
    stdio: 'inherit',
  });

  pythonProcess.on('error', (err: any) => {
    console.error('[SentimentIQ Server] Python process error:', err);
  });

  pythonProcess.on('exit', (code: number, signal: string) => {
    console.log(`[SentimentIQ Server] Python process exited with code ${code}, signal ${signal}. Scheduling restart...`);
    pythonProcess = null;
    setTimeout(() => {
      startFlaskBackend();
    }, 1500);
  });
}

// Check if Flask is already responding; if not, spawn it
async function checkOrStartFlask() {
  try {
    const res = await fetch(`${FLASK_URL}/api/health`, { signal: AbortSignal.timeout(1500) });
    if (res.ok) {
      console.log(`[SentimentIQ Server] Flask ML service already active on ${FLASK_URL}`);
      return;
    }
  } catch (e) {
    // Not running yet
  }
  startFlaskBackend();
}

checkOrStartFlask();

async function forwardRequest(req: Request, targetUrl: string) {
  const headers: Record<string, string> = {};
  if (req.headers['content-type']) {
    headers['content-type'] = req.headers['content-type'] as string;
  }

  const fetchOptions: RequestInit = {
    method: req.method,
    headers,
    signal: AbortSignal.timeout(10000),
  };

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    fetchOptions.body = JSON.stringify(req.body);
  }

  return await fetch(targetUrl, fetchOptions);
}

// Proxy /api/* to Flask backend with automatic reconnect retry
app.all('/api/*', async (req: Request, res: Response) => {
  const targetUrl = `${FLASK_URL}${req.originalUrl}`;
  try {
    let response: any;
    try {
      response = await forwardRequest(req, targetUrl);
    } catch (initialErr) {
      // If disconnected, ensure Flask process is spawned and retry once
      console.warn(`[Proxy Reconnect] Retrying ${targetUrl}...`);
      if (!pythonProcess) {
        startFlaskBackend();
      }
      await new Promise((resolve) => setTimeout(resolve, 1500));
      response = await forwardRequest(req, targetUrl);
    }

    const contentType = response.headers.get('content-type') || '';
    res.status(response.status);

    if (contentType.includes('application/json')) {
      const data = await response.json();
      return res.json(data);
    } else {
      const blob = await response.arrayBuffer();
      res.setHeader('content-type', contentType);
      const disposition = response.headers.get('content-disposition');
      if (disposition) {
        res.setHeader('content-disposition', disposition);
      }
      return res.send(Buffer.from(blob));
    }
  } catch (error: any) {
    console.error(`[Proxy Error] Failed to proxy to ${targetUrl}:`, error.message);
    return res.status(503).json({
      error: 'SentimentIQ ML Engine is initializing or temporarily unavailable.',
      details: error.message,
    });
  }
});

// Setup Vite in development or serve static in production
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    if (fs.existsSync(distPath)) {
      app.use(express.static(distPath));
      app.get('*', (_req, res) => {
        res.sendFile(path.resolve(distPath, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SentimentIQ Server] Running at http://localhost:${PORT}`);
  });
}

startServer();
