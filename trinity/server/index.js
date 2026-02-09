import express from 'express';
import cors from 'cors';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';
import dotenv from 'dotenv';

import { initializeDatabase, seedDatabase } from './db.js';
import chatRoutes from './routes/chat.js';
import memoriesRoutes from './routes/memories.js';
import directivesRoutes from './routes/directives.js';
import skillsRoutes from './routes/skills.js';
import stateRoutes from './routes/state.js';
import sessionsRoutes from './routes/sessions.js';
import pendingRoutes from './routes/pending.js';
import settingsRoutes from './routes/settings.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from project root
dotenv.config({ path: join(__dirname, '..', '.env') });

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Initialize database
initializeDatabase();
seedDatabase();

// API Routes
app.use('/api/chat', chatRoutes);
app.use('/api/memories', memoriesRoutes);
app.use('/api/directives', directivesRoutes);
app.use('/api/skills', skillsRoutes);
app.use('/api/state', stateRoutes);
app.use('/api/sessions', sessionsRoutes);
app.use('/api/pending', pendingRoutes);
app.use('/api/settings', settingsRoutes);

// Serve static frontend in production
const distPath = join(__dirname, '..', 'dist');
if (existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get('*', (req, res) => {
    res.sendFile(join(distPath, 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`Trinity v0.1 running on http://localhost:${PORT}`);
});
