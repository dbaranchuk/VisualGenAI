// Zero-dependency Node server for the VisualGenAI exam booking app.
// Run: `node server.js` (env: PORT, ADMIN_TOKEN, BOOKINGS_FILE)

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const DATA_FILE = process.env.BOOKINGS_FILE
  ? path.resolve(process.env.BOOKINGS_FILE)
  : path.join(ROOT, 'bookings.json');
const ADMIN_TOKEN = process.env.ADMIN_TOKEN || 'visualgenai-admin';

// Exam: 2026-05-18, 18:00 - 22:00 = 240 min = 12 atomic 20-min cells.
// "defend" booking takes 1 cell (20 min). "upgrade" booking takes 2 cells (40 min).
// Three parallel queues; each (queue, cell) pair holds at most one student.
const EXAM_DATE = '2026-05-18';
const EXAM_START_HOUR = 18;
const EXAM_END_HOUR = 22;
const BASE_SLOT_MIN = 20;
const TOTAL_BASE_SLOTS = ((EXAM_END_HOUR - EXAM_START_HOUR) * 60) / BASE_SLOT_MIN; // 12
const QUEUES_COUNT = 3;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon'
};

function fmtTime(totalMin) {
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

function buildBaseSlots() {
  const slots = [];
  for (let i = 0; i < TOTAL_BASE_SLOTS; i++) {
    const startMin = EXAM_START_HOUR * 60 + i * BASE_SLOT_MIN;
    slots.push({
      index: i,
      start: fmtTime(startMin),
      end: fmtTime(startMin + BASE_SLOT_MIN)
    });
  }
  return slots;
}

function durationFor(type) {
  return type === 'upgrade' ? 2 : 1;
}

function defaultQueues() {
  return Array.from({ length: QUEUES_COUNT }, () => ({ interviewer: '', zoom: '' }));
}

function normalizeQueues(input) {
  const out = defaultQueues();
  if (!Array.isArray(input)) return out;
  for (let i = 0; i < QUEUES_COUNT; i++) {
    const q = input[i] || {};
    out[i] = {
      interviewer: typeof q.interviewer === 'string' ? q.interviewer.trim().slice(0, 120) : '',
      zoom: typeof q.zoom === 'string' ? q.zoom.trim().slice(0, 500) : ''
    };
  }
  return out;
}

function load() {
  try {
    const raw = fs.readFileSync(DATA_FILE, 'utf8');
    const data = JSON.parse(raw);
    if (Array.isArray(data)) {
      return { bookings: data, queues: defaultQueues() };
    }
    return {
      bookings: Array.isArray(data.bookings) ? data.bookings : [],
      queues: normalizeQueues(data.queues)
    };
  } catch {
    return { bookings: [], queues: defaultQueues() };
  }
}

function save(data) {
  const tmp = DATA_FILE + '.tmp';
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
  fs.renameSync(tmp, DATA_FILE);
}

// queueGrid[queue][cell] = booking holding that cell, or null
function queueGrid(bookings) {
  const grid = Array.from({ length: QUEUES_COUNT }, () =>
    new Array(TOTAL_BASE_SLOTS).fill(null)
  );
  for (const b of bookings) {
    if (b.queue == null) continue;
    if (b.queue < 0 || b.queue >= QUEUES_COUNT) continue;
    for (let i = 0; i < b.duration; i++) {
      const idx = b.start + i;
      if (idx >= 0 && idx < TOTAL_BASE_SLOTS) grid[b.queue][idx] = b;
    }
  }
  return grid;
}

function publicBooking(b) {
  return {
    name: b.name,
    type: b.type,
    queue: b.queue,
    start: b.start,
    duration: b.duration
  };
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1e5) {
        reject(new Error('Body too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function send(res, status, payload, headers = {}) {
  const isString = typeof payload === 'string';
  const body = isString ? payload : JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': isString ? 'text/plain; charset=utf-8' : MIME['.json'],
    'Cache-Control': 'no-store',
    ...headers
  });
  res.end(body);
}

function sanitizeName(name) {
  if (typeof name !== 'string') return '';
  return name.replace(/[\x00-\x1f\x7f]/g, '').replace(/\s+/g, ' ').trim().slice(0, 60);
}

function requireAdmin(req, parsed) {
  const header = req.headers['x-admin-token'];
  const query = parsed.query && parsed.query.token;
  return header === ADMIN_TOKEN || query === ADMIN_TOKEN;
}

function handleApi(req, res, parsed) {
  const { pathname } = parsed;

  if (req.method === 'GET' && pathname === '/api/state') {
    const data = load();
    return send(res, 200, {
      examDate: EXAM_DATE,
      baseSlots: buildBaseSlots(),
      queues: data.queues,
      bookings: data.bookings.map(publicBooking)
    });
  }

  if (req.method === 'POST' && pathname === '/api/book') {
    return readBody(req)
      .then((raw) => {
        let body;
        try {
          body = JSON.parse(raw || '{}');
        } catch {
          return send(res, 400, { error: 'Invalid JSON' });
        }
        const name = sanitizeName(body.name);
        const type = body.type;
        let pick;
        if (Number.isFinite(body.pick)) pick = Number(body.pick);

        if (!name) return send(res, 400, { error: 'Please enter your name.' });
        if (type !== 'defend' && type !== 'upgrade') {
          return send(res, 400, { error: 'Invalid slot type.' });
        }
        if (!Number.isInteger(pick)) {
          return send(res, 400, { error: 'Pick a timeslot.' });
        }

        const duration = durationFor(type);
        if (pick < 0 || pick + duration > TOTAL_BASE_SLOTS) {
          return send(res, 400, { error: 'Invalid timeslot in selection.' });
        }

        const data = load();

        // 1 pick per student (case-insensitive name match)
        const existing = data.bookings.find(
          (b) => b.name.toLowerCase() === name.toLowerCase()
        );
        if (existing) {
          const interviewer = data.queues[existing.queue] && data.queues[existing.queue].interviewer;
          const who = interviewer ? ` with ${interviewer}` : '';
          return send(res, 400, {
            error: `${name} already booked queue ${existing.queue + 1}${who} at ${fmtTime((EXAM_START_HOUR * 60) + existing.start * BASE_SLOT_MIN)}. Each student books once.`
          });
        }

        // Pick a random queue that has the requested cell range free.
        const grid = queueGrid(data.bookings);
        const freeQueues = [];
        for (let q = 0; q < QUEUES_COUNT; q++) {
          let free = true;
          for (let i = 0; i < duration; i++) {
            if (grid[q][pick + i]) { free = false; break; }
          }
          if (free) freeQueues.push(q);
        }
        if (freeQueues.length === 0) {
          return send(res, 409, {
            error: `${fmtTime((EXAM_START_HOUR * 60) + pick * BASE_SLOT_MIN)} is full across all queues — pick another time.`
          });
        }
        const queue = freeQueues[Math.floor(Math.random() * freeQueues.length)];

        const now = new Date().toISOString();
        const booking = {
          id: Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8),
          name,
          type,
          queue,
          start: pick,
          duration,
          created_at: now
        };
        data.bookings.push(booking);
        save(data);

        return send(res, 200, {
          ok: true,
          booking: publicBooking(booking)
        });
      })
      .catch(() => send(res, 400, { error: 'Bad request' }));
  }

  if (req.method === 'POST' && pathname === '/api/admin/queues') {
    if (!requireAdmin(req, parsed)) return send(res, 403, { error: 'Forbidden' });
    return readBody(req)
      .then((raw) => {
        let body;
        try {
          body = JSON.parse(raw || '{}');
        } catch {
          return send(res, 400, { error: 'Invalid JSON' });
        }
        const queues = normalizeQueues(body.queues);
        const data = load();
        data.queues = queues;
        save(data);
        return send(res, 200, { ok: true, queues });
      })
      .catch(() => send(res, 400, { error: 'Bad request' }));
  }

  if (req.method === 'DELETE' && pathname.startsWith('/api/bookings/')) {
    if (!requireAdmin(req, parsed)) return send(res, 403, { error: 'Forbidden' });
    const id = pathname.split('/').pop();
    const data = load();
    const next = data.bookings.filter((b) => b.id !== id);
    if (next.length === data.bookings.length) return send(res, 404, { error: 'Not found' });
    data.bookings = next;
    save(data);
    return send(res, 200, { ok: true });
  }

  // Move a booking to a different queue (admin only).
  // Target queue must be free across all of the booking's cells.
  if (req.method === 'PATCH' && pathname.startsWith('/api/admin/bookings/')) {
    if (!requireAdmin(req, parsed)) return send(res, 403, { error: 'Forbidden' });
    const id = pathname.split('/').pop();
    return readBody(req)
      .then((raw) => {
        let body;
        try {
          body = JSON.parse(raw || '{}');
        } catch {
          return send(res, 400, { error: 'Invalid JSON' });
        }
        const newQueue = Number(body.queue);
        if (!Number.isInteger(newQueue) || newQueue < 0 || newQueue >= QUEUES_COUNT) {
          return send(res, 400, { error: 'Invalid queue.' });
        }
        const data = load();
        const booking = data.bookings.find((b) => b.id === id);
        if (!booking) return send(res, 404, { error: 'Not found' });
        if (booking.queue === newQueue) {
          return send(res, 200, { ok: true, booking: publicBooking(booking) });
        }
        const others = data.bookings.filter((b) => b.id !== id);
        const grid = queueGrid(others);
        for (let i = 0; i < booking.duration; i++) {
          if (grid[newQueue][booking.start + i]) {
            return send(res, 409, {
              error: `Queue ${newQueue + 1} is already taken at ${fmtTime((EXAM_START_HOUR * 60) + (booking.start + i) * BASE_SLOT_MIN)}.`
            });
          }
        }
        booking.queue = newQueue;
        save(data);
        return send(res, 200, { ok: true, booking: publicBooking(booking) });
      })
      .catch(() => send(res, 400, { error: 'Bad request' }));
  }

  if (req.method === 'GET' && pathname === '/api/admin/bookings') {
    if (!requireAdmin(req, parsed)) return send(res, 403, { error: 'Forbidden' });
    const data = load();
    return send(res, 200, { bookings: data.bookings, queues: data.queues });
  }

  return send(res, 404, { error: 'Not found' });
}

function safeStaticPath(pathname) {
  const cleaned = pathname === '/' ? '/index.html' : pathname;
  let decoded;
  try {
    decoded = decodeURIComponent(cleaned);
  } catch {
    return null;
  }
  const target = path.normalize(path.join(PUBLIC_DIR, decoded));
  if (!target.startsWith(PUBLIC_DIR)) return null;
  return target;
}

function serveStatic(req, res, parsed) {
  const target = safeStaticPath(parsed.pathname);
  if (!target) return send(res, 400, 'Bad path');
  fs.stat(target, (err, stat) => {
    if (err || !stat.isFile()) return send(res, 404, 'Not found');
    const ext = path.extname(target).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache'
    });
    fs.createReadStream(target).pipe(res);
  });
}

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  if (parsed.pathname.startsWith('/api/')) {
    return handleApi(req, res, parsed);
  }
  if (req.method !== 'GET') return send(res, 405, 'Method not allowed');
  serveStatic(req, res, parsed);
});

const PORT = Number(process.env.PORT) || 3000;
server.listen(PORT, () => {
  console.log(`VisualGenAI exam booking on http://localhost:${PORT}`);
  console.log(`Admin view:  http://localhost:${PORT}/admin.html?token=${ADMIN_TOKEN}`);
});
