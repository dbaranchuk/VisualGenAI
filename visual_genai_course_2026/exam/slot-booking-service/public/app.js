const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const QUEUES_COUNT = 3;

const state = {
  name: '',
  type: 'defend',           // 'defend' = 1 cell (20m), 'upgrade' = 2 cells (40m)
  pick: null,               // start cell index or null
  baseSlots: [],
  queues: [],               // [{interviewer, zoom}, ...]
  bookings: [],
};

const els = {
  name: $('#name'),
  typeCards: $$('.type-card'),
  queueInfo: $('#queue-info'),
  timeGrid: $('#time-grid'),
  schedule: $('#schedule'),
  submit: $('#submit'),
  status: $('#status'),
  picksCounter: $('#picks-counter'),
  refresh: $('#refresh'),
};

function durationCells() {
  return state.type === 'upgrade' ? 2 : 1;
}

function startStep() {
  return state.type === 'upgrade' ? 2 : 1;
}

function rangeLabel(startIdx, dur) {
  const a = state.baseSlots[startIdx];
  const b = state.baseSlots[startIdx + dur - 1];
  return a && b ? `${a.start} – ${b.end}` : '';
}

// grid[queue][cell] = booking or null
function buildGrid() {
  const grid = Array.from({ length: QUEUES_COUNT }, () =>
    new Array(state.baseSlots.length).fill(null)
  );
  for (const b of state.bookings) {
    if (b.queue == null || b.queue < 0 || b.queue >= QUEUES_COUNT) continue;
    for (let i = 0; i < b.duration; i++) {
      const idx = b.start + i;
      if (idx >= 0 && idx < state.baseSlots.length) grid[b.queue][idx] = b;
    }
  }
  return grid;
}

// How many queues are free across the cell range [start, start+dur)
function freeQueuesAt(startIdx, grid) {
  const dur = durationCells();
  let n = 0;
  for (let q = 0; q < QUEUES_COUNT; q++) {
    let free = true;
    for (let i = 0; i < dur; i++) {
      if (grid[q][startIdx + i]) { free = false; break; }
    }
    if (free) n++;
  }
  return n;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[c]));
}

function safeUrl(raw) {
  if (!raw) return '';
  const s = String(raw).trim();
  if (!s) return '';
  if (/^https?:\/\//i.test(s)) return s;
  return 'https://' + s.replace(/^\/+/, '');
}

function renderQueueInfo() {
  els.queueInfo.innerHTML = '';
  for (let qi = 0; qi < QUEUES_COUNT; qi++) {
    const q = state.queues[qi] || { interviewer: '', zoom: '' };
    const nameHtml = q.interviewer
      ? `<span class="queue-name">${escapeHtml(q.interviewer)}</span>`
      : `<span class="queue-name tbd">Interviewer · TBD</span>`;
    const zoomUrl = safeUrl(q.zoom);
    const zoomHtml = zoomUrl
      ? `<a class="queue-zoom" href="${escapeHtml(zoomUrl)}" target="_blank" rel="noopener">
           <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 10l5-3v10l-5-3"/><rect x="2" y="6" width="13" height="12" rx="2"/></svg>
           Zoom link
         </a>`
      : `<span class="queue-zoom muted">Zoom link · TBD</span>`;
    const card = document.createElement('div');
    card.className = 'queue-card';
    card.innerHTML = `
      <span class="queue-num">Queue ${qi + 1}</span>
      ${nameHtml}
      ${zoomHtml}
    `;
    els.queueInfo.appendChild(card);
  }
}

function renderTimeGrid() {
  const dur = durationCells();
  const step = startStep();
  const grid = buildGrid();
  const total = state.baseSlots.length;

  els.timeGrid.innerHTML = '';

  for (let s = 0; s + dur <= total; s += step) {
    const isPicked = state.pick === s;
    const freeQ = freeQueuesAt(s, grid);
    const taken = freeQ === 0;
    const st = isPicked ? 'picked' : (taken ? 'taken' : 'free');

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'slot';
    btn.dataset.state = st;
    btn.dataset.start = String(s);
    if (st === 'taken') btn.disabled = true;

    const time = document.createElement('div');
    time.className = 'slot-time';
    time.textContent = rangeLabel(s, dur);

    const meta = document.createElement('div');
    meta.className = 'slot-meta';
    if (isPicked) meta.textContent = 'Selected';
    else if (taken) meta.textContent = 'Full';
    else meta.textContent = `${freeQ} of ${QUEUES_COUNT} queues free`;

    btn.appendChild(time);
    btn.appendChild(meta);
    btn.addEventListener('click', () => onSlotClick(s));
    els.timeGrid.appendChild(btn);
  }

  els.picksCounter.textContent = state.pick === null
    ? 'Nothing selected'
    : `Selected ${rangeLabel(state.pick, dur)}`;
  updateSubmit();
}

function onSlotClick(start) {
  state.pick = state.pick === start ? null : start;
  renderTimeGrid();
}

function setType(newType) {
  if (newType === state.type) return;
  state.type = newType;
  state.pick = null;
  els.typeCards.forEach(c => c.setAttribute('aria-checked', String(c.dataset.type === newType)));
  renderTimeGrid();
}

function updateSubmit() {
  const ok = state.name.trim().length > 0 && state.pick !== null;
  els.submit.disabled = !ok;
}

function setStatus(msg, kind) {
  els.status.innerHTML = msg || '';
  els.status.className = 'status' + (kind ? ' ' + kind : '');
}

function renderSchedule() {
  els.schedule.innerHTML = '';
  const grid = buildGrid();
  const total = state.baseSlots.length;

  for (let cell = 0; cell < total; cell++) {
    const rowBookings = [];
    for (let qi = 0; qi < QUEUES_COUNT; qi++) {
      const b = grid[qi][cell];
      if (b && b.start === cell && b.queue === qi) rowBookings.push({ qi, b });
    }
    let anyCovered = false;
    for (let qi = 0; qi < QUEUES_COUNT; qi++) {
      if (grid[qi][cell]) { anyCovered = true; break; }
    }

    if (rowBookings.length === 0 && !anyCovered) {
      const li = document.createElement('li');
      li.className = 'sched-row empty';
      li.innerHTML = `
        <span class="sched-time">${rangeLabel(cell, 1)}</span>
        <span class="sched-name">Open · all queues free</span>
        <span class="sched-tag">—</span>
      `;
      els.schedule.appendChild(li);
      continue;
    }

    for (const { qi, b } of rowBookings) {
      const tagClass = b.type === 'upgrade' ? 'upgrade' : 'defend';
      const tagLabel = b.type === 'upgrade' ? 'upgrade · 40m' : 'defend · 20m';
      const li = document.createElement('li');
      li.className = 'sched-row';
      li.innerHTML = `
        <span class="sched-time">${rangeLabel(cell, b.duration)}</span>
        <span class="sched-name"><span style="color:var(--text-mute);font-family:var(--font-mono);font-size:12px;margin-right:8px">Q${qi + 1}</span>${escapeHtml(b.name)}</span>
        <span class="sched-tag ${tagClass}">${tagLabel}</span>
      `;
      els.schedule.appendChild(li);
    }
  }
}

async function loadState() {
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    const data = await r.json();
    state.baseSlots = data.baseSlots || [];
    state.queues = data.queues || [];
    state.bookings = data.bookings || [];

    // Drop pick if it has become invalid
    if (state.pick !== null) {
      if (state.pick + durationCells() > state.baseSlots.length) {
        state.pick = null;
      } else {
        const grid = buildGrid();
        if (freeQueuesAt(state.pick, grid) === 0) state.pick = null;
      }
    }

    renderQueueInfo();
    renderTimeGrid();
    renderSchedule();
  } catch (e) {
    setStatus('Could not load schedule. Check your connection.', 'error');
  }
}

async function submit() {
  const name = els.name.value.trim();
  if (!name) {
    setStatus('Please enter your name.', 'error');
    els.name.focus();
    return;
  }
  if (state.pick === null) {
    setStatus('Pick a time.', 'error');
    return;
  }

  els.submit.disabled = true;
  setStatus('Booking…');

  try {
    const r = await fetch('/api/book', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        type: state.type,
        pick: state.pick,
      }),
    });
    const data = await r.json();
    if (!r.ok) {
      setStatus(data.error || 'Booking failed.', 'error');
      await loadState();
      return;
    }
    const b = data.booking;
    const q = state.queues[b.queue] || {};
    const who = q.interviewer ? escapeHtml(q.interviewer) : `Queue ${b.queue + 1}`;
    const zoomUrl = safeUrl(q.zoom);
    const zoomLink = zoomUrl
      ? ` · <a href="${escapeHtml(zoomUrl)}" target="_blank" rel="noopener" style="color:var(--accent-2);text-decoration:underline">Zoom</a>`
      : '';
    state.pick = null;
    setStatus(
      `Booked! ${escapeHtml(name)} — Queue ${b.queue + 1} with ${who} at ${rangeLabel(b.start, b.duration)}${zoomLink}.`,
      'success'
    );
    await loadState();
  } catch (e) {
    setStatus('Network error. Please try again.', 'error');
  } finally {
    updateSubmit();
  }
}

els.name.addEventListener('input', (e) => { state.name = e.target.value; updateSubmit(); });
els.typeCards.forEach(c => {
  c.addEventListener('click', () => setType(c.dataset.type));
  c.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setType(c.dataset.type); }
  });
});
els.submit.addEventListener('click', submit);
els.refresh.addEventListener('click', loadState);

loadState();
setInterval(loadState, 15000);
