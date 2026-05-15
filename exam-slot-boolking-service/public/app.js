const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const state = {
  name: '',
  type: 'defend',           // 'defend' = 1 cell (20m), 'upgrade' = 2 cells (40m)
  pick: null,               // { queue, start } or null
  baseSlots: [],
  queues: [],               // [{interviewer, zoom}, ...]
  bookings: [],
};

const QUEUES_COUNT = 3;

const els = {
  name: $('#name'),
  typeCards: $$('.type-card'),
  queueGrid: $('#queue-grid'),
  schedule: $('#schedule'),
  submit: $('#submit'),
  status: $('#status'),
  picksCounter: $('#picks-counter'),
  refresh: $('#refresh'),
};

function durationCells() {
  return state.type === 'upgrade' ? 2 : 1;
}

function rangeLabel(startIdx, dur) {
  const a = state.baseSlots[startIdx];
  const b = state.baseSlots[startIdx + dur - 1];
  return a && b ? `${a.start} – ${b.end}` : '';
}

// grid[queue][cell] = booking holding that cell, or null
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

// Aligned 40-min blocks start at even cells (0, 2, 4, 6, 8, 10)
function startStep() {
  return state.type === 'upgrade' ? 2 : 1;
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

function renderQueueHeader(qi) {
  const q = state.queues[qi] || { interviewer: '', zoom: '' };
  const nameHtml = q.interviewer
    ? `<span class="queue-name">${escapeHtml(q.interviewer)}</span>`
    : `<span class="queue-name tbd">Interviewer · TBD</span>`;
  const zoomUrl = safeUrl(q.zoom);
  const zoomHtml = zoomUrl
    ? `<a class="queue-zoom" href="${escapeHtml(zoomUrl)}" target="_blank" rel="noopener">
         <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 10l5-3v10l-5-3"/><rect x="2" y="6" width="13" height="12" rx="2"/></svg>
         Join Zoom
       </a>`
    : `<span class="queue-zoom muted">Zoom link · TBD</span>`;
  return `
    <div class="queue-head">
      <div class="queue-title">
        <span class="queue-num">Queue ${qi + 1}</span>
      </div>
      ${nameHtml}
      ${zoomHtml}
    </div>
  `;
}

function renderQueues() {
  const dur = durationCells();
  const step = startStep();
  const grid = buildGrid();
  const total = state.baseSlots.length;

  els.queueGrid.innerHTML = '';

  for (let qi = 0; qi < QUEUES_COUNT; qi++) {
    const col = document.createElement('div');
    col.className = 'queue-col';
    col.innerHTML = renderQueueHeader(qi);

    const cellsWrap = document.createElement('div');
    cellsWrap.className = 'queue-cells';

    for (let s = 0; s + dur <= total; s += step) {
      const isPicked = state.pick && state.pick.queue === qi && state.pick.start === s;
      let taken = false;
      for (let i = 0; i < dur; i++) {
        if (grid[qi][s + i]) { taken = true; break; }
      }

      const st = isPicked ? 'picked' : (taken ? 'taken' : 'free');
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'slot';
      btn.dataset.state = st;
      btn.dataset.queue = String(qi);
      btn.dataset.start = String(s);
      if (st === 'taken') btn.disabled = true;

      const time = document.createElement('div');
      time.className = 'slot-time';
      time.textContent = rangeLabel(s, dur);

      const meta = document.createElement('div');
      meta.className = 'slot-meta';
      if (isPicked) {
        meta.textContent = 'Selected';
      } else if (taken) {
        const occupant = grid[qi][s] || grid[qi][s + 1];
        meta.textContent = occupant ? `${occupant.name}` : 'Taken';
      } else {
        meta.textContent = `${dur * 20} min`;
      }

      btn.appendChild(time);
      btn.appendChild(meta);
      btn.addEventListener('click', () => onSlotClick(qi, s));
      cellsWrap.appendChild(btn);
    }

    col.appendChild(cellsWrap);
    els.queueGrid.appendChild(col);
  }

  if (state.pick) {
    const dur = durationCells();
    els.picksCounter.textContent = `Queue ${state.pick.queue + 1} · ${rangeLabel(state.pick.start, dur)}`;
  } else {
    els.picksCounter.textContent = 'Nothing selected';
  }
  updateSubmit();
}

function onSlotClick(queue, start) {
  if (state.pick && state.pick.queue === queue && state.pick.start === start) {
    state.pick = null;
  } else {
    state.pick = { queue, start };
  }
  renderQueues();
}

function setType(newType) {
  if (newType === state.type) return;
  state.type = newType;
  state.pick = null;
  els.typeCards.forEach(c => c.setAttribute('aria-checked', String(c.dataset.type === newType)));
  renderQueues();
}

function updateSubmit() {
  const ok = state.name.trim().length > 0 && state.pick !== null;
  els.submit.disabled = !ok;
}

function setStatus(msg, kind) {
  els.status.textContent = msg || '';
  els.status.className = 'status' + (kind ? ' ' + kind : '');
}

function renderSchedule() {
  els.schedule.innerHTML = '';
  const grid = buildGrid();
  const total = state.baseSlots.length;

  for (let cell = 0; cell < total; cell++) {
    // Aggregate bookings starting exactly at this cell across all queues
    const rowBookings = [];
    for (let qi = 0; qi < QUEUES_COUNT; qi++) {
      const b = grid[qi][cell];
      if (b && b.start === cell && b.queue === qi) {
        rowBookings.push({ qi, b });
      }
    }

    // Detect cells fully empty (no booking covers this cell in any queue)
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
    if (state.pick) {
      const grid = buildGrid();
      const dur = durationCells();
      let invalid = state.pick.start + dur > state.baseSlots.length;
      if (!invalid) {
        for (let i = 0; i < dur; i++) {
          if (grid[state.pick.queue][state.pick.start + i]) { invalid = true; break; }
        }
      }
      if (invalid) state.pick = null;
    }

    renderQueues();
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
  if (!state.pick) {
    setStatus('Pick a queue and time.', 'error');
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
        queue: state.pick.queue,
        pick: state.pick.start,
      }),
    });
    const data = await r.json();
    if (!r.ok) {
      setStatus(data.error || 'Booking failed.', 'error');
      await loadState();
      return;
    }
    const picked = state.pick;
    state.pick = null;
    const q = state.queues[picked.queue];
    const who = q && q.interviewer ? ` with ${q.interviewer}` : '';
    setStatus(`Booked! ${name} · Queue ${picked.queue + 1}${who} at ${rangeLabel(picked.start, durationCells())}.`, 'success');
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
