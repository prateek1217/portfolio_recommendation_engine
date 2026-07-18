// Pravar.AI agent — two-step, live-streaming frontend.
// Step 1 "Fetch news": calls /api/fetch, shows what AgentQL returned.
// Step 2 "Analyze": opens an SSE stream to /api/stream and paints each
//         investor recommendation card the moment the server emits it.

let STAGE1 = null;  // cached event+impact+affected from step 1

const $ = (id) => document.getElementById(id);
const esc = (x) => (x ?? '').toString().replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const sev = (s) => (s || 'low').toLowerCase();

async function loadFields() {
  const d = await (await fetch('/api/fields')).json();
  const sel = $('field');
  d.fields.forEach(f => {
    const o = document.createElement('option');
    o.value = f.name;
    o.textContent = `${f.name}  (${f.sectors.slice(0,3).join(', ')})`;
    sel.appendChild(o);
  });
  $('mock').checked = !!d.mock_default;
}

function setStatus(txt, spinning) {
  $('status').innerHTML = (spinning ? '<span class="spin"></span>' : '') + esc(txt);
}

// ---- Step 1: fetch news -------------------------------------------------
async function fetchNews() {
  const field = $('field').value;
  const url = $('url').value.trim();
  const use_mock = $('mock').checked;

  $('btnFetch').disabled = true;
  $('btnAnalyze').disabled = true;
  $('event').innerHTML = '';
  $('alerts').innerHTML = '';
  $('trace').innerHTML = '';
  STAGE1 = null;

  const log = $('fetchlog');
  log.style.display = 'block';
  const t0 = Date.now();
  log.innerHTML = '<div class="row"><b>AgentQL</b> — contacting live source…</div>' +
    '<div class="row" id="ftimer">elapsed 0s… (the news call has no artificial timeout; heavy pages can take a while)</div>';
  const timer = setInterval(() => {
    const el = $('ftimer');
    if (el) el.textContent = `elapsed ${Math.round((Date.now()-t0)/1000)}s…`;
  }, 1000);

  setStatus('fetching news…', true);
  try {
    const r = await fetch('/api/fetch', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ field, news_url: url || null, use_mock })
    });
    if (!r.ok) throw new Error('server ' + r.status + ': ' + (await r.text()));
    STAGE1 = await r.json();

    clearInterval(timer);
    const ev = STAGE1.event || {};
    const live = (ev.source || '').startsWith('agentql-live');
    log.innerHTML =
      `<div class="row"><b>Fetched in ${Math.round((Date.now()-t0)/1000)}s</b> via ` +
      `<code>${esc(ev.source)}</code></div>` +
      `<div class="row">Headline: <b>${esc(ev.headline)}</b></div>` +
      `<div class="row">${esc(ev.summary || '')}</div>` +
      (STAGE1.candidates && STAGE1.candidates.length
        ? `<div class="row" style="margin-top:6px">Other headlines seen on page: ` +
          STAGE1.candidates.slice(0,4).map(c => '<code>'+esc(c)+'</code>').join(' · ') + '</div>'
        : '');

    renderEvent(STAGE1.event, STAGE1.impact, STAGE1.affected_investors.length);
    setStatus(`${STAGE1.affected_investors.length} affected investors ready — run step 2`, false);
    $('btnAnalyze').disabled = STAGE1.affected_investors.length === 0;
  } catch (e) {
    clearInterval(timer);
    $('event').innerHTML = '<div class="err">⚠️ ' + esc(e.message) + '</div>';
    setStatus('fetch failed', false);
  } finally {
    $('btnFetch').disabled = false;
  }
}

function renderEvent(ev, im, affectedCount) {
  ev = ev || {}; im = im || {};
  const live = (ev.source || '').startsWith('agentql-live');
  let h = '<div class="event"><h2>' + esc(ev.headline || '(no event)') +
    `<span class="pill ${live?'live':'cached'}">${live?'live':'cached'}</span></h2>` +
    '<div class="src">' + esc(ev.published || '') +
    (ev.source_url ? ' · <a href="' + esc(ev.source_url) + '" target="_blank">source</a>' : '') +
    '</div><div style="margin-top:8px">' + esc(ev.summary || '') + '</div>';
  if (im.one_line_summary)
    h += '<div style="margin-top:10px;color:#cdd6f0"><b>Impact:</b> ' + esc(im.one_line_summary) + '</div>';
  h += '<div class="sectors">';
  (im.affected_sectors || []).forEach(s => {
    h += `<span class="sector ${s.impact==='negative'?'neg':'pos'}">` +
      esc(s.sector) + ' <span class="ord">· ' + esc(s.order) + '</span></span>';
  });
  h += '</div></div>';
  $('event').innerHTML = h;
}

// ---- Step 2: stream recommendations -------------------------------------
function analyze() {
  if (!STAGE1) return;
  $('btnAnalyze').disabled = true;
  $('btnFetch').disabled = true;

  const total = STAGE1.affected_investors.length;
  $('alerts').innerHTML =
    `<h3 class="section">RM Alerts — <span id="done">0</span>/${total} ready</h3>` +
    '<div class="cards" id="cards"></div>';
  const cards = $('cards');
  let done = 0;
  const t0 = Date.now();
  setStatus('generating recommendations…', true);

  // POST the stage-1 payload, read the SSE stream back.
  fetch('/api/stream', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      event: STAGE1.event, impact: STAGE1.impact,
      affected_investors: STAGE1.affected_investors
    })
  }).then(async (resp) => {
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done: rdDone } = await reader.read();
      if (rdDone) break;
      buf += dec.decode(value, { stream:true });
      // SSE frames are separated by a blank line.
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const line = frame.split('\n').find(l => l.startsWith('data:'));
        if (!line) continue;
        const msg = JSON.parse(line.slice(5).trim());
        if (msg.type === 'rec') {
          cards.insertAdjacentHTML('beforeend', cardHTML(msg.rec));
          done++; $('done').textContent = done;
        } else if (msg.type === 'done') {
          setStatus(`done — ${done}/${total} in ${Math.round((Date.now()-t0)/1000)}s`, false);
          renderTrace(msg.trace, Math.round((Date.now()-t0)/1000));
        }
      }
    }
  }).catch(e => {
    $('alerts').insertAdjacentHTML('beforeend', '<div class="err">⚠️ ' + esc(e.message) + '</div>');
    setStatus('stream error', false);
  }).finally(() => {
    $('btnFetch').disabled = false;
    $('btnAnalyze').disabled = false;
  });
}

function cardHTML(r) {
  let h = '<div class="card"><div class="top"><div>' +
    '<div class="who">' + esc(r.name) + '</div>' +
    '<div class="meta">' + esc(r.risk_profile) + ' · ' + esc(r.total_exposure_pct) +
    '% exposed · ' + esc((r.matched_sectors || []).join(', ')) + '</div></div>' +
    `<span class="sev ${sev(r.severity)}">${esc((r.severity||'').toUpperCase())}</span></div>`;
  h += '<div class="rhead">' + esc(r.headline || '') + '</div>';
  h += '<div class="rat">' + esc(r.rationale || '') + '</div>';
  h += '<div class="actions">';
  (r.actions || []).forEach(a => {
    h += `<div class="act"><span class="t ${esc((a.type||'').toLowerCase())}">` +
      esc((a.type||'').toUpperCase()) + '</span>' +
      '<span class="body"><span class="fid">' + esc(a.fund_id) + '</span> ' +
      esc(a.fund_name || '') + '<br><span style="color:#9fb2e6">' +
      esc(a.reason || '') + '</span></span></div>';
  });
  if (!(r.actions || []).length) h += '<div class="meta">No in-universe action generated.</div>';
  h += '</div></div>';
  return h;
}

function renderTrace(trace, secs) {
  if (!trace || !trace.length) return;
  let h = `<div class="trace"><b>Pipeline trace · completed in ${secs}s:</b><br>`;
  trace.forEach(t => h += '· <code>' + esc(t) + '</code><br>');
  h += '</div>';
  $('trace').innerHTML = h;
}

loadFields();
