const $ = (s) => document.querySelector(s);

const priceCtx = document.getElementById('priceChart');
const equityCtx = document.getElementById('equityChart');

const priceChart = new Chart(priceCtx, {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Price', data: [], borderColor: '#2e86de', tension: 0.1 }] },
  options: { responsive: true, scales: { x: { display: false } } }
});

const equityChart = new Chart(equityCtx, {
  type: 'line',
  data: { labels: [], datasets: [{ label: 'Equity', data: [], borderColor: '#27ae60', tension: 0.1 }] },
  options: { responsive: true, scales: { x: { display: false } } }
});

async function refreshStatus() {
  const res = await fetch('/api/status');
  const st = await res.json();
  $('#status').textContent = `وضعیت: ${st.running ? 'در حال اجرا' : 'متوقف'}`;
}

async function refreshMetrics() {
  const res = await fetch('/api/metrics');
  const m = await res.json();
  const price = m.price || [];
  const equity = m.equity || [];

  priceChart.data.labels = price.map(p => p.t);
  priceChart.data.datasets[0].data = price.map(p => p.price);
  priceChart.update('none');

  equityChart.data.labels = equity.map(p => p.t);
  equityChart.data.datasets[0].data = equity.map(p => p.equity);
  equityChart.update('none');

  const tbody = document.querySelector('#trades tbody');
  tbody.innerHTML = '';
  (m.trades || []).slice(-100).forEach(tr => {
    const trEl = document.createElement('tr');
    trEl.innerHTML = `<td>${tr.time||''}</td><td>${tr.symbol||''}</td><td>${tr.side||''}</td><td>${tr.qty||''}</td><td>${tr.price||''}</td><td>${tr.status||''}</td>`;
    tbody.appendChild(trEl);
  });

  const evDiv = document.getElementById('events');
  if (evDiv) {
    evDiv.innerHTML = '';
    (m.events || []).forEach(e => {
      const p = document.createElement('div');
      p.textContent = e;
      evDiv.appendChild(p);
    });
  }
}

function pushEvent(msg) {
  const evDiv = document.getElementById('events');
  if (!evDiv) return;
  const p = document.createElement('div');
  p.textContent = msg;
  evDiv.prepend(p);
}

$('#startBtn').addEventListener('click', async () => {
  const startBtn = $('#startBtn');
  const stopBtn = $('#stopBtn');
  startBtn.disabled = true; stopBtn.disabled = true;
  try {
    const mode = $('#mode').value;
    const symbol = $('#symbol').value || undefined;
    const budget = $('#budget').value ? Number($('#budget').value) : undefined;
    const interval = $('#interval').value ? Number($('#interval').value) : undefined;
    const params = new URLSearchParams();
    if (mode) params.set('mode', mode);
    if (symbol) params.set('symbol', symbol);
    if (budget) params.set('budget', String(budget));
    if (interval) params.set('loop_interval', String(interval));
    const res = await fetch(`/api/start?${params.toString()}`, { method: 'POST' });
    let msg = 'ربات در حال شروع...';
    try {
      const body = await res.json();
      if (!res.ok || body.ok === false) {
        msg = `خطا در شروع ربات: ${body.error || res.statusText}`;
      } else if (body.already_running) {
        msg = 'ربات از قبل در حال اجراست.';
      } else {
        msg = 'ربات شروع شد.';
      }
    } catch (e) {
      if (!res.ok) {
        msg = `خطا در شروع ربات: ${res.status} ${res.statusText}`;
      }
    }
    pushEvent(msg);
  } catch (e) {
    pushEvent(`خطا در شروع: ${e}`);
  } finally {
    await refreshStatus();
    startBtn.disabled = false; stopBtn.disabled = false;
  }
});
$('#stopBtn').addEventListener('click', async () => {
  const startBtn = $('#startBtn');
  const stopBtn = $('#stopBtn');
  startBtn.disabled = true; stopBtn.disabled = true;
  try {
    const res = await fetch('/api/stop', { method: 'POST' });
    let msg = 'درخواست توقف ارسال شد.';
    try {
      const body = await res.json();
      if (!res.ok || body.ok === false) {
        msg = `خطا در توقف ربات: ${body.error || res.statusText}`;
      } else {
        msg = 'ربات متوقف شد.';
      }
    } catch (e) {
      if (!res.ok) {
        msg = `خطا در توقف ربات: ${res.status} ${res.statusText}`;
      }
    }
    pushEvent(msg);
  } catch (e) {
    pushEvent(`خطا در توقف: ${e}`);
  } finally {
    await refreshStatus();
    startBtn.disabled = false; stopBtn.disabled = false;
  }
});

$('#checkLiveBtn').addEventListener('click', async () => {
  const res = await fetch('/api/check-live-balance');
  const j = await res.json();
  const el = document.getElementById('liveBalance');
  if (j.ok) {
    el.textContent = `موجودی لایو → پایه: ${j.base.currency} کل=${j.base.total} قابل‌استفاده=${j.base.available} | مظنه: ${j.quote.currency} کل=${j.quote.total} قابل‌استفاده=${j.quote.available}`;
  } else {
    el.textContent = `خطای موجودی لایو: ${j.error || 'خطای نامشخص'}`;
  }
});

setInterval(refreshStatus, 2000);
setInterval(refreshMetrics, 2000);
refreshStatus();
refreshMetrics();
