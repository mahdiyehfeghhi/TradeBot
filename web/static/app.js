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
    trEl.innerHTML = `<td>${tr.time||''}</td><td>${tr.symbol||''}</td><td>${tr.side||''}</td><td>${tr.qty||''}</td><td>${tr.price||''}</td><td>${tr.status||''}</td><td>-</td><td>-</td>`;
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

// Enhanced functions for new features
async function refreshMarketPerformance() {
  try {
    const res = await fetch('/api/market-performance');
    const data = await res.json();
    
    if (data.ok) {
      const container = document.getElementById('marketPerformance');
      const sentiment = document.getElementById('marketSentiment');
      
      // Update market sentiment
      const marketSentiment = data.market_sentiment;
      sentiment.textContent = `بازار: ${marketSentiment.sentiment} (قدرت: ${(marketSentiment.strength * 100).toFixed(0)}%)`;
      sentiment.className = `sentiment ${marketSentiment.sentiment}`;
      
      // Update performance cards
      container.innerHTML = '';
      data.performances.forEach(perf => {
        const card = document.createElement('div');
        card.className = `market-card ${perf.price_change_24h >= 0 ? 'positive' : 'negative'}`;
        card.innerHTML = `
          <div class="symbol">${perf.symbol}</div>
          <div class="change ${perf.price_change_24h >= 0 ? 'positive' : 'negative'}">
            ${perf.price_change_24h >= 0 ? '+' : ''}${perf.price_change_24h.toFixed(2)}%
          </div>
          <div class="price">قیمت: ${perf.current_price.toLocaleString()}</div>
          <div class="volume">حجم: ${(perf.volume_24h / 1000000).toFixed(1)}M</div>
        `;
        container.appendChild(card);
      });
    }
  } catch (e) {
    console.error('Error refreshing market performance:', e);
  }
}

async function refreshStrategyPerformance() {
  try {
    const res = await fetch('/api/learning/strategy-performance');
    const data = await res.json();
    
    if (data.ok) {
      const container = document.getElementById('strategyPerformance');
      container.innerHTML = '';
      
      data.strategies.forEach(strategy => {
        const card = document.createElement('div');
        card.className = 'strategy-card';
        card.innerHTML = `
          <div class="strategy-name">${strategy.strategy_name}</div>
          <div class="strategy-metrics">
            <div class="metric">
              <span>تعداد معاملات:</span>
              <span>${strategy.total_trades}</span>
            </div>
            <div class="metric">
              <span>نرخ برد:</span>
              <span>${(strategy.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div class="metric">
              <span>سود کل:</span>
              <span class="${strategy.total_pnl >= 0 ? 'positive' : 'negative'}">${strategy.total_pnl.toFixed(0)}</span>
            </div>
            <div class="metric">
              <span>ضریب سود:</span>
              <span>${strategy.profit_factor.toFixed(2)}</span>
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    }
  } catch (e) {
    console.error('Error refreshing strategy performance:', e);
  }
}

async function refreshLearningInsights() {
  try {
    const res = await fetch('/api/learning/insights');
    const data = await res.json();
    
    if (data.ok) {
      const container = document.getElementById('learningInsights');
      const insights = data.insights;
      
      container.innerHTML = `
        <div class="insight-section">
          <h4>بهترین استراتژی‌ها</h4>
          <div class="strategy-list">
            ${insights.best_strategies.map(s => `
              <div class="strategy-item">
                ${s.strategy_used}: ${(s.win_rate * 100).toFixed(1)}% نرخ برد
              </div>
            `).join('')}
          </div>
        </div>
        
        <div class="insight-section">
          <h4>بهترین ساعات معاملاتی</h4>
          <div class="time-list">
            ${insights.best_trading_hours.map(h => `
              <div class="time-item">
                ساعت ${h.hour}: ${(h.win_rate * 100).toFixed(1)}% نرخ برد
              </div>
            `).join('')}
          </div>
        </div>
        
        <div class="insight-section">
          <h4>توصیه‌ها</h4>
          <ul class="recommendations">
            ${insights.recommendations.map(r => `<li>${r}</li>`).join('')}
          </ul>
        </div>
      `;
    }
  } catch (e) {
    console.error('Error refreshing learning insights:', e);
  }
}

async function refreshTradingEvents() {
  try {
    const hours = document.getElementById('eventTimeframe').value || 24;
    const res = await fetch(`/api/learning/events?hours=${hours}`);
    const data = await res.json();
    
    if (data.ok) {
      const tbody = document.querySelector('#tradingEvents tbody');
      tbody.innerHTML = '';
      
      data.events.forEach(event => {
        const row = document.createElement('tr');
        const timeStr = new Date(event.timestamp * 1000).toLocaleString('fa-IR');
        const pnlText = event.pnl ? event.pnl.toFixed(0) : '-';
        const outcomeClass = event.outcome === 'win' ? 'positive' : (event.outcome === 'loss' ? 'negative' : '');
        
        row.innerHTML = `
          <td>${timeStr}</td>
          <td>${event.symbol}</td>
          <td>${event.action}</td>
          <td>${event.reason}</td>
          <td>${event.entry_price.toFixed(2)}</td>
          <td>${event.exit_price ? event.exit_price.toFixed(2) : '-'}</td>
          <td class="${outcomeClass}">${pnlText}</td>
          <td>${event.outcome || '-'}</td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (e) {
    console.error('Error refreshing trading events:', e);
  }
}

function pushEvent(msg) {
  const evDiv = document.getElementById('events');
  if (!evDiv) return;
  const p = document.createElement('div');
  p.textContent = msg;
  evDiv.prepend(p);
}

// Enhanced start function to include strategy selection
$('#startBtn').addEventListener('click', async () => {
  const startBtn = $('#startBtn');
  const stopBtn = $('#stopBtn');
  startBtn.disabled = true; stopBtn.disabled = true;
  try {
    const mode = $('#mode').value;
    const symbol = $('#symbol').value || undefined;
    const budget = $('#budget').value ? Number($('#budget').value) : undefined;
    const interval = $('#interval').value ? Number($('#interval').value) : undefined;
    const strategy = $('#strategy').value || undefined;
    
    const params = new URLSearchParams();
    if (mode) params.set('mode', mode);
    if (symbol) params.set('symbol', symbol);
    if (budget) params.set('budget', String(budget));
    if (interval) params.set('loop_interval', String(interval));
    if (strategy) params.set('strategy', strategy);
    
    const res = await fetch(`/api/start?${params.toString()}`, { method: 'POST' });
    let msg = 'ربات در حال شروع...';
    try {
      const body = await res.json();
      if (!res.ok || body.ok === false) {
        msg = `خطا در شروع ربات: ${body.error || res.statusText}`;
      } else if (body.already_running) {
        msg = 'ربات از قبل در حال اجراست.';
      } else {
        msg = `ربات شروع شد با استراتژی ${strategy || 'پیش‌فرض'}.`;
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

// Event listeners for new features
document.getElementById('refreshMarketBtn').addEventListener('click', refreshMarketPerformance);
document.getElementById('refreshStrategyBtn').addEventListener('click', refreshStrategyPerformance);
document.getElementById('refreshInsightsBtn').addEventListener('click', refreshLearningInsights);
document.getElementById('refreshEventsBtn').addEventListener('click', refreshTradingEvents);
document.getElementById('eventTimeframe').addEventListener('change', refreshTradingEvents);

// Initialize enhanced features
setInterval(refreshStatus, 2000);
setInterval(refreshMetrics, 2000);
setInterval(refreshMarketPerformance, 30000); // Refresh market every 30 seconds
setInterval(refreshStrategyPerformance, 60000); // Refresh strategy performance every minute

// Initial load
refreshStatus();
refreshMetrics();
refreshMarketPerformance();
refreshStrategyPerformance();
refreshLearningInsights();
refreshTradingEvents();

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
