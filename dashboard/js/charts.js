/* ============================================================
   MLOps Experiments Dashboard - Charts Module
   Canvas-based charts for experiment visualization
   ============================================================ */

const Charts = (() => {

  const COLORS = {
    primary: '#818cf8',
    secondary: '#6366f1',
    completed: '#34d399',
    partial: '#fbbf24',
    blocked: '#f87171',
    external: '#38bdf8',
    sentiment: '#f472b6',
    nlpClass: '#fb923c',
    timeseries: '#34d399',
    cv: '#38bdf8',
    anomaly: '#a78bfa',
    ibm: '#fbbf24',
    regression: '#f87171',
    grid: 'rgba(129, 140, 248, 0.06)',
    gridLine: 'rgba(129, 140, 248, 0.1)',
    textMuted: '#6b71a0',
    textSecondary: '#9fa4c4',
  };

  function getPixelRatio(ctx) {
    return window.devicePixelRatio || 1;
  }

  function setupCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = getPixelRatio();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height };
  }

  /* -------------------------------------------------------
     Category Distribution - Horizontal Bar Chart
     ------------------------------------------------------- */
  function drawCategoryChart(canvas, data) {
    const { ctx, w, h } = setupCanvas(canvas);
    const padding = { top: 10, right: 20, bottom: 20, left: 140 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    const maxVal = Math.max(...data.map(d => d.count));
    const barHeight = Math.min(28, (chartH / data.length) - 8);
    const gap = (chartH - barHeight * data.length) / (data.length + 1);

    // Grid lines
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const x = padding.left + (chartW / 4) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, h - padding.bottom);
      ctx.stroke();
    }

    data.forEach((item, i) => {
      const y = padding.top + gap + i * (barHeight + gap);
      const barW = (item.count / maxVal) * chartW * 0.9;

      // Label
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.label, padding.left - 12, y + barHeight / 2);

      // Bar background
      ctx.fillStyle = COLORS.grid;
      roundRect(ctx, padding.left, y, chartW * 0.9, barHeight, 4);
      ctx.fill();

      // Bar fill with gradient
      const grad = ctx.createLinearGradient(padding.left, 0, padding.left + barW, 0);
      grad.addColorStop(0, item.color);
      grad.addColorStop(1, item.color + '88');
      ctx.fillStyle = grad;
      roundRect(ctx, padding.left, y, barW, barHeight, 4);
      ctx.fill();

      // Count text
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(item.count, padding.left + barW + 8, y + barHeight / 2);
    });
  }

  /* -------------------------------------------------------
     Status Donut Chart
     ------------------------------------------------------- */
  function drawStatusChart(canvas, data) {
    const { ctx, w, h } = setupCanvas(canvas);
    const cx = w * 0.38;
    const cy = h / 2;
    const radius = Math.min(cx - 20, cy - 20, 100);
    const innerRadius = radius * 0.62;
    const total = data.reduce((s, d) => s + d.count, 0);

    let startAngle = -Math.PI / 2;

    data.forEach((item, i) => {
      const sliceAngle = (item.count / total) * Math.PI * 2;
      const endAngle = startAngle + sliceAngle;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, endAngle);
      ctx.arc(cx, cy, innerRadius, endAngle, startAngle, true);
      ctx.closePath();
      ctx.fillStyle = item.color;
      ctx.fill();

      // Slight gap between segments
      ctx.strokeStyle = '#0a0c1a';
      ctx.lineWidth = 2;
      ctx.stroke();

      startAngle = endAngle;
    });

    // Center text
    ctx.fillStyle = '#e8eaf6';
    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(total, cx, cy - 6);
    ctx.fillStyle = COLORS.textMuted;
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Total', cx, cy + 16);

    // Legend
    const legendX = w * 0.65;
    let legendY = cy - (data.length * 28) / 2;

    data.forEach(item => {
      // Dot
      ctx.beginPath();
      ctx.arc(legendX, legendY + 6, 5, 0, Math.PI * 2);
      ctx.fillStyle = item.color;
      ctx.fill();

      // Label
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(item.label, legendX + 14, legendY + 10);

      // Count
      ctx.fillStyle = '#e8eaf6';
      ctx.font = 'bold 12px JetBrains Mono, monospace';
      ctx.fillText(item.count, legendX + 14 + ctx.measureText(item.label).width + 8, legendY + 10);

      legendY += 28;
    });
  }

  /* -------------------------------------------------------
     Senti-Pred Evolution Timeline Chart
     ------------------------------------------------------- */
  function drawEvolutionChart(canvas, data) {
    const { ctx, w, h } = setupCanvas(canvas);
    const padding = { top: 20, right: 30, bottom: 50, left: 50 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    const maxVal = 100;
    const minVal = 50;
    const range = maxVal - minVal;

    // Grid
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    for (let v = minVal; v <= maxVal; v += 10) {
      const y = padding.top + chartH - ((v - minVal) / range) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.textMuted;
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(v + '%', padding.left - 8, y);
    }

    const stepX = chartW / (data.length - 1);

    // Area fill
    ctx.beginPath();
    data.forEach((point, i) => {
      const x = padding.left + i * stepX;
      const y = padding.top + chartH - ((point.value - minVal) / range) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Close area
    ctx.lineTo(padding.left + (data.length - 1) * stepX, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    const areaGrad = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    areaGrad.addColorStop(0, 'rgba(129, 140, 248, 0.2)');
    areaGrad.addColorStop(1, 'rgba(129, 140, 248, 0.0)');
    ctx.fillStyle = areaGrad;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((point, i) => {
      const x = padding.left + i * stepX;
      const y = padding.top + chartH - ((point.value - minVal) / range) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    const lineGrad = ctx.createLinearGradient(padding.left, 0, w - padding.right, 0);
    lineGrad.addColorStop(0, COLORS.primary);
    lineGrad.addColorStop(0.5, COLORS.sentiment);
    lineGrad.addColorStop(1, COLORS.completed);
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Points
    data.forEach((point, i) => {
      const x = padding.left + i * stepX;
      const y = padding.top + chartH - ((point.value - minVal) / range) * chartH;

      // Outer glow for best
      if (point.best) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(52, 211, 153, 0.15)';
        ctx.fill();
      }

      // Point
      ctx.beginPath();
      ctx.arc(x, y, point.best ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = point.best ? COLORS.completed : COLORS.primary;
      ctx.fill();
      ctx.strokeStyle = '#0a0c1a';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      ctx.save();
      ctx.translate(x, padding.top + chartH + 12);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = COLORS.textMuted;
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(point.label, 0, 0);
      ctx.restore();
    });
  }

  /* -------------------------------------------------------
     Techniques Radar Chart
     ------------------------------------------------------- */
  function drawTechRadar(canvas, data) {
    const { ctx, w, h } = setupCanvas(canvas);
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 40;
    const n = data.length;
    const maxVal = Math.max(...data.map(d => d.count));

    // Background rings
    for (let r = 1; r <= 4; r++) {
      const ringR = (radius / 4) * r;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * ringR;
        const y = cy + Math.sin(angle) * ringR;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = COLORS.gridLine;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Axis lines
    for (let i = 0; i < n; i++) {
      const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = COLORS.gridLine;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Data polygon
    ctx.beginPath();
    data.forEach((item, i) => {
      const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
      const r = (item.count / maxVal) * radius;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(129, 140, 248, 0.15)';
    ctx.fill();
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points and labels
    data.forEach((item, i) => {
      const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
      const r = (item.count / maxVal) * radius;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;

      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.primary;
      ctx.fill();

      // Label
      const labelR = radius + 18;
      const lx = cx + Math.cos(angle) * labelR;
      const ly = cy + Math.sin(angle) * labelR;
      ctx.fillStyle = COLORS.textMuted;
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = angle > Math.PI / 2 || angle < -Math.PI / 2 ? 'right' : angle === -Math.PI / 2 || angle === Math.PI / 2 ? 'center' : 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(item.label, lx, ly);
    });
  }

  /* -------------------------------------------------------
     Helper: rounded rect
     ------------------------------------------------------- */
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  /* -------------------------------------------------------
     Public API
     ------------------------------------------------------- */
  return {
    drawCategoryChart,
    drawStatusChart,
    drawEvolutionChart,
    drawTechRadar,
    COLORS,
  };

})();
