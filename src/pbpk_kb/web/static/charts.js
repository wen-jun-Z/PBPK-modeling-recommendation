/* SVG plots of actual returned data. No interpolation, smoothing or invented points. */
'use strict';
window.PBPKCharts = (() => {
  const escape = v => String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colors=['#286b5a','#6b8eaf','#b38a55','#9982ae','#b7747c','#648e8b','#7d8652','#79828f'];
  const number=v=>Math.abs(v)>=10000||Math.abs(v)>0&&Math.abs(v)<.001?Number(v).toExponential(1):Number(v.toPrecision(4)).toLocaleString('en-US');
  function curve(series,{xLabel='时间（h）',yLabel='浓度（ng/mL）',title='浓度—时间曲线'}={}){
    const data=series.map(s=>({...s,points:(s.points||[]).filter(p=>Array.isArray(p)&&p.length>=2&&p[0]!=null&&p[1]!=null&&Number.isFinite(Number(p[0]))&&Number.isFinite(Number(p[1]))).map(p=>[Number(p[0]),Number(p[1])]).sort((a,b)=>a[0]-b[0])})).filter(s=>s.points.length);
    if(!data.length)return '<div class="empty">此结果暂无可绘制的曲线。</div>';
    const all=data.flatMap(s=>s.points),w=820,h=340,m={l:78,r:24,t:26,b:56};
    let xmin=Math.min(0,...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0]));
    let ymin=Math.min(0,...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));
    if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
    ymax+=(ymax-ymin)*.08;
    const x=v=>m.l+(v-xmin)/(xmax-xmin)*(w-m.l-m.r),y=v=>h-m.b-(v-ymin)/(ymax-ymin)*(h-m.t-m.b);
    const ticks=Array.from({length:6},(_,i)=>i/5);
    return `<div class="pk-chart"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escape(title)}"><title>${escape(title)}</title>${ticks.map(f=>{const v=ymin+(ymax-ymin)*f;return `<line x1="${m.l}" y1="${y(v)}" x2="${w-m.r}" y2="${y(v)}" stroke="#e8ede8"/><text x="${m.l-10}" y="${y(v)+4}" text-anchor="end">${escape(number(v))}</text>`}).join('')}${ticks.map(f=>{const v=xmin+(xmax-xmin)*f;return `<text x="${x(v)}" y="${h-m.b+23}" text-anchor="middle">${escape(number(v))}</text>`}).join('')}<line x1="${m.l}" y1="${h-m.b}" x2="${w-m.r}" y2="${h-m.b}" stroke="#c2d1c7"/>${data.map((s,i)=>s.kind==='points'?s.points.map(p=>`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3.5" fill="${colors[i%colors.length]}" stroke="white" stroke-width="1"><title>${escape(s.name)} · ${number(p[0])}，${number(p[1])}</title></circle>`).join(''):`<polyline fill="none" stroke="${colors[i%colors.length]}" stroke-width="2.3" stroke-linejoin="round" ${i>5?'stroke-dasharray="6 3"':''} points="${s.points.map(p=>`${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ')}"><title>${escape(s.name)}</title></polyline>`).join('')}<text x="${(w+m.l-m.r)/2}" y="${h-9}" text-anchor="middle" class="axis-label">${escape(xLabel)}</text><text transform="translate(18 ${h/2}) rotate(-90)" text-anchor="middle" class="axis-label">${escape(yLabel)}</text></svg><div class="chart-legend">${data.map((s,i)=>`<span><i style="background:${colors[i%colors.length]};border-radius:${s.kind==='points'?'50%':'2px'}"></i>${escape(s.name)}</span>`).join('')}</div></div>`;
  }
  function bars(rows,{valueLabel='数值'}={}){
    const values=rows.filter(r=>r.value!=null&&Number.isFinite(Number(r.value))),max=Math.max(1,...values.map(r=>Number(r.value)));
    return `<div class="result-bars" aria-label="${escape(valueLabel)}">${values.map((r,i)=>`<div><span>${escape(r.label)}</span><div><i style="width:${Math.max(0,Number(r.value))/max*100}%;background:${colors[i%colors.length]}"></i></div><strong>${number(Number(r.value))}</strong></div>`).join('')}</div>`;
  }
  return {curve,bars};
})();
