/* Private, dependency-free graph explorer. Groups preserve the original edges. */
'use strict';
window.PBPKGraph = (() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const kinds = {compound:'化合物',evidence:'证据',source:'数据来源',model:'公共模型',protein_target:'酶与转运体'};
  const categories = {adme:'吸收与代谢',clinical_pk:'临床药代',ddi:'药物相互作用',physicochemical:'理化性质',public_pbpk_model:'模型证据',special_population:'特殊人群',tissue_expression:'组织表达'};
  const relations = {HAS_EVIDENCE:'具有证据',DERIVED_FROM:'来源于',REPRESENTED_BY:'由模型描述',SUPPORTS_RELATIONSHIP:'提供关系证据',substrate_of:'是其底物',not_substrate_of:'不是其底物',transported_by:'由其转运',inhibits:'抑制',induces:'诱导'};
  const parameters = {volume_of_distribution:'分布容积',clearance:'清除率',half_life:'半衰期','half-life':'半衰期',bioavailability:'生物利用度',excretion_unchanged:'原型药排泄',protein_binding:'蛋白结合率'};
  const text = value => parameters[value] || categories[value] || value || '未命名';
  const titleOf = n => n.label==='osp_pbpk_model_library'?'OSP 模型库':text(n.label || n.name || n.id);
  const sourceLink = n => {
    const p=n.properties||{};
    return p.evidence_type&&p.evidence_id ? `<a class="graph-source-link" href="/api/workspace/evidence/${encodeURIComponent(p.evidence_type)}/${encodeURIComponent(p.evidence_id)}/source" target="_blank" rel="noopener noreferrer">查看来源 ↗</a>` : '';
  };
  const markup = () => `<section class="graph-explorer" aria-label="交互知识图谱">
    <header class="graph-toolbar"><div class="graph-mode" role="group" aria-label="图谱视图"><button data-graph-action="overview" aria-pressed="true">分类概览</button><button data-graph-action="evidence" aria-pressed="false">逐条证据</button></div><div class="graph-tools" role="group" aria-label="图谱缩放"><button data-graph-action="out" aria-label="缩小图谱">−</button><span class="graph-zoom">100%</span><button data-graph-action="in" aria-label="放大图谱">＋</button><button data-graph-action="reset">适应画布</button></div></header>
    <div class="graph-legend">${Object.entries(kinds).map(([key,name])=>`<span><i class="graph-swatch ${key}"></i>${name}</span>`).join('')}</div>
    <div class="graph-evidence-controls" hidden><label>证据类别<select aria-label="图谱证据类别"></select></label><span class="graph-page-status"></span><button data-graph-action="prev" aria-label="上一页证据">←</button><button data-graph-action="next" aria-label="下一页证据">→</button></div>
    <div class="graph-workspace"><div class="graph-stage"><svg class="graph-svg" role="group" aria-label="化合物知识关系图" tabindex="0"></svg><div class="graph-gesture">滚轮缩放 · 拖动画布 · 点击节点查看关联</div></div><aside class="graph-inspector" aria-label="节点详情" aria-live="polite"></aside></div>
    <footer class="graph-caption"></footer></section>`;

  function mount(container, data, loadEvidence) {
    const root=container.querySelector('.graph-explorer');
    if(!root)return ()=>{};
    const abort=new AbortController(), signal=abort.signal;
    const svg=root.querySelector('svg'), inspector=root.querySelector('.graph-inspector');
    const nodes=data.nodes||[], edges=data.edges||data.links||[];
    const byId=new Map(nodes.map(n=>[n.id,n]));
    const compound=nodes.find(n=>n.type==='compound');
    const evidence=nodes.filter(n=>n.type==='evidence');
    const types=[...new Set(evidence.map(n=>n.properties?.evidence_type||'other'))];
    const select=root.querySelector('select');
    select.innerHTML='<option value="">全部证据类别</option>'+types.map(t=>`<option value="${escape(t)}">${escape(categories[t]||t)} · ${evidence.filter(n=>n.properties?.evidence_type===t).length}</option>`).join('');
    let mode='overview', filter='', page=0, selected='', revision=0, view, alias, drawn, drawnEdges;
    let zoom=1, pan={x:0,y:0}, drag=null, moved=false;
    const pageSize=8;
    const evidenceCache=new Map();
    const groupId=n=>n.type==='evidence'?'group:evidence:'+n.properties?.evidence_type:['model','protein_target'].includes(n.type)?'group:'+n.type:n.id;
    const memberIds=n=>new Set((n.members||[n]).map(x=>x.id));
    const describe=n=>n.group?`${n.members.length} ${n.type==='evidence'?'条证据':n.type==='model'?'个登记模型':'个酶/转运体'}`:kinds[n.type]||n.type;

    function build() {
      alias=new Map();
      const result=new Map();
      const included=new Set();
      const filtered=evidence.filter(n=>!filter||n.properties?.evidence_type===filter);
      if(mode==='evidence') {
        page=Math.min(page,Math.max(0,Math.ceil(filtered.length/pageSize)-1));
        const slice=filtered.slice(page*pageSize,(page+1)*pageSize);
        slice.forEach(n=>included.add(n.id));
        if(compound)included.add(compound.id);
        edges.forEach(e=>{if(included.has(e.source)&&byId.get(e.target)?.type==='source')included.add(e.target)});
      }
      for(const n of nodes) {
        if(mode==='evidence'&&!included.has(n.id))continue;
        const id=mode==='overview'?groupId(n):n.id;
        alias.set(n.id,id);
        if(id===n.id)result.set(id,{...n,members:[n]});
        else {
          if(!result.has(id))result.set(id,{id,type:n.type,group:true,label:n.type==='evidence'?(categories[n.properties?.evidence_type]||n.properties?.evidence_type):kinds[n.type],members:[],category:n.properties?.evidence_type});
          result.get(id).members.push(n);
        }
      }
      const groupedEdges=new Map();
      edges.forEach(e=>{
        const source=alias.get(e.source),target=alias.get(e.target);
        if(!source||!target||source===target)return;
        const key=JSON.stringify([source,target,e.type]);
        if(!groupedEdges.has(key))groupedEdges.set(key,{source,target,type:e.type,count:0});
        groupedEdges.get(key).count++;
      });
      drawn=[...result.values()];drawnEdges=[...groupedEdges.values()];
      const middle=drawn.filter(n=>!['compound','source'].includes(n.type));
      const sources=drawn.filter(n=>n.type==='source');
      const rank={evidence:0,model:1,protein_target:2};
      middle.sort((a,b)=>(rank[a.type]??3)-(rank[b.type]??3)||String(a.label).localeCompare(String(b.label),'zh-CN'));
      // Place sources near the average row of their connected categories to reduce crossings.
      const order=new Map(middle.map((n,i)=>[n.id,i]));
      const center=n=>{const rows=drawnEdges.filter(e=>e.target===n.id&&order.has(e.source)).map(e=>order.get(e.source));return rows.length?rows.reduce((a,b)=>a+b,0)/rows.length:0};
      sources.sort((a,b)=>center(a)-center(b)||String(a.label).localeCompare(String(b.label)));
      const height=Math.max(560,104+Math.max(middle.length,sources.length)*84);
      const position=(rows,x,w)=>rows.forEach((n,i)=>Object.assign(n,{x,y:height/2+(i-(rows.length-1)/2)*84,w,h:68}));
      position(middle,342,244);position(sources,764,212);
      position(drawn.filter(n=>n.type==='compound'),22,200);
      view={width:1000,height};
      return {filtered};
    }
    function fitLabels() {
      if(signal.aborted||!svg.isConnected)return;
      const context=document.createElement('canvas').getContext('2d');
      for(const el of svg.querySelectorAll('.graph-node')){
        const n=drawn.find(x=>x.id===el.dataset.node), name=el.querySelector('.node-name');
        const style=getComputedStyle(name);context.font=`${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
        const max=n.w-44, all=[];let line='';
        for(const char of String(titleOf(n))){
          if(line&&context.measureText(line+char).width>max){all.push(line);line=''}
          line+=char;
        }
        if(line)all.push(line);
        const rows=all.slice(0,2);
        if(all.length>2){while(context.measureText(rows[1]+'…').width>max)rows[1]=rows[1].slice(0,-1);rows[1]+='…'}
        name.replaceChildren();name.setAttribute('y',rows.length>1?'23':'29');
        rows.forEach((row,i)=>{const span=document.createElementNS('http://www.w3.org/2000/svg','tspan');span.setAttribute('x','30');span.setAttribute('dy',i?'17':'0');span.textContent=row;name.append(span)});
        // Source names under evidence cards can be longer than the parameter name.
        const meta=el.querySelector('.node-meta');meta.setAttribute('y',rows.length>1?'60':'55');const metaStyle=getComputedStyle(meta);context.font=`${metaStyle.fontWeight} ${metaStyle.fontSize} ${metaStyle.fontFamily}`;
        let subtitle=meta.dataset.full||meta.textContent;meta.dataset.full=subtitle;
        if(context.measureText(subtitle).width>max){while(subtitle&&context.measureText(subtitle+'…').width>max)subtitle=subtitle.slice(0,-1);subtitle+='…'}
        meta.textContent=subtitle;
        for(const line of [...name.querySelectorAll('tspan'),meta]){
          if(line.getComputedTextLength()<=max)continue;
          let value=line.textContent.replace(/…$/,'');
          while(value&&line.getComputedTextLength()>max){value=value.slice(0,-1);line.textContent=value+'…'}
        }
      }
    }
    function draw() {
      revision++;
      const {filtered}=build();
      const positions=new Map(drawn.map(n=>[n.id,n]));
      const columnY=24;
      svg.innerHTML=`<defs><marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#a5b8b2" stroke-width="1.5"/></marker></defs>
        <g class="graph-column-labels"><text x="22" y="${columnY}">研究化合物</text><text x="342" y="${columnY}">${mode==='overview'?'证据与模型':'证据条目'}</text><text x="764" y="${columnY}">数据来源</text></g>
        <g class="graph-edges">${drawnEdges.map((e,i)=>{const a=positions.get(e.source),b=positions.get(e.target);let path;
          if(a.x===b.x){const x=a.x+a.w;path=`M${x},${a.y} C${x+75},${a.y} ${x+75},${b.y} ${x},${b.y}`}
          else {const forward=a.x<b.x;const x1=forward?a.x+a.w:a.x,x2=forward?b.x:b.x+b.w;const mid=(x1+x2)/2;path=`M${x1},${a.y} C${mid},${a.y} ${mid},${b.y} ${x2},${b.y}`}
          return `<path class="graph-edge ${e.type==='not_substrate_of'?'negative':''}" data-edge="${i}" d="${path}" marker-end="url(#graph-arrow)"><title>${escape(relations[e.type]||e.type)} · ${e.count} 条关联记录</title></path>`}).join('')}</g>
        <g class="graph-nodes">${drawn.map(n=>{
          const title=[titleOf(n)];
          const lower=n.group?describe(n):n.type==='evidence'?(n.properties?.source_name||n.properties?.source_id||categories[n.properties?.evidence_type]):kinds[n.type];
          return `<g class="graph-node ${n.type}" data-node="${escape(n.id)}" transform="translate(${n.x},${n.y-n.h/2})" role="button" tabindex="0" aria-label="${escape(titleOf(n)+'，'+describe(n))}" aria-pressed="false"><title>${escape(titleOf(n)+' · '+describe(n))}</title><rect class="node-card" width="${n.w}" height="${n.h}" rx="12"/><circle cx="17" cy="25" r="4"/><text class="node-name" x="30" y="${title.length>1?18:23}">${title.map((line,i)=>`<tspan x="30" dy="${i?17:0}">${escape(line)}</tspan>`).join('')}</text><text class="node-meta" x="30" y="51">${escape(lower)}</text></g>`}).join('')}</g>`;
      fitLabels();document.fonts.ready.then(fitLabels);
      root.querySelectorAll('.graph-mode button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.graphAction===mode?'true':'false'));
      root.querySelector('.graph-evidence-controls').hidden=mode!=='evidence';
      select.value=filter;
      root.querySelector('.graph-page-status').textContent=filtered.length?`${page*pageSize+1}–${Math.min((page+1)*pageSize,filtered.length)} / ${filtered.length} 条`:'暂无证据';
      root.querySelector('[data-graph-action="prev"]').disabled=page===0;
      root.querySelector('[data-graph-action="next"]').disabled=(page+1)*pageSize>=filtered.length;
      root.querySelector('.graph-caption').textContent=mode==='overview'?`${evidence.length} 条证据 · ${nodes.filter(n=>n.type==='model').length} 个登记模型 · ${nodes.filter(n=>n.type==='protein_target').length} 个酶/转运体。分类节点包含全部关联记录，点击可展开。`:`当前展示 ${drawn.filter(n=>n.type==='evidence').length} / ${filtered.length} 条证据；可翻页查看，下载保留完整图谱。`;
      reset();
      if(!drawn.some(n=>n.id===selected))selected='';
      const selection=selected;
      inspect(drawn.find(n=>n.id===selected)||drawn.find(n=>n.type==='compound'));
      if(!selection){selected='';highlight('')}
    }
    function transform() {
      const w=view.width/zoom,h=view.height/zoom;
      svg.setAttribute('viewBox',`${(view.width-w)/2-pan.x} ${(view.height-h)/2-pan.y} ${w} ${h}`);
      root.querySelector('.graph-zoom').textContent=Math.round(zoom*100)+'%';
    }
    function reset(){zoom=1;pan={x:0,y:0};transform()}
    function changeZoom(factor){zoom=Math.max(.65,Math.min(3.5,zoom*factor));transform()}
    function highlight(id) {
      const focus=alias.get(id)||id;
      const adjacent=new Set([focus]);
      drawnEdges.forEach(e=>{if(e.source===focus||e.target===focus){adjacent.add(e.source);adjacent.add(e.target)}});
      svg.querySelectorAll('.graph-node').forEach(el=>{el.classList.toggle('is-selected',el.dataset.node===focus);el.classList.toggle('is-muted',Boolean(id)&&!adjacent.has(el.dataset.node));el.setAttribute('aria-pressed',el.dataset.node===focus?'true':'false')});
      svg.querySelectorAll('.graph-edge').forEach(el=>{const e=drawnEdges[Number(el.dataset.edge)],active=e.source===focus||e.target===focus;el.classList.toggle('is-active',Boolean(id)&&active);el.classList.toggle('is-muted',Boolean(id)&&!active)});
    }
    function records(items) {
      return `<div class="graph-records">${items.map(n=>`<button data-graph-node="${escape(n.id)}"><span>${escape(titleOf(n))}</span><small>${escape(n.properties?.source_name||n.properties?.source_id||kinds[n.type])}</small><span aria-hidden="true">↗</span></button>`).join('')}</div>`;
    }
    async function inspect(n) {
      if(!n){inspector.innerHTML='<p>暂无关联节点。</p>';return}
      selected=n.id;highlight(selected);
      const ticket=++revision;
      const ids=memberIds(n), connections=edges.filter(e=>ids.has(e.source)||ids.has(e.target));
      const rels=new Map();connections.forEach(e=>rels.set(e.type,(rels.get(e.type)||0)+1));
      let html=`<p class="eyebrow">${escape(kinds[n.type]||'节点详情')}</p><h2>${escape(titleOf(n))}</h2><p class="graph-description">${escape(describe(n))}</p>`;
      if(n.group){
        html+=n.type==='evidence'?`<button class="primary graph-expand" data-graph-action="expand" data-category="${escape(n.category)}">展开 ${n.members.length} 条证据 →</button>`:'';
        html+=`<h3>${n.type==='evidence'?'证据预览':'包含条目'}</h3>`+records(n.members.slice(0,n.type==='evidence'?6:100));
        if(n.type==='evidence'&&n.members.length>6)html+='<p class="small muted">展开分类可查看全部条目。</p>';
      } else if(n.type==='compound'){
        html+=`<div class="graph-counts"><div><strong>${evidence.length}</strong><span>证据</span></div><div><strong>${nodes.filter(x=>x.type==='source').length}</strong><span>来源</span></div><div><strong>${nodes.filter(x=>x.type==='model').length}</strong><span>模型</span></div></div><p class="small muted">点击分类，查看相关参数与原文；点击模型或酶与转运体，查看具体条目。</p>`;
      } else if(n.type==='evidence'){
        const p=n.properties||{};
        html+=`<p>${escape(categories[p.evidence_type]||p.evidence_type)}</p><p class="small muted">${escape(p.source_name||p.source_id)}</p>${sourceLink(n)}<div class="graph-evidence-detail"><p class="small muted">正在读取证据…</p></div>`;
      } else if(n.type==='source') {
        const linked=nodes.filter(x=>connections.some(e=>(e.source===x.id||e.target===x.id)&&x.id!==n.id));
        html+=`<p class="small muted">当前化合物有 ${linked.filter(x=>x.type==='evidence').length} 条证据关联到此数据源。具体文献链接见单条证据。</p>`+records(linked.filter(x=>mode==='overview'||alias.has(x.id)).slice(0,8));
      } else {
        const p=n.properties||{};
        const fields=n.type==='model'?{'软件':p.software,'验证状态':p.validation_status}:{'基因符号':p.gene_symbol,'类型':p.target_type};
        html+=`<dl>${Object.entries(fields).filter(([,v])=>v).map(([k,v])=>`<dt>${escape(k)}</dt><dd>${escape(v)}</dd>`).join('')}</dl>`;
      }
      if(rels.size)html+=`<h3 class="graph-relations-title">关联关系</h3><div class="graph-relation-list">${[...rels].map(([type,count])=>`<div><span>${escape(relations[type]||type)}</span><small>${count}</small></div>`).join('')}</div>`;
      inspector.innerHTML=html;
      if(n.type==='evidence'&&!n.group){
        try {
          const p=n.properties;
          let record=evidenceCache.get(n.id);
          if(!record){record=await loadEvidence(p.evidence_type,p.evidence_id);evidenceCache.set(n.id,record)}
          if(signal.aborted||ticket!==revision)return;
          const s=record.summary||{}, f=record.extracted_fields||{};
          const value=s.numeric_value??s.value_text;
          inspector.querySelector('.graph-evidence-detail').innerHTML=(value!=null?`<p class="graph-value">${escape(value)} <small>${escape(s.unit||'')}</small></p>`:'')+`<p class="small muted">${escape(s.context||'')}</p><p class="graph-excerpt">${escape(s.evidence_text||f.source_text||'暂未记录原文摘录。')}</p>`;
        } catch(error){if(!signal.aborted&&ticket===revision)inspector.querySelector('.graph-evidence-detail').textContent='证据详情暂时无法读取，请通过来源链接查看原文。'}
      }
    }
    root.addEventListener('click',event=>{
      if(moved){moved=false;return}
      const node=event.target.closest('[data-node],[data-graph-node]');
      if(node){const id=node.dataset.node||node.dataset.graphNode;inspect(drawn.find(n=>n.id===id)||byId.get(id));return}
      const button=event.target.closest('[data-graph-action]');if(!button)return;
      const action=button.dataset.graphAction;
      if(action==='in')changeZoom(1.25);
      else if(action==='out')changeZoom(.8);
      else if(action==='reset')reset();
      else {if(action==='overview'||action==='evidence'){mode=action;filter='';page=0}else if(action==='expand'){mode='evidence';filter=button.dataset.category;page=0}else if(action==='prev')page--;else if(action==='next')page++;selected='';draw()}
    },{signal});
    select.addEventListener('change',()=>{filter=select.value;page=0;selected='';draw()},{signal});
    svg.addEventListener('keydown',event=>{
      const node=event.target.closest('[data-node]');
      if(node&&['Enter',' '].includes(event.key)){event.preventDefault();inspect(drawn.find(n=>n.id===node.dataset.node))}
      else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')reset();else changeZoom(event.key==='-'?.8:1.25)}
    },{signal});
    svg.addEventListener('wheel',event=>{event.preventDefault();changeZoom(event.deltaY<0?1.12:1/1.12)},{passive:false,signal});
    const point=event=>{const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;return p.matrixTransform(svg.getScreenCTM().inverse())};
    svg.addEventListener('pointerdown',event=>{if(event.button!==0||event.target.closest('[data-node]'))return;const p=point(event);drag={id:event.pointerId,x:event.clientX,y:event.clientY,px:p.x,py:p.y,pan:{...pan}};moved=false;svg.setPointerCapture(event.pointerId)},{signal});
    svg.addEventListener('pointermove',event=>{if(!drag||drag.id!==event.pointerId)return;if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>4)moved=true;if(moved){const p=point(event);pan.x+=p.x-drag.px;pan.y+=p.y-drag.py;transform();svg.classList.add('is-dragging')}},{signal});
    const stop=event=>{if(!drag||drag.id!==event.pointerId)return;drag=null;svg.classList.remove('is-dragging');if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);setTimeout(()=>{moved=false},0)};
    svg.addEventListener('pointerup',stop,{signal});svg.addEventListener('pointercancel',stop,{signal});
    draw();
    return ()=>{revision++;abort.abort()};
  }
  return {markup,mount};
})();
