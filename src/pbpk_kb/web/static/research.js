/* Model recommendation requests and qualification results. */
'use strict';
window.PBPKResearch = (() => {
  let job=null,result=null,gateResult=null,timer=null,container=null,disposed=false,generation=0,request={compound:'midazolam',objective:'base_pk',route:'oral',species:'human'};
  const format=v=>v==null?'—':typeof v==='number'?Number(v.toPrecision(4)).toLocaleString('en-US'):String(v);
  function gapText(text){const fields={route:'给药途径',dose:'剂量',dose_unit:'剂量单位',frequency:'给药频次'};if(text.startsWith('Dose regimen is incomplete: missing '))return '给药方案待补充：'+text.replace('Dose regimen is incomplete: missing ','').replace(/\.$/,'').split(', ').map(x=>fields[x]||x).join('、');return {'Target population definition is missing.':'请补充目标人群','DDI request requires an interacting drug or explicit victim/perpetrator.':'请补充相互作用药物'}[text]||text}
  const objectives=[['base_pk','基础药代模型'],['public_model_reuse','已有模型复用']];
  const endpoints=[['AUC','AUC'],['Cmax','Cmax'],['concentration_time_profile','浓度曲线']];
  const extraFields=['population','dose','dose_unit','single_or_multiple','frequency','formulation','enzymes','transporters'];
  const representedFields=new Set(['compound','objective','route','species','prediction_endpoints','population','dose','dose_unit','single_or_multiple','frequency','formulation','required_enzymes','required_transporters','top_k_subgraphs','max_evidence_per_component']);
  function hasLegacyConditions(){const omitted=Object.entries(request).some(([k,v])=>!representedFields.has(k)&&v!=null&&(Array.isArray(v)?v.length>0:String(v).trim()!==''));return omitted||(request.objective&&!objectives.some(([v])=>v===request.objective))||(request.species&&request.species!=='human')||(request.prediction_endpoints||[]).some(v=>!endpoints.some(([e])=>e===v))||(request.route&&!['oral','intravenous'].includes(request.route))||(request.dose_unit&&!['mg','mg/kg','µg'].includes(request.dose_unit))}
  function extraCount(values){return extraFields.filter(k=>(k!=='dose_unit'||String(values.dose??'').trim())&&String(values[k]??'').trim()).length}
  function extraSummary(n){return n?`更多筛选条件（已填写 ${n} 项）`:'更多筛选条件（可选）'}
  function refreshExtra(form){const values=Object.fromEntries(new FormData(form));const n=extraCount(values);form.querySelector('[data-extra-summary]').textContent=extraSummary(n);form.querySelector('[data-research-action="clear-extra"]').hidden=!n}
  function form(all){const n=extraCount({...request,enzymes:(request.required_enzymes||[]).join(', '),transporters:(request.required_transporters||[]).join(', ')});return `<form id="research-recommend-form" class="panel"><div class="panel-heading"><h2>你的建模需求</h2><span class="badge">人体公共模型筛选</span></div>
    <p class="research-form-intro">用于查找和检查公共模型；实际浓度预测请进入<a href="#blueprints">蓝图设计</a>。</p>
    ${hasLegacyConditions()?'<p class="note warn" data-legacy-request>旧请求包含当前页面未提供的条件；新请求请按下方条件重新确认。已有结果仍按原请求展示。</p>':''}
    <div class="query-grid">
    <label>化合物<select name="compound" required>${options(all.map(c=>[c.preferred_name,c.preferred_name]),request.compound,'请选择')}</select></label>
    <label>研究目标<select name="objective" required>${options(objectives,request.objective,'请选择')}</select></label>
    <label>给药途径<select name="route">${options([['oral','口服'],['intravenous','静脉']],request.route,'不限')}</select></label></div>
    <fieldset class="research-endpoints"><legend>关注指标（可选）</legend><div>${endpoints.map(([v,t])=>`<label><input type="checkbox" name="prediction_endpoints" value="${v}" ${(request.prediction_endpoints||[]).includes(v)?'checked':''}><span>${t}</span></label>`).join('')}</div></fieldset>
    <div class="research-extra-controls"><details class="research-extra"><summary data-extra-summary>${extraSummary(n)}</summary><p class="small muted">按需填写；这些条件会用于检查模型能否直接复用。留空不会自动补入假定值。</p><div class="query-grid">
    <label>目标人群<input name="population" value="${esc(request.population||'')}" placeholder="例如 healthy adult"></label>
    <label>剂量（可选）<input name="dose" type="number" min="0.000001" step="any" value="${esc(request.dose??'')}" placeholder="例如 5"></label>
    <label>剂量单位<select name="dose_unit">${options([['mg','mg'],['mg/kg','mg/kg'],['µg','µg']],request.dose_unit||'','未指定')}</select></label>
    <label>给药方式<select name="single_or_multiple">${options([['single','单次给药'],['multiple','多次给药']],request.single_or_multiple||'','未指定')}</select></label>
    <label>给药频次<input name="frequency" value="${esc(request.frequency||'')}" placeholder="例如 once / once daily"></label>
    <label>剂型<input name="formulation" value="${esc(request.formulation||'')}" placeholder="例如 tablet / solution"></label>
    <label>所需酶<input name="enzymes" value="${esc((request.required_enzymes||[]).join(', '))}" placeholder="例如 CYP3A4"></label>
    <label>所需转运体<input name="transporters" value="${esc((request.required_transporters||[]).join(', '))}" placeholder="例如 P-gp"></label>
    </div></details><button type="button" class="text-button research-clear-extra" data-research-action="clear-extra" ${n?'':'hidden'}>清空额外条件</button></div>
    <div class="inline-actions research-form-actions"><button class="primary" type="submit">推荐模型 →</button><button type="button" data-research-action="gate">仅检查用途与适用条件</button></div><p class="small muted">结合请求与模型资料排序，并展示适用条件检查。推荐分数不等于临床预测准确率。</p></form>`}
  async function renderRecommend(){
    const all=(await compounds()).filter(c=>c.evidence_count);
    return head('推荐工作台','找到相关公共模型，核对用途与适用范围，再进入蓝图设计。')+`<div class="tabs"><button data-action="tab-rank" class="selected">模型推荐</button><button data-action="tab-components">组件推荐</button></div>`+form(all)+`<div id="research-result">${gateResult?rankingResult(gateResult):result?renderResult(result):empty('填写建模需求，开始模型推荐。')}</div>`;
  }
  function renderResult(r){
    return `<div class="result-title"><h2>候选建议</h2>${badge(r.method||'PBPK-Match light','neutral')}<button data-research-action="download">下载结果 ↓</button></div><p class="research-summary">${esc(r.request?.compound||request.compound)} · ${esc(label(r.request?.objective))} · ${esc(scenarioName(r.request?.route)||'未指定途径')}${r.request?.dose!=null?' · '+esc(r.request.dose)+' '+esc(r.request.dose_unit||''):''} · 本次比较 ${fmt(r.candidate_count)} 个候选版本 · 点击模型查看资产和验证信息</p>
    ${r.display_notice?`<p class="small muted">${esc(r.display_notice)}</p>`:''}
    ${(r.request_gaps||[]).length?`<div class="note warn">如需判断能否直接复用，还需补充：${esc((r.request_gaps||[]).map(gapText).join('；'))}</div>`:''}
    <div class="inline-actions" style="margin-bottom:20px"><button data-research-action="blueprint">设计 ${esc(r.request?.compound||request.compound)} 的蓝图 →</button></div>
    ${(r.top_candidates||[]).map((c,i)=>{const gate=c.gate||{},prob=c.probabilities||{};const category=Object.entries(prob).sort((a,b)=>b[1]-a[1])[0]?.[0];const judgment={Y:'文档支持',N:'文档不支持',U:'文档依据不足'}[category];return `<article class="result-candidate"><header><span class="rank-number">${c.rank||i+1}</span><div class="candidate-main"><h3>${esc(c.name)}</h3><p>${esc(list(c.compounds))}</p><div class="tags">${judgment?badge(judgment,category==='Y'?'':'warn'):''}${(c.decision_label||gate.candidate_layer)?badge(c.decision_label||label(gate.candidate_layer),'neutral'):''}</div></div><div class="score">${format(c.score)}<small>相关性分数</small></div></header>
    ${(gate.hard_blockers||[]).length?`<p>需要核对：${esc((gate.hard_blockers||[]).slice(0,3).map(b=>label(b.reason_code||b.message||b)).join('；'))}</p>`:''}
    <div class="candidate-footer"><span class="small muted">${category?'文档判断概率 '+format(prob[category]):'请核对完整模型说明'}</span>${c.model_version_id?`<button data-action="model-detail" data-id="${esc(c.model_version_id)}">查看模型详情 ↗</button>`:''}</div></article>`}).join('')||empty('当前没有通过用途分层的候选建议；可查看下方排除原因或调整需求。')}
    ${(r.reference_candidates||[]).length?`<section class="research-card"><h3>跨药物机制参考</h3><p class="small muted">用于参考机制和结构，不能直接当作目标药物模型。</p>${candidateTable(r.reference_candidates)}</section>`:''}
    ${(r.blocked_candidates||[]).length?`<details><summary>当前不适用的候选 · ${r.blocked_candidates.length}</summary>${candidateTable(r.blocked_candidates)}</details>`:''}
    ${r.unassessed_count?`<p class="small muted">另有 ${fmt(r.unassessed_count)} 个候选尚未确认用途，未列入建议。完整研究排名可下载。</p>`:''}
    ${(r.limitations||[]).length?`<details><summary>方法与适用范围</summary><ul class="small">${r.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></details>`:''}`;
  }
  function candidateTable(rows){return table(['模型','用途结论','原因',''],rows.map(c=>`<tr><td>${esc(c.name)}</td><td>${esc(c.decision_label||label(c.gate?.candidate_layer))}</td><td>${esc((c.reason_codes||[]).map(label).join('；')||'—')}</td><td>${c.model_version_id?`<button data-action="model-detail" data-id="${esc(c.model_version_id)}">查看详情</button>`:''}</td></tr>`))}
  function progress(row){return `<div class="job-progress" role="status"><h3>${row.status==='queued'?'请求已排队':'正在匹配模型'}</h3><p>${esc(row.message||row.stage||'正在读取请求与候选模型资料')}</p><div class="progress-track"><i></i></div><p>可以切换页面；返回后会继续显示此请求的进度。</p></div>`}
  function payload(form){const fd=new FormData(form);const out={species:'human'};for(const key of ['compound','objective','route','population','dose_unit','single_or_multiple','frequency','formulation']){const value=String(fd.get(key)||'').trim();if(value)out[key]=value}const dose=String(fd.get('dose')||'').trim();if(dose)out.dose=Number(dose);else delete out.dose_unit;for(const [key,field] of [['enzymes','required_enzymes'],['transporters','required_transporters']]){const values=String(fd.get(key)||'').split(/[,，;；]/).map(x=>x.trim()).filter(Boolean);if(values.length)out[field]=values}out.prediction_endpoints=fd.getAll('prediction_endpoints').filter(v=>endpoints.some(([e])=>e===v));return out}
  function validateForm(form){const invalid=form.querySelector('.research-extra :invalid');if(invalid)form.querySelector('.research-extra').open=true;return form.reportValidity()}
  async function poll(ticket=generation){
    clearTimeout(timer);if(disposed||ticket!==generation||!job)return;const id=job.job_id;
    try{const row=await api('/workspace/research/jobs/'+encodeURIComponent(id));if(disposed||ticket!==generation||job?.job_id!==id)return;job=row;
      if(['complete','completed'].includes(row.status)){gateResult=null;result=row.result;request=result.request||request;const old=container.querySelector('#research-recommend-form');if(old){const all=state.compounds.filter(c=>c.evidence_count);old.outerHTML=form(all)}sessionStorage.removeItem('pbpk_model_job');container.querySelector('#research-result').innerHTML=renderResult(result);job=null;return}
      if(['failed','cancelled','interrupted'].includes(row.status)){sessionStorage.removeItem('pbpk_model_job');container.querySelector('#research-result').innerHTML=`<div class="note warn"><strong>此次推荐未完成</strong><p>${esc(typeof row.error==='string'?row.error:row.message||row.error?.message||'请调整输入后重新提交。')}</p></div>`;job=null;return}
      container.querySelector('#research-result').innerHTML=progress(row);timer=setTimeout(()=>poll(ticket),1600);
    }catch(error){if(!disposed&&ticket===generation){container.querySelector('#research-result').innerHTML=`<div class="note warn">${esc(error.message)}<button data-research-action="resume">重新连接任务</button></div>`}}
  }
  function mountRecommend(host){
    container=host;disposed=false;const ticket=++generation;const current=()=>!disposed&&ticket===generation;const controller=new AbortController();
    host.addEventListener('input',e=>{if(e.target.closest('.research-extra'))refreshExtra(e.target.closest('form'))},{signal:controller.signal});
    host.addEventListener('change',e=>{if(e.target.closest('.research-extra'))refreshExtra(e.target.closest('form'))},{signal:controller.signal});
    host.addEventListener('invalid',e=>{const details=e.target.closest('.research-extra');if(details)details.open=true}, {capture:true,signal:controller.signal});
    const remembered=sessionStorage.getItem('pbpk_model_job');if(remembered){job={job_id:remembered,status:'running'};container.querySelector('#research-result').innerHTML=progress(job);poll(ticket)}
    else if(!result&&!gateResult){api('/workspace/research/jobs/current').then(row=>{if(current()&&row?.job_id&&!job){job=row;sessionStorage.setItem('pbpk_model_job',row.job_id);host.querySelector('#research-result').innerHTML=progress(row);poll(ticket)}}).catch(()=>{})}
    host.addEventListener('submit',async e=>{if(e.target.id!=='research-recommend-form')return;e.preventDefault();e.stopPropagation();if(!validateForm(e.target))return;if(job){toast('当前推荐仍在运行，完成后可提交新的请求。');return}const button=e.target.querySelector('[type="submit"]');button.disabled=true;
      try{request=payload(e.target);result=null;gateResult=null;const row=await post('/workspace/research/recommend',request);job=row;sessionStorage.setItem('pbpk_model_job',row.job_id);if(current()){host.querySelector('#research-result').innerHTML=progress(row);poll(ticket)}}
      catch(error){if(error.status===409&&error.detail?.job_id){job=error.detail;sessionStorage.setItem('pbpk_model_job',job.job_id);toast('已有推荐正在计算，已恢复任务进度。');if(current())poll(ticket)}else if(current())host.querySelector('#research-result').innerHTML=`<div class="note warn">${esc(error.message)}</div>`}finally{button.disabled=false}}, {signal:controller.signal});
    host.addEventListener('click',async e=>{const b=e.target.closest('[data-research-action]');if(!b)return;const action=b.dataset.researchAction;
      if(action==='clear-extra'){const form=b.closest('form');request={...request};for(const key of extraFields){form.elements.namedItem(key).value='';delete request[key]}delete request.required_enzymes;delete request.required_transporters;refreshExtra(form)}
      else if(action==='download')download(result,'pbpk-model-recommendation.json');
      else if(action==='resume')poll(ticket);
      else if(action==='blueprint'){sessionStorage.setItem('pbpk_blueprint_drug',result?.request?.compound||request.compound);location.hash='blueprints'}
      else if(action==='gate'){const form=host.querySelector('#research-recommend-form');if(!validateForm(form))return;if(job){toast('当前推荐正在计算，请完成后再检查其他需求。');return}b.disabled=true;try{request=payload(form);result=null;gateResult=null;const r=await post('/recommendation/public-model-ranking',request);state.rec=r;gateResult=r;if(current())host.querySelector('#research-result').innerHTML=rankingResult(r)}catch(error){toast(error.message)}finally{b.disabled=false}}
    },{signal:controller.signal});
    return ()=>{if(ticket===generation){disposed=true;++generation;clearTimeout(timer)}controller.abort()};
  }
  return {renderRecommend,mountRecommend};
})();
