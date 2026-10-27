/* Model recommendation requests and qualification results. */
'use strict';
window.PBPKResearch = (() => {
  let job=null,result=null,gateResult=null,timer=null,container=null,disposed=false,generation=0,request={compound:'midazolam',objective:'base_pk',route:'oral',species:'human'};
  const format=v=>v==null?'—':typeof v==='number'?Number(v.toPrecision(4)).toLocaleString('en-US'):String(v);
  function gapText(text){const fields={route:'Route',dose:'Dose',dose_unit:'Dose unit',frequency:'Dosing frequency'};if(text.startsWith('Dose regimen is incomplete: missing '))return 'Dosing details needed: '+text.replace('Dose regimen is incomplete: missing ','').replace(/\.$/,'').split(', ').map(x=>fields[x]||x).join(', ');return {'Target population definition is missing.':'Specify the target population','DDI request requires an interacting drug or explicit victim/perpetrator.':'Specify the interacting drug'}[text]||text}
  const objectives=[['base_pk','Basic pharmacokinetic model'],['public_model_reuse','Public model reuse']];
  const endpoints=[['AUC','AUC'],['Cmax','Cmax'],['concentration_time_profile','Concentration profile']];
  const extraFields=['population','dose','dose_unit','single_or_multiple','frequency','formulation','enzymes','transporters'];
  const representedFields=new Set(['compound','objective','route','species','prediction_endpoints','population','dose','dose_unit','single_or_multiple','frequency','formulation','required_enzymes','required_transporters','top_k_subgraphs','max_evidence_per_component']);
  function hasLegacyConditions(){const omitted=Object.entries(request).some(([k,v])=>!representedFields.has(k)&&v!=null&&(Array.isArray(v)?v.length>0:String(v).trim()!==''));return omitted||(request.objective&&!objectives.some(([v])=>v===request.objective))||(request.species&&request.species!=='human')||(request.prediction_endpoints||[]).some(v=>!endpoints.some(([e])=>e===v))||(request.route&&!['oral','intravenous'].includes(request.route))||(request.dose_unit&&!['mg','mg/kg','µg'].includes(request.dose_unit))}
  function extraCount(values){return extraFields.filter(k=>(k!=='dose_unit'||String(values.dose??'').trim())&&String(values[k]??'').trim()).length}
  function extraSummary(n){return n?`Additional filters (${n} entered)`:'Additional filters (optional)'}
  function refreshExtra(form){const values=Object.fromEntries(new FormData(form));const n=extraCount(values);form.querySelector('[data-extra-summary]').textContent=extraSummary(n);form.querySelector('[data-research-action="clear-extra"]').hidden=!n}
  function form(all){const n=extraCount({...request,enzymes:(request.required_enzymes||[]).join(', '),transporters:(request.required_transporters||[]).join(', ')});return `<form id="research-recommend-form" class="panel"><div class="panel-heading"><h2>Your modeling request</h2><span class="badge">Screen public human models</span></div>
    <p class="research-form-intro">Find and assess public models. For concentration predictions, use <a href="#blueprints">Blueprint design</a>.</p>
    ${hasLegacyConditions()?'<p class="note warn" data-legacy-request>The saved request includes conditions not supported by this form. Confirm the fields below for a new request; saved results retain their original conditions.</p>':''}
    <div class="query-grid">
    <label>Compound<select name="compound" required>${options(all.map(c=>[c.preferred_name,c.preferred_name]),request.compound,'Select an option')}</select></label>
    <label>Study objective<select name="objective" required>${options(objectives,request.objective,'Select an option')}</select></label>
    <label>Route<select name="route">${options([['oral','Oral'],['intravenous','Intravenous']],request.route,'Any')}</select></label></div>
    <fieldset class="research-endpoints"><legend>Endpoints of interest (optional)</legend><div>${endpoints.map(([v,t])=>`<label><input type="checkbox" name="prediction_endpoints" value="${v}" ${(request.prediction_endpoints||[]).includes(v)?'checked':''}><span>${t}</span></label>`).join('')}</div></fieldset>
    <div class="research-extra-controls"><details class="research-extra"><summary data-extra-summary>${extraSummary(n)}</summary><p class="small muted">Optional conditions inform direct-reuse checks. Blank fields are not filled with assumed values.</p><div class="query-grid">
    <label>Target population<input name="population" value="${esc(request.population||'')}" placeholder="e.g. healthy adult"></label>
    <label>Dose (optional)<input name="dose" type="number" min="0.000001" step="any" value="${esc(request.dose??'')}" placeholder="e.g. 5"></label>
    <label>Dose unit<select name="dose_unit">${options([['mg','mg'],['mg/kg','mg/kg'],['µg','µg']],request.dose_unit||'','Unspecified')}</select></label>
    <label>Dosing schedule<select name="single_or_multiple">${options([['single','Single dose'],['multiple','Multiple doses']],request.single_or_multiple||'','Unspecified')}</select></label>
    <label>Dosing frequency<input name="frequency" value="${esc(request.frequency||'')}" placeholder="e.g. once / once daily"></label>
    <label>Formulation<input name="formulation" value="${esc(request.formulation||'')}" placeholder="e.g. tablet / solution"></label>
    <label>Required enzymes<input name="enzymes" value="${esc((request.required_enzymes||[]).join(', '))}" placeholder="e.g. CYP3A4"></label>
    <label>Required transporters<input name="transporters" value="${esc((request.required_transporters||[]).join(', '))}" placeholder="e.g. P-gp"></label>
    </div></details><button type="button" class="text-button research-clear-extra" data-research-action="clear-extra" ${n?'':'hidden'}>Clear additional filters</button></div>
    <div class="inline-actions research-form-actions"><button class="primary" type="submit">Recommend models →</button><button type="button" data-research-action="gate">Check applicability only</button></div><p class="small muted">Candidates are ranked using the request and model documentation, with separate applicability checks. Recommendation scores are not clinical prediction accuracy.</p></form>`}
  async function renderRecommend(){
    const all=(await compounds()).filter(c=>c.evidence_count);
    return head('Recommendations','Find relevant public models, check their intended use and scope, and proceed to blueprint design.')+`<div class="tabs"><button data-action="tab-rank" class="selected">Model recommendations</button><button data-action="tab-components">Component recommendations</button></div>`+form(all)+`<div id="research-result">${gateResult?rankingResult(gateResult):result?renderResult(result):empty('Enter a modeling request to view model recommendations.')}</div>`;
  }
  function renderResult(r){
    return `<div class="result-title"><h2>Candidate suggestions</h2>${badge(r.method||'PBPK-Match light','neutral')}<button data-research-action="download">Download results ↓</button></div><p class="research-summary">${esc(r.request?.compound||request.compound)} · ${esc(label(r.request?.objective))} · ${esc(scenarioName(r.request?.route)||'Route unspecified')}${r.request?.dose!=null?' · '+esc(r.request.dose)+' '+esc(r.request.dose_unit||''):''} · Compared ${fmt(r.candidate_count)} candidate versions · Select a model to inspect its assets and validation</p>
    ${r.display_notice?`<p class="small muted">${esc(r.display_notice)}</p>`:''}
    ${(r.request_gaps||[]).length?`<div class="note warn">Details needed to assess direct reuse: ${esc((r.request_gaps||[]).map(gapText).join('; '))}</div>`:''}
    <div class="inline-actions" style="margin-bottom:20px"><button data-research-action="blueprint">Design a blueprint for ${esc(r.request?.compound||request.compound)} →</button></div>
    ${(r.top_candidates||[]).map((c,i)=>{const gate=c.gate||{},prob=c.probabilities||{};const category=Object.entries(prob).sort((a,b)=>b[1]-a[1])[0]?.[0];const judgment={Y:'Supported by documentation',N:'Not supported by documentation',U:'Insufficient documentation'}[category];return `<article class="result-candidate"><header><span class="rank-number">${c.rank||i+1}</span><div class="candidate-main"><h3>${esc(c.name)}</h3><p>${esc(list(c.compounds))}</p><div class="tags">${judgment?badge(judgment,category==='Y'?'':'warn'):''}${(c.decision_label||gate.candidate_layer)?badge(c.decision_label||label(gate.candidate_layer),'neutral'):''}</div></div><div class="score">${format(c.score)}<small>Relevance score</small></div></header>
    ${(gate.hard_blockers||[]).length?`<p>Needs review: ${esc((gate.hard_blockers||[]).slice(0,3).map(b=>label(b.reason_code||b.message||b)).join('; '))}</p>`:''}
    <div class="candidate-footer"><span class="small muted">${category?'Probability of documentation support: '+format(prob[category]):'Check the complete model documentation'}</span>${c.model_version_id?`<button data-action="model-detail" data-id="${esc(c.model_version_id)}">View model details ↗</button>`:''}</div></article>`}).join('')||empty('No candidate passed the applicability checks. Review the exclusion reasons below or revise the request.')}
    ${(r.reference_candidates||[]).length?`<section class="research-card"><h3>Mechanism references from other compounds</h3><p class="small muted">References for mechanisms and structure, not directly reusable models of the target drug.</p>${candidateTable(r.reference_candidates)}</section>`:''}
    ${(r.blocked_candidates||[]).length?`<details><summary>Excluded candidates · ${r.blocked_candidates.length}</summary>${candidateTable(r.blocked_candidates)}</details>`:''}
    ${r.unassessed_count?`<p class="small muted">Candidates with unconfirmed applicability: ${fmt(r.unassessed_count)}. These are not recommended. Download the complete research ranking.</p>`:''}
    ${(r.limitations||[]).length?`<details><summary>Method and scope</summary><ul class="small">${r.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></details>`:''}`;
  }
  function candidateTable(rows){return table(['Model','Applicability','Reason',''],rows.map(c=>`<tr><td>${esc(c.name)}</td><td>${esc(c.decision_label||label(c.gate?.candidate_layer))}</td><td>${esc((c.reason_codes||[]).map(label).join('; ')||'—')}</td><td>${c.model_version_id?`<button data-action="model-detail" data-id="${esc(c.model_version_id)}">View details</button>`:''}</td></tr>`))}
  function progress(row){return `<div class="job-progress" role="status"><h3>${row.status==='queued'?'Request queued':'Matching models'}</h3><p>${esc(row.message||row.stage||'Reading the request and candidate model documentation')}</p><div class="progress-track"><i></i></div><p>You can switch pages and return to view this request’s progress.</p></div>`}
  function payload(form){const fd=new FormData(form);const out={species:'human'};for(const key of ['compound','objective','route','population','dose_unit','single_or_multiple','frequency','formulation']){const value=String(fd.get(key)||'').trim();if(value)out[key]=value}const dose=String(fd.get('dose')||'').trim();if(dose)out.dose=Number(dose);else delete out.dose_unit;for(const [key,field] of [['enzymes','required_enzymes'],['transporters','required_transporters']]){const values=String(fd.get(key)||'').split(/[,，;；]/).map(x=>x.trim()).filter(Boolean);if(values.length)out[field]=values}out.prediction_endpoints=fd.getAll('prediction_endpoints').filter(v=>endpoints.some(([e])=>e===v));return out}
  function validateForm(form){const invalid=form.querySelector('.research-extra :invalid');if(invalid)form.querySelector('.research-extra').open=true;return form.reportValidity()}
  async function poll(ticket=generation){
    clearTimeout(timer);if(disposed||ticket!==generation||!job)return;const id=job.job_id;
    try{const row=await api('/workspace/research/jobs/'+encodeURIComponent(id));if(disposed||ticket!==generation||job?.job_id!==id)return;job=row;
      if(['complete','completed'].includes(row.status)){gateResult=null;result=row.result;request=result.request||request;const old=container.querySelector('#research-recommend-form');if(old){const all=state.compounds.filter(c=>c.evidence_count);old.outerHTML=form(all)}sessionStorage.removeItem('pbpk_model_job');container.querySelector('#research-result').innerHTML=renderResult(result);job=null;return}
      if(['failed','cancelled','interrupted'].includes(row.status)){sessionStorage.removeItem('pbpk_model_job');container.querySelector('#research-result').innerHTML=`<div class="note warn"><strong>Recommendation incomplete</strong><p>${esc(typeof row.error==='string'?row.error:row.message||row.error?.message||'Revise the inputs and submit again.')}</p></div>`;job=null;return}
      container.querySelector('#research-result').innerHTML=progress(row);timer=setTimeout(()=>poll(ticket),1600);
    }catch(error){if(!disposed&&ticket===generation){container.querySelector('#research-result').innerHTML=`<div class="note warn">${esc(error.message)}<button data-research-action="resume">Reconnect to job</button></div>`}}
  }
  function mountRecommend(host){
    container=host;disposed=false;const ticket=++generation;const current=()=>!disposed&&ticket===generation;const controller=new AbortController();
    host.addEventListener('input',e=>{if(e.target.closest('.research-extra'))refreshExtra(e.target.closest('form'))},{signal:controller.signal});
    host.addEventListener('change',e=>{if(e.target.closest('.research-extra'))refreshExtra(e.target.closest('form'))},{signal:controller.signal});
    host.addEventListener('invalid',e=>{const details=e.target.closest('.research-extra');if(details)details.open=true}, {capture:true,signal:controller.signal});
    const remembered=sessionStorage.getItem('pbpk_model_job');if(remembered){job={job_id:remembered,status:'running'};container.querySelector('#research-result').innerHTML=progress(job);poll(ticket)}
    else if(!result&&!gateResult){api('/workspace/research/jobs/current').then(row=>{if(current()&&row?.job_id&&!job){job=row;sessionStorage.setItem('pbpk_model_job',row.job_id);host.querySelector('#research-result').innerHTML=progress(row);poll(ticket)}}).catch(()=>{})}
    host.addEventListener('submit',async e=>{if(e.target.id!=='research-recommend-form')return;e.preventDefault();e.stopPropagation();if(!validateForm(e.target))return;if(job){toast('A recommendation is still running. Submit another request after it finishes.');return}const button=e.target.querySelector('[type="submit"]');button.disabled=true;
      try{request=payload(e.target);result=null;gateResult=null;const row=await post('/workspace/research/recommend',request);job=row;sessionStorage.setItem('pbpk_model_job',row.job_id);if(current()){host.querySelector('#research-result').innerHTML=progress(row);poll(ticket)}}
      catch(error){if(error.status===409&&error.detail?.job_id){job=error.detail;sessionStorage.setItem('pbpk_model_job',job.job_id);toast('A recommendation is already running; its progress has been restored.');if(current())poll(ticket)}else if(current())host.querySelector('#research-result').innerHTML=`<div class="note warn">${esc(error.message)}</div>`}finally{button.disabled=false}}, {signal:controller.signal});
    host.addEventListener('click',async e=>{const b=e.target.closest('[data-research-action]');if(!b)return;const action=b.dataset.researchAction;
      if(action==='clear-extra'){const form=b.closest('form');request={...request};for(const key of extraFields){form.elements.namedItem(key).value='';delete request[key]}delete request.required_enzymes;delete request.required_transporters;refreshExtra(form)}
      else if(action==='download')download(result,'pbpk-model-recommendation.json');
      else if(action==='resume')poll(ticket);
      else if(action==='blueprint'){sessionStorage.setItem('pbpk_blueprint_drug',result?.request?.compound||request.compound);location.hash='blueprints'}
      else if(action==='gate'){const form=host.querySelector('#research-recommend-form');if(!validateForm(form))return;if(job){toast('A recommendation is running. Wait for it to finish before checking another request.');return}b.disabled=true;try{request=payload(form);result=null;gateResult=null;const r=await post('/recommendation/public-model-ranking',request);state.rec=r;gateResult=r;if(current())host.querySelector('#research-result').innerHTML=rankingResult(r)}catch(error){toast(error.message)}finally{b.disabled=false}}
    },{signal:controller.signal});
    return ()=>{if(ticket===generation){disposed=true;++generation;clearTimeout(timer)}controller.abort()};
  }
  return {renderRecommend,mountRecommend};
})();
