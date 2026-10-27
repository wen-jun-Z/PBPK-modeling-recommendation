/* The blueprint workspace uses the existing PBPK executor and persisted jobs. */
(() => {
  'use strict';

  const storageKey = 'pbpk_blueprint_workspace';
  // Sampling layouts identify frozen experiment inputs, not alternative methods.
  // New case views use the published main result; saved jobs retain their inputs.
  const defaultRegime = 'late_four';
  const view = {
    cases: [], selected: '', regime: defaultRegime, detail: null, tab: 'structure',
    history: [], job: null, requestedDrug: '', search: '', form: null, referenceHuman: null, referenceCalibration: null,
  };
  const endpointNames = {
    auc: '区间暴露量 AUC', cmax: '峰浓度 Cmax', tmax: '达峰时间 Tmax',
    c_last: '末次浓度', terminal_half_life: '终末半衰期', auc_inf: 'AUC 外推至无穷',
    clearance: '清除率 CL', apparent_clearance: '表观清除率 CL/F',
    vz: '终末分布容积 Vz', apparent_vz: '表观分布容积 Vz/F', mrt: '平均滞留时间 MRT', vss: '稳态分布容积 Vss',
  };
  const organNames = {
    Liver: '肝脏', Kidney: '肾脏', Kidneys: '肾脏', Lung: '肺', Lungs: '肺',
    Brain: '脑', Heart: '心脏', Muscle: '肌肉', Fat: '脂肪', Skin: '皮肤',
    Bone: '骨', Spleen: '脾脏', Pancreas: '胰腺', Stomach: '胃',
    SmallIntestine: '小肠', LargeIntestine: '大肠', Gonads: '生殖腺',
    VenousBlood: '静脉血', ArterialBlood: '动脉血', Rest: '其他组织',
  };
  const basisNames = { measured: '测量值', derived: '推导值', predicted: '预测值', assumption: '假设值', calibrated: '校准值', development_calibrated: '开发数据校准', reference_case_transferred: '沿用案例校准' };
  const evidenceNames = {
    molecular_weight: '分子量', molecular_formula: '分子式', logp: '脂水分配系数 logP',
    pka_types: '解离常数 pKa', solubility: '溶解度', solubility_ph: '溶解度条件 pH',
    binding_partner: '结合蛋白', elimination_scope: '清除范围', total_clearance: '总清除率',
    fraction_unbound: '游离分数', hepatic_clearance: '肝清除率', renal_clearance: '肾清除率',
  };
  const stages = {
    queued: '等待执行', running: '正在运行', preparing: '准备模型与参数',
    validating: '检查输入证据', assembling: '构建模型', building_human: '按人体条件重新构建', calibrating: '校准开发曲线',
    fitting: '校准开发曲线', predicting: '运行浓度预测', simulating: '执行 PBPK 仿真',
    calculating: '计算药代指标', packaging: '整理结果与下载文件', completed: '预测完成', failed: '运行未完成', interrupted: '任务已中断',
  };

  function remembered() {
    try { return JSON.parse(sessionStorage.getItem(storageKey) || '{}'); } catch { return {}; }
  }
  function remember() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({
        selected: view.selected, jobId: view.job?.job_id, requestedDrug: view.requestedDrug,
      }));
    } catch { /* Private browsing can disable session storage. */ }
  }
  function readable(value) {
    if (Array.isArray(value)) return value.map(readable).join(' · ');
    if (value && typeof value === 'object') {
      return value.label || value.name || Object.entries(value).map(([key, val]) => `${key}: ${readable(val)}`).join(' · ');
    }
    return value == null ? '—' : String(value);
  }
  function number(value) {
    if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
    const n = Number(value);
    return n === 0 ? '0' : Math.abs(n) < 0.01 || Math.abs(n) >= 100000 ? n.toExponential(2) : Number(n.toPrecision(4)).toLocaleString('en-US');
  }
  function date(value) {
    if (!value) return '已保存';
    const d = new Date(typeof value === 'number' ? value * 1000 : value);
    return Number.isNaN(d.getTime()) ? '已保存' : d.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
  }
  function safeUrl(value) {
    if (!value) return null;
    try {
      const url = new URL(value, location.origin);
      return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch { return null; }
  }
  function source(value, title = '查看来源 ↗') {
    const url = safeUrl(value);
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(title)}</a>` : '—';
  }
  function status(value) {
    const text = stages[value] || (value === 'ready' ? '已有预测结果' : '已保存');
    return `<span class="bp-status ${['failed', 'interrupted'].includes(value) ? 'failed' : ['running', 'queued'].includes(value) ? 'pending' : ''}">${esc(text)}</span>`;
  }
  function frameTable(headers, rows) {
    return `<div class="table-wrap"><table class="bp-table"><thead><tr>${headers.map(x => `<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
  }
  function caseName(row) { return row.drug || row.compound_name || row.case_id; }
  function caseOptions() {
    const search = view.search.toLowerCase().trim();
    const rows = view.cases.filter(row => !search || [row.drug, row.study_id, row.case_id].join(' ').toLowerCase().includes(search));
    const selected = view.cases.find(row => row.case_id === view.selected);
    if (selected && !rows.some(row => row.case_id === selected.case_id)) rows.unshift(selected);
    return `<option value="">选择药物与研究案例</option>` + rows.map(row => `<option value="${esc(row.case_id)}" ${row.case_id === view.selected ? 'selected' : ''}>${esc(caseName(row))}${row.dose_mg != null ? ' · ' + esc(number(row.dose_mg)) + ' mg 静脉' : ''}</option>`).join('');
  }
  function selector() {
    return `<div class="bp-selector">
      <label>查找药物<input type="search" data-bp-input="search" value="${esc(view.search)}" placeholder="药物名称或研究名称" autocomplete="off"></label>
      <label>研究案例<select data-bp-input="case" aria-label="选择蓝图研究案例">${caseOptions()}</select></label>
    </div>`;
  }
  const humanFields = ['age_years', 'sex', 'weight_kg', 'height_cm'];
  function activeHuman(detail = view.detail) {
    return detail?.human_context?.active_human || detail?.request?.human || {};
  }
  function referenceHuman() {
    return view.referenceHuman || view.detail?.human_context?.reference_human || activeHuman();
  }
  function humanSummary(human) {
    return [human.age_years != null ? `${number(human.age_years)} 岁` : '', human.sex === 'MALE' ? '男性' : human.sex === 'FEMALE' ? '女性' : '', human.weight_kg != null ? `${number(human.weight_kg)} kg` : '', human.height_cm != null ? `${number(human.height_cm)} cm` : ''].filter(Boolean).join(' · ');
  }
  function dosingSummary(dosing = {}) {
    return [dosing.dose_mg != null ? `${number(dosing.dose_mg)} mg` : '', dosing.infusion_minutes != null ? Number(dosing.infusion_minutes) === 0 ? '静脉推注' : `静滴 ${number(dosing.infusion_minutes)} min` : ''].filter(Boolean).join(' · ');
  }
  function calibrationBasis(calibration = {}, title = '校准所用观测资料') {
    const visible = calibration?.visible_points;
    const count = Array.isArray(visible) ? visible.length : visible;
    const human = humanSummary(calibration?.human || {});
    const dose = dosingSummary(calibration?.dosing);
    const label = calibration?.source_label || calibration?.study_id;
    return `<div class="bp-calibration-basis"><h3>${esc(title)}</h3>
      ${label ? `<p class="bp-calibration-study">${esc(label)}</p>` : ''}
      <p>${esc(human || '该历史结果未记录观测数据的人体条件')}</p>
      <p>${[dose, count != null ? `${number(count)} 个采样点` : ''].filter(Boolean).map(esc).join(' · ')}</p></div>`;
  }
  function humanResultNote(detail) {
    const human = detail.human_context;
    if (human?.calibration_mode === 'development_refit') return '<p class="bp-human-result">已使用新观测数据及其人体条件重新校准；拟合结果不等于独立验证。</p>';
    if (human?.calibration_mode === 'transferred' || human?.changed) return '<p class="bp-human-result">历史探索结果：沿用原校准参数，未针对该人体重新校准。</p>';
    return '';
  }
  function context(detail) {
    const dosing = detail.request?.dosing || {};
    const human = activeHuman(detail);
    return `<div class="bp-context">${[
      dosing.route === 'intravenous' ? '静脉单次给药' : dosing.route,
      human.age_years != null ? `${number(human.age_years)} 岁` : null,
      human.weight_kg != null ? `${number(human.weight_kg)} kg` : null,
      human.height_cm != null ? `${number(human.height_cm)} cm` : null,
      human.sex === 'MALE' ? '男性' : human.sex === 'FEMALE' ? '女性' : null,
      detail.structure?.organ_count != null ? `${detail.structure.organ_count} 个器官` : null,
    ].filter(Boolean).map(value => `<span>${esc(value)}</span>`).join('')}</div>`;
  }
  function structure(detail) {
    const model = detail.structure || {};
    const organs = model.organs || [];
    return `<p class="bp-caption">各研究案例共用同一套人体基础模板，包含 15 个器官及动脉、静脉和门静脉循环，并结合药物参数描述分布与清除过程。</p>
      <p class="bp-caption"><strong>备注：</strong>人体条件由研究案例或新观测资料确定，部分案例设定包含假设。下方器官体积与血流量对应上方已计算的预测条件；校准使用的观测人体条件单独列出。</p>
      ${(model.processes || []).length ? `<div class="bp-processes">${model.processes.map(value => `<span>${esc(readable(value))}</span>`).join('')}</div>` : ''}
      ${organs.length ? `<div class="bp-organ-list">${organs.map(organ => `<article class="bp-organ">
        <strong>${esc(organNames[organ.name] || organ.name)}</strong>
        <small>${organ.volume_l != null ? '体积 ' + number(organ.volume_l) + ' L' : ''}${organ.volume_l != null && organ.flow_l_h != null ? ' · ' : ''}${organ.flow_l_h != null ? '血流量 ' + number(organ.flow_l_h) + ' L/h' : ''}</small>
      </article>`).join('')}</div>` : '<p class="bp-caption">当前结果未附组织明细。</p>'}`;
  }
  function parameters(detail) {
    const rows = detail.parameters || [];
    if (!rows.length) return empty('该案例暂无可展示的参数。');
    return frameTable(['参数', '数值', '依据'], rows.map(row => `<tr>
      <td>${esc(row.label || row.name)}</td><td>${esc(typeof row.value === 'number' ? number(row.value) : readable(row.value))}<small>${esc(row.unit || '')}</small></td>
      <td>${esc(basisNames[row.basis] || row.basis || '已登记')}</td></tr>`));
  }
  function evidence(detail) {
    const rows = detail.evidence || [];
    if (!rows.length) return empty('该案例没有可展示的证据条目。');
    const origin = row => {
      if (safeUrl(row.source_url)) return source(row.source_url);
      if (row.basis === 'predicted') return '模型计算记录（非实测）';
      if (row.basis === 'assumption') return '建模假设（非实测）';
      return '原始记录已保存，来源链接待补充';
    };
    return `<p class="bp-caption">本表包含来源数据、模型估算和建模假设。可追溯的原文提供链接，其余条目注明计算或设定依据。</p>` + frameTable(['证据参数', '数值与单位', '来源或依据'], rows.map(row => `<tr>
      <td>${esc(row.label || evidenceNames[row.parameter] || row.parameter)}<small>${esc(basisNames[row.basis] || row.basis || '')}</small></td>
      <td>${esc(typeof row.value === 'number' ? number(row.value) : readable(row.value))} ${esc(row.unit || '')}</td>
      <td>${origin(row)}</td></tr>`));
  }
  function caseCard() {
    const detail = view.detail;
    if (!detail) return `<section class="panel">${empty('选择已有药物案例，查看人体基础模板、参数与证据。')}</section>`;
    const panel = { structure, parameters, evidence }[view.tab] || structure;
    return `<section class="panel bp-case"><div class="bp-kicker">MODEL BLUEPRINT</div>
      <h2 class="bp-case-name">${esc(caseName(detail))}</h2><p class="bp-display-label">${detail.human_context?.changed ? '本次已计算的预测人体条件' : '案例已计算的预测人体条件'}</p>${context(detail)}${humanResultNote(detail)}
      <div class="bp-tabs" role="tablist" aria-label="蓝图内容">
        ${[['structure', '人体基础模板'], ['parameters', '参数'], ['evidence', '证据来源']].map(([key, title]) => `<button type="button" role="tab" data-bp-action="tab" data-tab="${key}" aria-selected="${view.tab === key}" aria-controls="bp-detail-panel">${title}</button>`).join('')}
      </div><div id="bp-detail-panel" role="tabpanel">${panel(detail)}</div>
    </section>`;
  }
  function defaults(detail) {
    const dosing = detail?.request?.dosing || {};
    const target = detail?.request?.target || {};
    return {
      ...Object.fromEntries(humanFields.map(key => [key, ''])),
      dose_mg: dosing.dose_mg ?? '', infusion_minutes: dosing.infusion_minutes ?? 0,
      window_end_h: target.window_end_h ?? 24, development: false,
      source_label: '', source_url: '', observations: '',
      development_dose_mg: '', development_infusion_minutes: '',
    };
  }
  function jobDefaults(detail, job) {
    const request = job?.input || job?.request || {};
    const curve = request.development_curve;
    return {
      ...defaults(detail),
      ...Object.fromEntries(['dose_mg', 'infusion_minutes', 'window_end_h'].filter(key => request[key] != null).map(key => [key, request[key]])),
      ...(curve ? {
        development: true, source_label: curve.source_label || '', source_url: curve.source_url || '',
        development_dose_mg: curve.dose_mg ?? '', development_infusion_minutes: curve.infusion_minutes ?? '',
        observations: (curve.observations || []).map(row => `${row.time_h}, ${row.concentration_ng_ml}`).join('\n'),
        ...Object.fromEntries(humanFields.map(key => [key, curve.human?.[key] ?? ''])),
      } : {}),
    };
  }
  function scenario() {
    const form = view.form || defaults(view.detail);
    const busy = ['queued', 'running'].includes(view.job?.status);
    return `<section class="panel bp-scenario"><div class="bp-kicker">PREDICTION SETTINGS</div>
      <h2>设置预测场景</h2><p>使用观测数据校准后的参数，计算给药方案下的预测曲线。</p>
      ${view.detail ? calibrationBasis(view.referenceCalibration, '已有案例校准依据') : ''}
      <form id="bp-prediction-form">
        ${!form.development ? `<div class="bp-prediction-human"><h3>本案例预测人体条件</h3><p>${esc(humanSummary(referenceHuman()) || '请先选择研究案例')}</p><p>按案例既定条件预测。改变人体条件须提供对应的新观测数据。</p></div>` : ''}
        <h3 class="bp-secondary-title">给药方案</h3>
        <div class="form-grid"><label>剂量（mg）<input name="dose_mg" type="number" min="0.000001" max="100000" step="any" required value="${esc(form.dose_mg)}"></label>
          <label>静滴时长（min）<input name="infusion_minutes" type="number" min="0" max="10080" step="any" required value="${esc(form.infusion_minutes)}"><span class="small muted">0 表示静脉推注</span></label>
          <label class="full">预测结束时间（h）<input name="window_end_h" type="number" min="0.01" max="168" step="any" required value="${esc(form.window_end_h)}"></label>
        </div>
        <label class="bp-check"><input name="development" type="checkbox" ${form.development ? 'checked' : ''}>使用新的观测数据校准</label>
        ${!form.development ? '<p class="bp-calibration-note">复用已有案例的校准参数；改变给药方案不会自动重新调参。</p>' : ''}
        <div class="bp-development" ${form.development ? '' : 'hidden'}>
          <p>填写实际观测曲线及其研究条件。系统按这些人体条件重新构建、校准，再计算上方给药方案；本次预测使用相同的人体条件。</p>
          <fieldset class="bp-human-fields"><legend>观测数据对应的人体条件</legend>
            <div class="form-grid">
              <label>年龄（岁）<input name="age_years" type="number" min="18" max="80" step="any" placeholder="按观测资料填写" value="${esc(form.age_years)}" ${form.development ? 'required' : 'disabled'}></label>
              <label>性别<select name="sex" ${form.development ? 'required' : 'disabled'}><option value="" ${!form.sex ? 'selected' : ''}>按观测资料选择</option><option value="MALE" ${form.sex === 'MALE' ? 'selected' : ''}>男性</option><option value="FEMALE" ${form.sex === 'FEMALE' ? 'selected' : ''}>女性</option></select></label>
              <label>体重（kg）<input name="weight_kg" type="number" min="0.000001" max="300" step="any" placeholder="按观测资料填写" value="${esc(form.weight_kg)}" ${form.development ? 'required' : 'disabled'}></label>
              <label>身高（cm）<input name="height_cm" type="number" min="0.000001" max="250" step="any" placeholder="按观测资料填写" value="${esc(form.height_cm)}" ${form.development ? 'required' : 'disabled'}></label>
            </div>
            <p>18–80 岁仅为当前构建器的输入范围，不表示全部年龄均已验证。请按资料填写；信息缺失时先补齐。</p>
          </fieldset>
          <label>观测数据来源<input name="source_label" minlength="3" maxlength="300" placeholder="例如：研究名称、图表编号" value="${esc(form.source_label)}" ${form.development ? 'required' : 'disabled'}></label>
          <label>来源网页（可选）<input name="source_url" type="url" placeholder="https://…" value="${esc(form.source_url)}" ${form.development ? '' : 'disabled'}></label>
          <div class="form-grid"><label>观测曲线剂量（mg）<input name="development_dose_mg" type="number" min="0.000001" max="100000" step="any" value="${esc(form.development_dose_mg)}" ${form.development ? 'required' : 'disabled'}></label>
            <label>静滴时长（min）<input name="development_infusion_minutes" type="number" min="0" max="10080" step="any" value="${esc(form.development_infusion_minutes)}" ${form.development ? 'required' : 'disabled'}></label></div>
          <label style="margin-top:13px">时间（h）, 浓度（ng/mL）<textarea name="observations" spellcheck="false" placeholder="填写至少 5 个实际采样点，每行两列" ${form.development ? 'required' : 'disabled'}>${esc(form.observations)}</textarea></label>
          <label>或导入两列 CSV<input data-bp-input="csv" type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values" ${form.development ? '' : 'disabled'}></label>
          <p class="bp-human-draft">表单内容尚未提交。上方人体参数与下方结果仍属于已完成的计算；新数据拟合不等于独立验证。</p>
        </div>
        <div class="bp-error" id="bp-form-error" role="alert" hidden></div>
        <button class="primary" type="submit" ${!view.selected || !view.detail || busy ? 'disabled' : ''}>${busy ? '正在执行当前任务…' : '生成预测蓝图 →'}</button>
      </form>
    </section>`;
  }
  function progress() {
    const job = view.job;
    if (!job) return '';
    const busy = ['queued', 'running'].includes(job.status);
    const failed = ['failed', 'interrupted'].includes(job.status);
    const stage = job.stage_label || stages[job.stage] || (typeof job.stage === 'string' && /[\u3400-\u9fff]/.test(job.stage) ? job.stage : stages[job.status]);
    return `<section class="bp-progress ${failed ? 'error' : ''}" aria-live="polite"><header>
      <h3>${busy ? '<span class="bp-pulse" aria-hidden="true"></span>' : ''}${esc(stage || '正在读取任务')}</h3>
      <button type="button" data-bp-action="refresh-job">刷新状态</button></header>
      <p>${failed ? esc(job.error || '运行失败，检查给药与开发数据后可以重新提交。') : busy ? '任务在服务器持续执行。切换页面或刷新后仍可查看进度。' : '预测结果已保存，可以查看曲线、药代指标与下载文件。'}</p>
      ${job.message ? `<p>${esc(job.message)}</p>` : ''}
      <p data-bp-poll-error class="bp-error" hidden></p>
    </section>`;
  }
  function endpointValue(key, value) {
    if (!value || value.status === 'not_estimable' || value.status === 'unavailable') return null;
    const mapping = [
      ['value_ng_h_per_ml', 'ng·h/mL', 1], ['value_ng_per_ml', 'ng/mL', 1],
      ['value_mg_h_per_l', 'ng·h/mL', 1000], ['value_mg_per_l', 'ng/mL', 1000],
      ['value_h', 'h', 1], ['value_l_per_h', 'L/h', 1], ['value_l_h', 'L/h', 1],
      ['value_l', 'L', 1], ['value_umol_h_per_l', 'μmol·h/L', 1], ['value_umol_per_l', 'μmol/L', 1],
    ];
    for (const [field, unit, factor] of mapping) {
      if (value[field] != null && Number.isFinite(Number(value[field]))) return { value: Number(value[field]) * factor, unit };
    }
    if (typeof value === 'number') return { value, unit: '' };
    if (value.value != null && Number.isFinite(Number(value.value))) return { value: Number(value.value), unit: value.unit || '' };
    if (key === 'tmax' && value.tmax_h != null) return { value: value.tmax_h, unit: 'h' };
    return null;
  }
  function downloads(data) {
    const base = data.download_url;
    if (!base || !safeUrl(base)) return '';
    const link = format => {
      const url = new URL(base, location.origin); url.searchParams.set('format', format); return url.pathname + url.search;
    };
    return `<div class="bp-downloads">${[['json', '蓝图 JSON'], ['csv', '曲线 CSV'], ['zip', '完整模型包']].map(([format, title]) => `<a href="${esc(link(format))}" download>${title} ↓</a>`).join('')}</div>`;
  }
  function resultCard() {
    const detail = view.job?.status === 'completed' && view.job.result ? view.job.result : view.detail;
    if (!detail) return '';
    const endpoints = detail.endpoints?.endpoints || detail.endpoints || {};
    const curve = detail.curve || {};
    // Endpoint trajectories include the exact requested window boundaries.
    // The raw solver curve can extend much farther to estimate terminal PK.
    const windowCurve = endpoints.plasma_concentration_time;
    const useWindowCurve = windowCurve?.time_h?.length && windowCurve?.concentration_mg_per_l?.length;
    const times = useWindowCurve ? windowCurve.time_h : curve.time_h || [];
    const values = useWindowCurve ? windowCurve.concentration_mg_per_l.map(value => value * 1000) : curve.concentration_ng_ml || curve.concentration_ng_per_ml || [];
    const points = times.map((time, i) => [Number(time), Number(values[i])]).filter(([time, value]) => Number.isFinite(time) && Number.isFinite(value));
    const cards = Object.entries(endpointNames).map(([key, title]) => ({ key, title, ...endpointValue(key, endpoints[key]) })).filter(row => row.value != null);
    const unavailable = Object.entries(endpointNames).filter(([key]) => endpoints[key] && !endpointValue(key, endpoints[key]));
    const request = detail.request || {};
    const dosing = request.dosing || {};
    const target = request.target || {};
    const series = points.length > 1 ? [{ name: 'PBPK 预测', points, kind: 'line' }] : [];
    const chart = series.length && window.PBPKCharts?.curve ? window.PBPKCharts.curve(series, { xLabel: '时间（h）', yLabel: '血浆浓度（ng/mL）' }) : empty(points.length ? '图表模块尚未就绪，请刷新页面或下载曲线。' : '当前结果没有可绘制的浓度曲线。');
    const visible = detail.calibration?.visible_points;
    const calibrationPoints = visible != null ? esc(Array.isArray(visible) ? visible.length : visible) : null;
    const calibrationDescription = calibrationPoints != null
      ? detail.human_context?.calibration_mode === 'transferred'
        ? `原案例校准使用 ${calibrationPoints} 个观测采样点；本次未重新校准。`
        : `当前校准使用 ${calibrationPoints} 个观测采样点。`
      : '';
    return `<section class="panel bp-result"><div class="bp-card-heading"><div><div class="bp-kicker">PREDICTION RESULT</div>
      <h2>${view.job?.status === 'completed' && view.job.result ? '本次预测结果' : '已有案例结果'} · ${esc(caseName(detail))}</h2></div>${downloads({ ...detail, download_url: view.job?.status === 'completed' ? view.job.download_url || detail.download_url : detail.download_url })}</div>
      <p class="bp-result-meta">${[dosing.dose_mg != null ? `${number(dosing.dose_mg)} mg` : '', dosing.infusion_minutes != null ? dosing.infusion_minutes === 0 ? '静脉推注' : `静滴 ${number(dosing.infusion_minutes)} min` : '', target.window_end_h != null ? `预测至 ${number(target.window_end_h)} h` : ''].filter(Boolean).map(esc).join(' · ')}</p>
      <p class="bp-result-human">计算所用人体：${esc(humanSummary(activeHuman(detail)))}</p>${humanResultNote(detail)}
      ${calibrationBasis(detail.calibration)}
      <div class="bp-curve">${chart}</div>
      ${cards.length ? `<div class="bp-pk">${cards.map(row => `<article><h3>${esc(row.title)}</h3><strong>${number(row.value)}</strong><small>${esc(row.unit)}</small></article>`).join('')}</div>` : ''}
      ${unavailable.length ? `<p class="bp-caption">当前曲线不足以可靠估计：${unavailable.map(([, name]) => esc(name)).join('、')}。未用零值代替。</p>` : ''}
      <p class="bp-caption">以上为模型预测值。${calibrationDescription}执行完成与独立科学验证分开记录；本次预测不代表已验证临床准确性。</p>
      ${(detail.limitations || []).length ? `<details><summary>适用范围与结果说明</summary><ul class="bp-limitations">${detail.limitations.map(item => `<li>${esc(item)}</li>`).join('')}</ul></details>` : ''}
    </section>`;
  }
  function history() {
    return `<section class="panel bp-history"><div class="bp-card-heading"><h2>我的运行记录</h2><button type="button" data-bp-action="refresh-history">刷新记录 ↻</button></div>
      <p class="small muted">浏览器提交的任务单独保存；原有研究结果保持可查看与下载。</p>
      ${view.history.length ? frameTable(['药物 / 场景', '提交时间（北京时间）', '执行状态', ''], view.history.map(job => {
        const request = job.request || job.input || {};
        const row = view.cases.find(item => item.case_id === (job.case_id || request.case_id));
        return `<tr><td><strong>${esc(job.drug || request.drug || (row ? caseName(row) : '蓝图预测'))}</strong><small>${request.dose_mg != null ? esc(number(request.dose_mg)) + ' mg' : ''}${request.window_end_h != null ? ' · ' + number(request.window_end_h) + ' h' : ''}</small>${request.human ? `<small>${esc(humanSummary(request.human))}</small>` : ''}</td>
          <td>${esc(date(job.created_at))}</td><td>${status(job.status)}</td>
          <td><button type="button" data-bp-action="open-job" data-job="${esc(job.job_id)}">${job.status === 'completed' ? '查看结果 ↗' : '查看进度 ↗'}</button></td></tr>`;
      })) : '<p class="bp-caption">尚未在网页提交新任务。上方已完成案例可直接查看。</p>'}</section>`;
  }
  function content() {
    const unsupported = view.requestedDrug && !view.cases.some(row => caseName(row).toLowerCase() === view.requestedDrug.toLowerCase());
    return `<header class="bp-intro"><div><p class="eyebrow">EVIDENCE → BLUEPRINT → PREDICTION</p><h1>模型蓝图与预测</h1><p>药物证据与人体模板 → 参数校准 → 可行性检查与修复 → 目标条件预测。</p></div><div class="bp-intro-count"><strong>${view.cases.length}</strong>可执行研究案例</div></header>
      ${unsupported ? `<div class="bp-unavailable">${esc(view.requestedDrug)} 暂无完整的可执行输入。请先选择下方已有案例；不会自动替换药物并运行。</div>` : ''}
      ${selector()}<div data-bp-region="progress">${progress()}</div>
      <div class="bp-layout"><div data-bp-region="case">${caseCard()}</div><div data-bp-region="scenario">${scenario()}</div></div>
      <div data-bp-region="result">${resultCard()}</div><div data-bp-region="history">${history()}</div>`;
  }
  function readForm(root) {
    const form = root.querySelector('#bp-prediction-form');
    if (!form) return;
    const data = Object.fromEntries(new FormData(form));
    view.form = { ...view.form, ...data, development: form.elements.development.checked };
  }
  function parseCurve(text) {
    const rows = text.trim().split(/\r?\n/).filter(line => line.trim());
    const points = [];
    rows.forEach((line, index) => {
      const fields = line.trim().split(/[,;\t ]+/);
      if (index === 0 && fields.length === 2 && !Number.isFinite(Number(fields[0]))) return;
      if (fields.length !== 2 || fields.some(value => value === '' || !Number.isFinite(Number(value)))) throw new Error(`开发曲线第 ${index + 1} 行应为两列数值：时间（h）, 浓度（ng/mL）。`);
      const [time_h, concentration_ng_ml] = fields.map(Number);
      if (time_h < 0 || time_h > 168 || concentration_ng_ml <= 0 || concentration_ng_ml > 1e9) throw new Error('采样时间须在 0–168 h，浓度须大于 0 且不超过 10⁹ ng/mL。');
      points.push({ time_h, concentration_ng_ml });
    });
    if (points.length < 5) throw new Error('请提供至少 5 个开发采样点。');
    if (points.length > 128) throw new Error('单次最多支持 128 个开发采样点。');
    if (points.some((row, i) => i && row.time_h <= points[i - 1].time_h)) throw new Error('采样时间应严格递增，不能重复。');
    return points;
  }
  async function getDetail() {
    return view.selected ? api(`/workspace/blueprint-cases/${encodeURIComponent(view.selected)}?regime=${encodeURIComponent(view.regime)}`) : null;
  }
  async function render() {
    const saved = remembered();
    const [cases, jobs] = await Promise.all([api('/workspace/blueprint-cases'), api('/workspace/blueprint-predictions')]);
    view.cases = cases.items || [];
    view.history = jobs.items || [];
    if (!view.requestedDrug && saved.requestedDrug) view.requestedDrug = saved.requestedDrug;
    if (!view.selected && saved.selected) view.selected = saved.selected;
    view.regime = defaultRegime;
    let requested = null;
    try { requested = sessionStorage.getItem('pbpk_blueprint_drug'); sessionStorage.removeItem('pbpk_blueprint_drug'); } catch { /* Optional handoff. */ }
    if (requested) {
      view.requestedDrug = requested;
      view.selected = view.cases.find(row => caseName(row).toLowerCase() === requested.toLowerCase())?.case_id || '';
      view.job = null;
      view.form = null;
    } else if (!view.selected && !view.requestedDrug) {
      view.selected = view.cases.find(row => row.drug?.toLowerCase() === 'midazolam')?.case_id || view.cases[0]?.case_id || '';
    }
    if (view.selected && !view.cases.some(row => row.case_id === view.selected)) view.selected = '';
    if (!requested && saved.jobId && !view.job) {
      try {
        const job = await api('/workspace/blueprint-predictions/' + encodeURIComponent(saved.jobId));
        const caseId = job.case_id || job.request?.case_id || job.input?.case_id || job.result?.case_id;
        if (!caseId || caseId === view.selected) view.job = job;
      } catch { /* Removed task records do not block the case browser. */ }
    }
    const jobRegime = (view.job?.input || view.job?.request || {}).regime;
    if (['early_four', 'late_four'].includes(jobRegime)) view.regime = jobRegime;
    const caseDetail = await getDetail();
    view.referenceHuman = caseDetail?.request?.human || null;
    view.referenceCalibration = caseDetail?.calibration || null;
    view.detail = view.job?.status === 'completed' && view.job.result ? view.job.result : caseDetail;
    if (!view.form) view.form = jobDefaults(view.detail, view.job);
    remember();
    return `<div class="bp-workspace" id="bp-workspace">${content()}</div>`;
  }
  function mount(container) {
    const root = container.querySelector('#bp-workspace');
    if (!root) return () => {};
    let disposed = false;
    let timer = null;
    let loadVersion = 0;
    let polling = false;
    const replace = (region, html) => { if (!disposed) root.querySelector(`[data-bp-region="${region}"]`).innerHTML = html; };
    const drawJob = () => {
      readForm(root);
      if (view.job?.status === 'completed' && view.job.result) {
        view.detail = view.job.result;
        replace('case', caseCard());
      }
      replace('progress', progress()); replace('result', resultCard()); replace('scenario', scenario());
    };
    const showError = error => {
      const field = root.querySelector('#bp-form-error');
      if (field && !disposed) { field.hidden = false; field.textContent = error.message || String(error); }
    };
    const schedule = () => {
      clearTimeout(timer);
      if (!disposed && ['queued', 'running'].includes(view.job?.status)) timer = setTimeout(refreshJob, 3000);
    };
    async function refreshHistory() {
      const response = await api('/workspace/blueprint-predictions');
      if (disposed) return;
      view.history = response.items || [];
      replace('history', history());
    }
    async function refreshJob() {
      if (disposed || polling || !view.job?.job_id) return;
      polling = true;
      const id = view.job.job_id;
      try {
        const job = await api('/workspace/blueprint-predictions/' + encodeURIComponent(id));
        if (disposed || view.job?.job_id !== id) return;
        const wasBusy = ['queued', 'running'].includes(view.job.status);
        view.job = job; remember();
        replace('progress', progress());
        if (!['queued', 'running'].includes(job.status) && wasBusy) { drawJob(); await refreshHistory(); }
      } catch (error) {
        const field = root.querySelector('[data-bp-poll-error]');
        if (!disposed && field) { field.hidden = false; field.textContent = '暂时无法读取状态：' + error.message; }
      } finally { polling = false; schedule(); }
    }
    async function selectCase() {
      const version = ++loadVersion;
      clearTimeout(timer);
      view.job = null; view.form = null; view.detail = null; view.referenceHuman = null; view.referenceCalibration = null; remember();
      replace('case', '<section class="panel"><div class="loading">正在读取模型蓝图…</div></section>');
      replace('progress', ''); replace('result', ''); replace('scenario', scenario());
      const detail = await getDetail();
      if (disposed || version !== loadVersion) return;
      view.detail = detail; view.referenceHuman = detail?.request?.human || null; view.referenceCalibration = detail?.calibration || null; view.form = defaults(detail);
      replace('case', caseCard()); replace('scenario', scenario()); replace('result', resultCard());
    }
    async function onChange(event) {
      const field = event.target;
      try {
        if (field.dataset.bpInput === 'case') {
          view.selected = field.value; view.requestedDrug = '';
          view.regime = defaultRegime;
          root.querySelector('.bp-unavailable')?.remove();
          await selectCase();
        } else if (field.name === 'development') {
          readForm(root); replace('scenario', scenario());
        } else if (field.dataset.bpInput === 'csv' && field.files?.[0]) {
          const file = field.files[0];
          if (file.size > 200000) throw new Error('开发曲线文件应小于 200 KB。');
          const text = await file.text(); parseCurve(text);
          if (disposed) return;
          root.querySelector('[name="observations"]').value = text;
          readForm(root);
        }
      } catch (error) { showError(error); }
    }
    function onInput(event) {
      if (event.target.dataset.bpInput === 'search') {
        view.search = event.target.value;
        root.querySelector('[data-bp-input="case"]').innerHTML = caseOptions();
      } else if (event.target.closest('#bp-prediction-form')) readForm(root);
    }
    async function onClick(event) {
      const button = event.target.closest('[data-bp-action]');
      if (!button || !root.contains(button)) return;
      const action = button.dataset.bpAction;
      try {
        if (action === 'tab') { view.tab = button.dataset.tab; replace('case', caseCard()); }
        else if (action === 'refresh-history') { button.disabled = true; await refreshHistory(); }
        else if (action === 'refresh-job') await refreshJob();
        else if (action === 'open-job') {
          const job = await api('/workspace/blueprint-predictions/' + encodeURIComponent(button.dataset.job));
          if (disposed) return;
          const request = job.input || job.request || {};
          const caseId = job.result?.case_id || job.case_id || request.case_id;
          if (caseId) view.selected = caseId;
          view.regime = ['early_four', 'late_four'].includes(request.regime) ? request.regime : defaultRegime;
          const caseDetail = await getDetail();
          if (disposed) return;
          view.referenceHuman = caseDetail?.request?.human || null;
          view.referenceCalibration = caseDetail?.calibration || null;
          view.detail = job.result || caseDetail;
          view.form = jobDefaults(view.detail, job);
          view.requestedDrug = '';
          view.search = '';
          root.querySelector('[data-bp-input="search"]').value = '';
          root.querySelector('[data-bp-input="case"]').innerHTML = caseOptions();
          view.job = job; remember();
          replace('case', caseCard()); replace('scenario', scenario());
          drawJob(); schedule();
          root.querySelector('[data-bp-region="progress"]').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      } catch (error) { showError(error); } finally { button.disabled = false; }
    }
    async function onSubmit(event) {
      if (event.target.id !== 'bp-prediction-form') return;
      event.preventDefault(); event.stopPropagation();
      readForm(root);
      const form = view.form;
      const button = event.target.querySelector('button[type="submit"]');
      button.disabled = true;
      root.querySelector('#bp-form-error').hidden = true;
      try {
        if (!view.selected) throw new Error('请先选择已有药物案例。');
        const payload = { case_id: view.selected, regime: view.regime, dose_mg: Number(form.dose_mg), infusion_minutes: Number(form.infusion_minutes), window_end_h: Number(form.window_end_h) };
        if (payload.infusion_minutes / 60 >= payload.window_end_h) throw new Error('预测结束时间应晚于静滴结束时间。');
        if (form.development) {
          if (humanFields.some(key => form[key] == null || String(form[key]).trim() === '')) throw new Error('请补齐观测数据对应的年龄、性别、体重和身高。');
          const human = { age_years: Number(form.age_years), sex: form.sex, weight_kg: Number(form.weight_kg), height_cm: Number(form.height_cm) };
          if (!Number.isFinite(human.age_years) || human.age_years < 18 || human.age_years > 80) throw new Error('观测数据的人体年龄须在当前构建范围 18–80 岁内；此范围不代表均已验证。');
          if (!['MALE', 'FEMALE'].includes(human.sex)) throw new Error('请填写观测资料记载的性别。');
          if (!Number.isFinite(human.weight_kg) || human.weight_kg <= 0 || human.weight_kg > 300) throw new Error('观测体重须大于 0 且不超过 300 kg。');
          if (!Number.isFinite(human.height_cm) || human.height_cm <= 0 || human.height_cm > 250) throw new Error('观测身高须大于 0 且不超过 250 cm。');
          payload.human = human;
          payload.development_curve = {
            human,
            source_label: form.source_label.trim(), ...(form.source_url ? { source_url: form.source_url } : {}),
            dose_mg: Number(form.development_dose_mg), infusion_minutes: Number(form.development_infusion_minutes),
            observations: parseCurve(form.observations),
          };
        }
        const job = await post('/workspace/blueprint-predictions', payload);
        view.job = job; remember();
        if (disposed) return;
        drawJob(); await refreshHistory(); schedule();
      } catch (error) { showError(error); } finally { if (!['queued', 'running'].includes(view.job?.status)) button.disabled = false; }
    }
    root.addEventListener('click', onClick);
    root.addEventListener('input', onInput);
    root.addEventListener('change', onChange);
    root.addEventListener('submit', onSubmit);
    schedule();
    return () => {
      disposed = true; loadVersion += 1; clearTimeout(timer);
      root.removeEventListener('click', onClick); root.removeEventListener('input', onInput);
      root.removeEventListener('change', onChange); root.removeEventListener('submit', onSubmit);
    };
  }
  window.PBPKBlueprint = {
    render, mount,
    setCompound(name) {
      view.requestedDrug = String(name || '').trim(); view.selected = ''; view.job = null; view.form = null; view.referenceHuman = null; view.referenceCalibration = null;
      try { sessionStorage.setItem('pbpk_blueprint_drug', view.requestedDrug); } catch { /* Optional persistence. */ }
    },
  };
})();
