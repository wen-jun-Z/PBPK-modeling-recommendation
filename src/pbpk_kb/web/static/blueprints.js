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
    auc: 'Exposure over the interval (AUC)', cmax: 'Peak concentration Cmax', tmax: 'Time to peak Tmax',
    c_last: 'Last concentration', terminal_half_life: 'Terminal half-life', auc_inf: 'AUC extrapolated to infinity',
    clearance: 'Clearance CL', apparent_clearance: 'Apparent clearance CL/F',
    vz: 'Terminal distribution volume Vz', apparent_vz: 'Apparent distribution volume Vz/F', mrt: 'Mean residence time MRT', vss: 'Steady-state distribution volume Vss',
  };
  const organNames = {
    Liver: 'Liver', Kidney: 'Kidneys', Kidneys: 'Kidneys', Lung: 'Lungs', Lungs: 'Lungs',
    Brain: 'Brain', Heart: 'Heart', Muscle: 'Muscle', Fat: 'Adipose tissue', Skin: 'Skin',
    Bone: 'Bone', Spleen: 'Spleen', Pancreas: 'Pancreas', Stomach: 'Stomach',
    SmallIntestine: 'Small intestine', LargeIntestine: 'Large intestine', Gonads: 'Gonads',
    VenousBlood: 'Venous blood', ArterialBlood: 'Arterial blood', Rest: 'Other tissues',
  };
  const basisNames = { measured: 'Measured', derived: 'Derived', predicted: 'Predicted', assumption: 'Assumed', calibrated: 'Calibrated', development_calibrated: 'Calibrated on development data', reference_case_transferred: 'Saved case calibration' };
  const evidenceNames = {
    molecular_weight: 'Molecular weight', molecular_formula: 'Molecular formula', logp: 'Lipophilicity logP',
    pka_types: 'Dissociation constant pKa', solubility: 'Solubility', solubility_ph: 'Solubility measurement pH',
    binding_partner: 'Binding protein', elimination_scope: 'Clearance scope', total_clearance: 'Total clearance',
    fraction_unbound: 'Unbound fraction', hepatic_clearance: 'Hepatic clearance', renal_clearance: 'Renal clearance',
  };
  const stages = {
    queued: 'Waiting to run', running: 'Running', preparing: 'Preparing model and parameters',
    validating: 'Checking input evidence', assembling: 'Building model', building_human: 'Rebuilding for the specified human characteristics', calibrating: 'Calibrating to development observations',
    fitting: 'Calibrating to development observations', predicting: 'Predicting concentration', simulating: 'Running PBPK simulation',
    calculating: 'Calculating PK endpoints', packaging: 'Preparing results and downloads', completed: 'Prediction complete', failed: 'Run incomplete', interrupted: 'Job interrupted',
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
    if (!value) return 'Saved';
    const d = new Date(typeof value === 'number' ? value * 1000 : value);
    return Number.isNaN(d.getTime()) ? 'Saved' : d.toLocaleString('en-US', { timeZone: 'Asia/Shanghai', hour12: false });
  }
  function safeUrl(value) {
    if (!value) return null;
    try {
      const url = new URL(value, location.origin);
      return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
    } catch { return null; }
  }
  function source(value, title = 'View source ↗') {
    const url = safeUrl(value);
    return url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(title)}</a>` : '—';
  }
  function status(value) {
    const text = stages[value] || (value === 'ready' ? 'Saved prediction results' : 'Saved');
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
    return `<option value="">Choose a compound and study case</option>` + rows.map(row => `<option value="${esc(row.case_id)}" ${row.case_id === view.selected ? 'selected' : ''}>${esc(caseName(row))}${row.dose_mg != null ? ' · ' + esc(number(row.dose_mg)) + ' mg intravenously' : ''}</option>`).join('');
  }
  function selector() {
    return `<div class="bp-selector">
      <label>Find a compound<input type="search" data-bp-input="search" value="${esc(view.search)}" placeholder="Compound or study name" autocomplete="off"></label>
      <label>Study case<select data-bp-input="case" aria-label="Select blueprint study case">${caseOptions()}</select></label>
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
    return [human.age_years != null ? `${number(human.age_years)} years` : '', human.sex === 'MALE' ? 'Male' : human.sex === 'FEMALE' ? 'Female' : '', human.weight_kg != null ? `${number(human.weight_kg)} kg` : '', human.height_cm != null ? `${number(human.height_cm)} cm` : ''].filter(Boolean).join(' · ');
  }
  function dosingSummary(dosing = {}) {
    return [dosing.dose_mg != null ? `${number(dosing.dose_mg)} mg` : '', dosing.infusion_minutes != null ? Number(dosing.infusion_minutes) === 0 ? 'IV bolus' : `IV infusion ${number(dosing.infusion_minutes)} min` : ''].filter(Boolean).join(' · ');
  }
  function calibrationBasis(calibration = {}, title = 'Calibration observations') {
    const visible = calibration?.visible_points;
    const count = Array.isArray(visible) ? visible.length : visible;
    const human = humanSummary(calibration?.human || {});
    const dose = dosingSummary(calibration?.dosing);
    const label = calibration?.source_label || calibration?.study_id;
    return `<div class="bp-calibration-basis"><h3>${esc(title)}</h3>
      ${label ? `<p class="bp-calibration-study">${esc(label)}</p>` : ''}
      <p>${esc(human || 'Human conditions for these historical observations were not recorded')}</p>
      <p>${[dose, count != null ? `${number(count)} sampling points` : ''].filter(Boolean).map(esc).join(' · ')}</p></div>`;
  }
  function humanResultNote(detail) {
    const human = detail.human_context;
    if (human?.calibration_mode === 'development_refit') return '<p class="bp-human-result">Recalibrated using new observations and their human conditions. A fitted result is not independent validation.</p>';
    if (human?.calibration_mode === 'transferred' || human?.changed) return '<p class="bp-human-result">Historical exploratory result: original calibration parameters were reused without refitting for this human profile.</p>';
    return '';
  }
  function context(detail) {
    const dosing = detail.request?.dosing || {};
    const human = activeHuman(detail);
    return `<div class="bp-context">${[
      dosing.route === 'intravenous' ? 'Single intravenous dose' : dosing.route,
      human.age_years != null ? `${number(human.age_years)} years` : null,
      human.weight_kg != null ? `${number(human.weight_kg)} kg` : null,
      human.height_cm != null ? `${number(human.height_cm)} cm` : null,
      human.sex === 'MALE' ? 'Male' : human.sex === 'FEMALE' ? 'Female' : null,
      detail.structure?.organ_count != null ? `${detail.structure.organ_count} organs` : null,
    ].filter(Boolean).map(value => `<span>${esc(value)}</span>`).join('')}</div>`;
  }
  function structure(detail) {
    const model = detail.structure || {};
    const organs = model.organs || [];
    return `<p class="bp-caption">Study cases share a human physiology template with 15 organs and arterial, venous, and portal circulation; drug parameters describe distribution and clearance.</p>
      <p class="bp-caption"><strong>Note: </strong>Human conditions come from the study case or new observations; some case settings include assumptions. Organ volumes and flows below belong to the completed prediction shown above. Calibration study conditions are listed separately.</p>
      ${(model.processes || []).length ? `<div class="bp-processes">${model.processes.map(value => `<span>${esc(readable(value))}</span>`).join('')}</div>` : ''}
      ${organs.length ? `<div class="bp-organ-list">${organs.map(organ => `<article class="bp-organ">
        <strong>${esc(organNames[organ.name] || organ.name)}</strong>
        <small>${organ.volume_l != null ? 'Volume ' + number(organ.volume_l) + ' L' : ''}${organ.volume_l != null && organ.flow_l_h != null ? ' · ' : ''}${organ.flow_l_h != null ? 'Blood flow ' + number(organ.flow_l_h) + ' L/h' : ''}</small>
      </article>`).join('')}</div>` : '<p class="bp-caption">No tissue details are attached to this result.</p>'}`;
  }
  function parameters(detail) {
    const rows = detail.parameters || [];
    if (!rows.length) return empty('No parameters are available for this case.');
    return frameTable(['Parameter', 'Value', 'Basis'], rows.map(row => `<tr>
      <td>${esc(row.label || row.name)}</td><td>${esc(typeof row.value === 'number' ? number(row.value) : readable(row.value))}<small>${esc(row.unit || '')}</small></td>
      <td>${esc(basisNames[row.basis] || row.basis || 'Registered')}</td></tr>`));
  }
  function evidence(detail) {
    const rows = detail.evidence || [];
    if (!rows.length) return empty('No evidence records are available for this case.');
    const origin = row => {
      if (safeUrl(row.source_url)) return source(row.source_url);
      if (row.basis === 'predicted') return 'Model calculation (not measured)';
      if (row.basis === 'assumption') return 'Modeling assumption (not measured)';
      return 'Original record preserved; source link pending';
    };
    return `<p class="bp-caption">This table includes source data, model estimates, and assumptions. Verified sources have links; other entries identify their calculation or assumption basis.</p>` + frameTable(['Evidence parameter', 'Value and unit', 'Source or basis'], rows.map(row => `<tr>
      <td>${esc(row.label || evidenceNames[row.parameter] || row.parameter)}<small>${esc(basisNames[row.basis] || row.basis || '')}</small></td>
      <td>${esc(typeof row.value === 'number' ? number(row.value) : readable(row.value))} ${esc(row.unit || '')}</td>
      <td>${origin(row)}</td></tr>`));
  }
  function caseCard() {
    const detail = view.detail;
    if (!detail) return `<section class="panel">${empty('Choose an existing study case to inspect its physiology template, parameters, and evidence.')}</section>`;
    const panel = { structure, parameters, evidence }[view.tab] || structure;
    return `<section class="panel bp-case"><div class="bp-kicker">MODEL BLUEPRINT</div>
      <h2 class="bp-case-name">${esc(caseName(detail))}</h2><p class="bp-display-label">${detail.human_context?.changed ? 'Human conditions of this completed prediction' : 'Human conditions of the saved case prediction'}</p>${context(detail)}${humanResultNote(detail)}
      <div class="bp-tabs" role="tablist" aria-label="Blueprint content">
        ${[['structure', 'Human physiology template'], ['parameters', 'Parameters'], ['evidence', 'Evidence sources']].map(([key, title]) => `<button type="button" role="tab" data-bp-action="tab" data-tab="${key}" aria-selected="${view.tab === key}" aria-controls="bp-detail-panel">${title}</button>`).join('')}
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
      <h2>Set the prediction scenario</h2><p>Use parameters calibrated to observations to predict the specified dosing regimen.</p>
      ${view.detail ? calibrationBasis(view.referenceCalibration, 'Existing case calibration evidence') : ''}
      <form id="bp-prediction-form">
        ${!form.development ? `<div class="bp-prediction-human"><h3>Human characteristics used for prediction</h3><p>${esc(humanSummary(referenceHuman()) || 'Select a study case first')}</p><p>Predict under the fixed case conditions. Changing human conditions requires matching new observations.</p></div>` : ''}
        <h3 class="bp-secondary-title">Dosing regimen</h3>
        <div class="form-grid"><label>Dose (mg)<input name="dose_mg" type="number" min="0.000001" max="100000" step="any" required value="${esc(form.dose_mg)}"></label>
          <label>Infusion duration (min)<input name="infusion_minutes" type="number" min="0" max="10080" step="any" required value="${esc(form.infusion_minutes)}"><span class="small muted">0 means IV bolus</span></label>
          <label class="full">Prediction end time (h)<input name="window_end_h" type="number" min="0.01" max="168" step="any" required value="${esc(form.window_end_h)}"></label>
        </div>
        <label class="bp-check"><input name="development" type="checkbox" ${form.development ? 'checked' : ''}>Calibrate using new observations</label>
        ${!form.development ? '<p class="bp-calibration-note">Reuse the case calibration parameters. Changing the dosing regimen does not automatically refit parameters.</p>' : ''}
        <div class="bp-development" ${form.development ? '' : 'hidden'}>
          <p>Provide the observed curve and its study conditions. The system rebuilds and calibrates under those human conditions, then predicts the regimen above using the same conditions.</p>
          <fieldset class="bp-human-fields"><legend>Human conditions of the observations</legend>
            <div class="form-grid">
              <label>Age (years)<input name="age_years" type="number" min="18" max="80" step="any" placeholder="Enter the observed study value" value="${esc(form.age_years)}" ${form.development ? 'required' : 'disabled'}></label>
              <label>Sex<select name="sex" ${form.development ? 'required' : 'disabled'}><option value="" ${!form.sex ? 'selected' : ''}>Select the observed study value</option><option value="MALE" ${form.sex === 'MALE' ? 'selected' : ''}>Male</option><option value="FEMALE" ${form.sex === 'FEMALE' ? 'selected' : ''}>Female</option></select></label>
              <label>Body weight (kg)<input name="weight_kg" type="number" min="0.000001" max="300" step="any" placeholder="Enter the observed study value" value="${esc(form.weight_kg)}" ${form.development ? 'required' : 'disabled'}></label>
              <label>Height (cm)<input name="height_cm" type="number" min="0.000001" max="250" step="any" placeholder="Enter the observed study value" value="${esc(form.height_cm)}" ${form.development ? 'required' : 'disabled'}></label>
            </div>
            <p>The current model accepts ages 18–80 years; this range does not establish validated age coverage. Use documented values and complete missing information first.</p>
          </fieldset>
          <label>Observation source<input name="source_label" minlength="3" maxlength="300" placeholder="For example: study name, figure or table number" value="${esc(form.source_label)}" ${form.development ? 'required' : 'disabled'}></label>
          <label>Source URL (optional)<input name="source_url" type="url" placeholder="https://…" value="${esc(form.source_url)}" ${form.development ? '' : 'disabled'}></label>
          <div class="form-grid"><label>Observed curve dose (mg)<input name="development_dose_mg" type="number" min="0.000001" max="100000" step="any" value="${esc(form.development_dose_mg)}" ${form.development ? 'required' : 'disabled'}></label>
            <label>Infusion duration (min)<input name="development_infusion_minutes" type="number" min="0" max="10080" step="any" value="${esc(form.development_infusion_minutes)}" ${form.development ? 'required' : 'disabled'}></label></div>
          <label style="margin-top:13px">Time (h), concentration (ng/mL)<textarea name="observations" spellcheck="false" placeholder="Provide at least 5 observed sampling points, with two columns per line" ${form.development ? 'required' : 'disabled'}>${esc(form.observations)}</textarea></label>
          <label>Or import two columns from a CSV file<input data-bp-input="csv" type="file" accept=".csv,.tsv,.txt,text/csv,text/tab-separated-values" ${form.development ? '' : 'disabled'}></label>
          <p class="bp-human-draft">These form edits have not been submitted. Physiology and results still describe the completed calculation. Fitting new data is not independent validation.</p>
        </div>
        <div class="bp-error" id="bp-form-error" role="alert" hidden></div>
        <button class="primary" type="submit" ${!view.selected || !view.detail || busy ? 'disabled' : ''}>${busy ? 'This job is running…' : 'Generate blueprint and prediction →'}</button>
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
      <h3>${busy ? '<span class="bp-pulse" aria-hidden="true"></span>' : ''}${esc(stage || 'Loading job')}</h3>
      <button type="button" data-bp-action="refresh-job">Refresh status</button></header>
      <p>${failed ? esc(job.error || 'The run failed. Check dosing and development data before submitting again.') : busy ? 'The job continues on the server. You can return or refresh to inspect progress.' : 'Prediction saved. Inspect the curve, PK endpoints, and downloads.'}</p>
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
    return `<div class="bp-downloads">${[['json', 'Blueprint JSON'], ['csv', 'Curve CSV'], ['zip', 'Complete model package']].map(([format, title]) => `<a href="${esc(link(format))}" download>${title} ↓</a>`).join('')}</div>`;
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
    const series = points.length > 1 ? [{ name: 'PBPK prediction', points, kind: 'line' }] : [];
    const chart = series.length && window.PBPKCharts?.curve ? window.PBPKCharts.curve(series, { xLabel: 'Time (h)', yLabel: 'Plasma concentration (ng/mL)' }) : empty(points.length ? 'The chart module is not ready. Refresh the page or download the curve.' : 'No plottable concentration curve is available for this result.');
    const visible = detail.calibration?.visible_points;
    const calibrationPoints = visible != null ? esc(Array.isArray(visible) ? visible.length : visible) : null;
    const calibrationDescription = calibrationPoints != null
      ? detail.human_context?.calibration_mode === 'transferred'
        ? `The original case calibration used ${calibrationPoints} observed sampling points; this run did not recalibrate.`
        : `The current calibration uses ${calibrationPoints} observed sampling points.`
      : '';
    return `<section class="panel bp-result"><div class="bp-card-heading"><div><div class="bp-kicker">PREDICTION RESULT</div>
      <h2>${view.job?.status === 'completed' && view.job.result ? 'Current prediction result' : 'Saved case result'} · ${esc(caseName(detail))}</h2></div>${downloads({ ...detail, download_url: view.job?.status === 'completed' ? view.job.download_url || detail.download_url : detail.download_url })}</div>
      <p class="bp-result-meta">${[dosing.dose_mg != null ? `${number(dosing.dose_mg)} mg` : '', dosing.infusion_minutes != null ? dosing.infusion_minutes === 0 ? 'IV bolus' : `IV infusion ${number(dosing.infusion_minutes)} min` : '', target.window_end_h != null ? `Prediction through ${number(target.window_end_h)} h` : ''].filter(Boolean).map(esc).join(' · ')}</p>
      <p class="bp-result-human">Human conditions used: ${esc(humanSummary(activeHuman(detail)))}</p>${humanResultNote(detail)}
      ${calibrationBasis(detail.calibration)}
      <div class="bp-curve">${chart}</div>
      ${cards.length ? `<div class="bp-pk">${cards.map(row => `<article><h3>${esc(row.title)}</h3><strong>${number(row.value)}</strong><small>${esc(row.unit)}</small></article>`).join('')}</div>` : ''}
      ${unavailable.length ? `<p class="bp-caption">The current curve cannot reliably estimate: ${unavailable.map(([, name]) => esc(name)).join(', ')}. Missing estimates are not replaced by zero.</p>` : ''}
      <p class="bp-caption">These are model predictions. ${calibrationDescription} Execution and independent scientific validation are recorded separately. This prediction does not establish clinical accuracy.</p>
      ${(detail.limitations || []).length ? `<details><summary>Scope and interpretation</summary><ul class="bp-limitations">${detail.limitations.map(item => `<li>${esc(item)}</li>`).join('')}</ul></details>` : ''}
    </section>`;
  }
  function history() {
    return `<section class="panel bp-history"><div class="bp-card-heading"><h2>My runs</h2><button type="button" data-bp-action="refresh-history">Refresh runs ↻</button></div>
      <p class="small muted">Browser-submitted jobs are saved separately. Original research results remain available to inspect and download.</p>
      ${view.history.length ? frameTable(['Compound / scenario', 'Submitted (China Standard Time)', 'Execution status', ''], view.history.map(job => {
        const request = job.request || job.input || {};
        const row = view.cases.find(item => item.case_id === (job.case_id || request.case_id));
        return `<tr><td><strong>${esc(job.drug || request.drug || (row ? caseName(row) : 'Blueprint prediction'))}</strong><small>${request.dose_mg != null ? esc(number(request.dose_mg)) + ' mg' : ''}${request.window_end_h != null ? ' · ' + number(request.window_end_h) + ' h' : ''}</small>${request.human ? `<small>${esc(humanSummary(request.human))}</small>` : ''}</td>
          <td>${esc(date(job.created_at))}</td><td>${status(job.status)}</td>
          <td><button type="button" data-bp-action="open-job" data-job="${esc(job.job_id)}">${job.status === 'completed' ? 'View result ↗' : 'View progress ↗'}</button></td></tr>`;
      })) : '<p class="bp-caption">No new browser jobs have been submitted. Inspect the completed cases above.</p>'}</section>`;
  }
  function content() {
    const unsupported = view.requestedDrug && !view.cases.some(row => caseName(row).toLowerCase() === view.requestedDrug.toLowerCase());
    return `<header class="bp-intro"><div><p class="eyebrow">EVIDENCE → BLUEPRINT → PREDICTION</p><h1>Blueprint design and prediction</h1><p>Drug evidence and human physiology → Parameter calibration → Feasibility checks and repair → Prediction under target conditions.</p></div><div class="bp-intro-count"><strong>${view.cases.length}</strong>Executable study cases</div></header>
      ${unsupported ? `<div class="bp-unavailable">${esc(view.requestedDrug)}: complete executable inputs are unavailable. Select an existing case below; compounds will not be substituted automatically.</div>` : ''}
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
      if (fields.length !== 2 || fields.some(value => value === '' || !Number.isFinite(Number(value)))) throw new Error(`Development curve row ${index + 1} must contain two numeric columns: time (h), concentration (ng/mL).`);
      const [time_h, concentration_ng_ml] = fields.map(Number);
      if (time_h < 0 || time_h > 168 || concentration_ng_ml <= 0 || concentration_ng_ml > 1e9) throw new Error('Sampling time must be within 0–168 h; concentration must be greater than 0 and at most 10⁹ ng/mL.');
      points.push({ time_h, concentration_ng_ml });
    });
    if (points.length < 5) throw new Error('Provide at least 5 development sampling points.');
    if (points.length > 128) throw new Error('Provide no more than 128 development sampling points.');
    if (points.some((row, i) => i && row.time_h <= points[i - 1].time_h)) throw new Error('Sampling times must increase strictly without duplicates.');
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
        if (!disposed && field) { field.hidden = false; field.textContent = 'Status is temporarily unavailable: ' + error.message; }
      } finally { polling = false; schedule(); }
    }
    async function selectCase() {
      const version = ++loadVersion;
      clearTimeout(timer);
      view.job = null; view.form = null; view.detail = null; view.referenceHuman = null; view.referenceCalibration = null; remember();
      replace('case', '<section class="panel"><div class="loading">Loading model blueprint…</div></section>');
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
          if (file.size > 200000) throw new Error('The development-curve file must be smaller than 200 KB.');
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
        if (!view.selected) throw new Error('Select an existing compound case first.');
        const payload = { case_id: view.selected, regime: view.regime, dose_mg: Number(form.dose_mg), infusion_minutes: Number(form.infusion_minutes), window_end_h: Number(form.window_end_h) };
        if (payload.infusion_minutes / 60 >= payload.window_end_h) throw new Error('The prediction window must end after the infusion finishes.');
        if (form.development) {
          if (humanFields.some(key => form[key] == null || String(form[key]).trim() === '')) throw new Error('Enter the age, sex, weight, and height recorded in the study.');
          const human = { age_years: Number(form.age_years), sex: form.sex, weight_kg: Number(form.weight_kg), height_cm: Number(form.height_cm) };
          if (!Number.isFinite(human.age_years) || human.age_years < 18 || human.age_years > 80) throw new Error('Observed age must fall within the current construction range of 18–80 years; this range does not establish validation coverage.');
          if (!['MALE', 'FEMALE'].includes(human.sex)) throw new Error('Enter the sex recorded in the observation study.');
          if (!Number.isFinite(human.weight_kg) || human.weight_kg <= 0 || human.weight_kg > 300) throw new Error('Recorded body weight must be greater than 0 and at most 300 kg.');
          if (!Number.isFinite(human.height_cm) || human.height_cm <= 0 || human.height_cm > 250) throw new Error('Recorded height must be greater than 0 and at most 250 cm.');
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
