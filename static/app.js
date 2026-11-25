'use strict';
const $ = id => document.getElementById(id),
    fmt = (v, n = 1) => v == null || !Number.isFinite(+v) ? '—' : (+v).toFixed(n),
    avg = a => a.length ? a.reduce((s, x) => s + (+x), 0) / a.length : NaN,
    mean = (r, k) => avg(r.map(x => x[k]).filter(x => x != null)),
    pct = v => fmt(v * 100, 1) + '%',
    esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let rows = [],
    stats = [],
    audit = {},
    current = null,
    habits = null,
    researchData = null,
    clusterData = null;
const colors = ['#3987df', '#f07b48', '#24b68c', '#a18cf2', '#e7ba62'];
async function api(path, method = 'GET', body) {
    const r = await fetch(path, {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body: body ? JSON.stringify(body) : undefined
    });
    if (!r.ok) {
        let m = await r.json().catch(() => ({ detail: r.statusText }));
        throw Error(typeof m.detail === 'string' ? m.detail : JSON.stringify(m.detail));
    }
    return r.json();
}
async function task(fn) {
    $('status').textContent = '';
    try {
        await fn();
    }
    catch (e) {
        $('status').textContent = e.message;
        console.error(e);
    }
}
// Explicit axis titles and units for every Cartesian chart.
const axisLabels = {
    heatmap: ['Productivity category', 'Daily phone time (hours)'],
    appsChart: ['Reported activity', 'Mean reported time (hours/day)'],
    trend: ['Daily phone time group (hours)', 'Mean score (0–100)'],
    sleepChart: ['Daily phone time (hours)', 'Sleep duration (hours)'],
    overviewHist: ['Daily phone time (hours)', 'Number of students'],
    overviewScatter: ['Daily phone time (hours)', 'Productivity score (0–100)'],
    overviewBox: ['Daily phone time group (hours)', 'Productivity score (0–100)'],
    globalImportance: ['Mean absolute SHAP (risk probability)', 'Model input feature'],
    hist: ['Daily phone time (hours)', 'Number of students'],
    boxplot: ['Daily phone time group (hours)', 'Productivity score (0–100)'],
    notifChart: ['Social media use (hours/day)', 'Productivity score (0–100)'],
    checkChart: ['Gaming (hours/day)', 'Productivity score (0–100)'],
    corr: ['Measured variable', 'Measured variable'],
    explanation: ['SHAP contribution to low-productivity risk', 'Model input feature'],
    beforeAfter: ['Daily phone time group (hours)', 'Mean productivity (0–100)'],
    probChart: [null, 'Frequency (simulated observations)'],
    clusterPopulation: ['Behavioural group', 'Number of students'],
    clusterHabits: ['Reported habit', 'Mean time (hours/day)'],
    roc: ['False positive rate (0–1)', 'True positive rate (0–1)'],
    bias: ['Decision tree maximum depth', 'Macro F1 score (0–1)'],
    confusion: ['Predicted productivity class', 'Actual productivity class'],
    errorTrend: ['Prediction date (UTC)', 'Mean absolute error (score points)'],
    drift: ['Model input feature', 'Absolute mean shift (training SD units)']
};
const featureLabels = {
    screen_hours: 'Phone time (h/day)',
    social_hours: 'Social media (h/day)',
    video_hours: 'YouTube (h/day)',
    gaming_hours: 'Gaming (h/day)',
    study_hours: 'Study (h/day)',
    sleep_hours: 'Sleep (h/day)',
    productivity: 'Productivity (0–100)',
    focus_score: 'Focus (0–100)',
    stress: 'Stress (1–10)',
    breaks: 'Breaks (count/day)',
    exercise: 'Exercise (min/day)',
    age: 'Age (years)',
    gender: 'Gender'
};
const featureLabel = key => featureLabels[key] || key.replaceAll('_', ' ');

function chartAxis(axis = {}, label) {
    const existing = axis.title;
    const titleText = label || (typeof existing === 'string' ? existing : existing?.text);
    return {
        gridcolor: '#e3e8ee',
        zerolinecolor: '#b8c4d0',
        showline: true,
        linecolor: '#8493a3',
        automargin: true,
        ...axis,
        ...(titleText ? {
            title: {
                ...(typeof existing === 'object' ? existing : {}),
                text: titleText,
                standoff: 10
            }
        } : {})
    };
}

function chart(id, traces, layout = {}) {
    if (!$(id)?.offsetWidth)
        return;

    // Plotly 3 requires title.text, rather than a bare title string.
    // Preserve the individual chart's ranges, tick angles, and axis settings.
    const finalLayout = {
        height: 360,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'system-ui', size: 12, color: '#23364a' },
        colorway: colors,
        legend: { orientation: 'h', y: 1.12 },
        ...layout,
        margin: {
            l: 78, r: 32, t: 65, b: 85,
            ...layout.margin
        }
    };
    const labels = axisLabels[id];
    if (labels) {
        finalLayout.xaxis = chartAxis(layout.xaxis, labels[0]);
        finalLayout.yaxis = chartAxis(layout.yaxis, labels[1]);
    }
    // Label every scatter-matrix panel.
    for (const key of Object.keys(finalLayout)) {
        if (/^[xy]axis[0-9]*$/.test(key) && !(labels && (key === 'xaxis' || key === 'yaxis'))) {
            finalLayout[key] = chartAxis(finalLayout[key]);
        }
    }
    // Keep individual histogram bins visually distinguishable.
    const styledTraces = traces.map(trace => trace.type === 'histogram' ? {
        ...trace,
        marker: {
            ...trace.marker,
            line: { color: '#83b8f7', width: 1 }
        }
    } : trace);
    Plotly.react(id, styledTraces, finalLayout, {
        responsive: true,
        displayModeBar: false
    });
}
function metric(id, items) {
    $(id).innerHTML = items.map(([k, v]) => `<div class="metric"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`).join('');
}
function table(id, headers, data) {
    $(id).innerHTML = '<div class="tablewrap"><table><thead><tr>' + headers.map(x => `<th>${esc(x)}</th>`).join('') + '</tr></thead><tbody>' + data.map(row => '<tr>' + row.map(x => `<td>${esc(x)}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
}
const ageGroups = ['17–20', '21–25', '26–29'];
const ageColors = ['#2563eb', '#b45309', '#087e67'];
function ageTraces(sample, x, y, axis = '', showlegend = true) {
    return ageGroups.map((age, i) => {
        const group = sample.filter(r => r.age_group === age);
        return {
            x: group.map(r => r[x]), y: group.map(r => r[y]),
            type: 'scatter', mode: 'markers', name: `Age ${age}`,
            legendgroup: age, showlegend, xaxis: 'x' + axis, yaxis: 'y' + axis,
            marker: { size: 6, opacity: .55, color: ageColors[i], symbol: ['circle', 'diamond', 'square'][i] },
            hovertemplate: `Age ${age}<br>${featureLabel(x)}: %{x}<br>${featureLabel(y)}: %{y}<extra></extra>`
        };
    });
}
function scatter(id, x, y, xlabel, ylabel = 'Productivity score (0–100)') {
    const sample = rows.filter((r, i) => i % Math.max(1, Math.ceil(rows.length / 1800)) === 0);
    chart(id, ageTraces(sample, x, y), {
        showlegend: true, xaxis: { title: xlabel }, yaxis: { title: ylabel }
    });
}
function corr(a, b) {
    let pairs = rows.filter(r => r[a] != null && r[b] != null),
        x = mean(pairs, a),
        y = mean(pairs, b),
        s = 0,
        sx = 0,
        sy = 0;
    for (const p of pairs) {
        const dx = p[a] - x, dy = p[b] - y;
        s += dx * dy;
        sx += dx * dx;
        sy += dy * dy;
    }
    return sx * sy ? s / Math.sqrt(sx * sy) : null;
}
function drawOverview() {
    metric('kpis', [
        ['Student records', rows.length],
        ['Avg. phone time', fmt(mean(rows, 'screen_hours')) + ' h'],
        ['Avg. productivity', fmt(mean(rows, 'productivity')) + '/100'],
        ['Avg. focus', fmt(mean(rows, 'focus_score')) + '/100'],
        ['Avg. sleep', fmt(mean(rows, 'sleep_hours')) + ' h']
    ]);
    const xb = ['0–2', '2–4', '4–6', '6–8', '8–10', '10+'],
        yb = ['0–20', '20–40', '40–60', '60–80', '80–100'],
        z = Array.from({ length: 5 }, () => Array(6).fill(0));
    rows.forEach(r => z[Math.min(4, Math.floor(r.productivity / 20))][Math.min(5, Math.floor(r.screen_hours / 2))]++);
    chart('heatmap',
        [
        {
            x: xb,
            y: yb,
            z,
            type: 'heatmap',
            colorscale: [[0, '#f0f5ed'], [1, '#197766']],
            hovertemplate: 'Phone %{x} h<br>Productivity %{y}<br>Count %{z}<extra></extra>'
        }
    ],
        { xaxis: { title: 'Phone hours' }, yaxis: { title: 'Productivity score' } });
    chart('appsChart',
        [
        {
            type: 'bar',
            x: ['Social media', 'YouTube', 'Gaming', 'Study'],
            y: ['social_hours', 'video_hours', 'gaming_hours', 'study_hours'].map(x => mean(rows, x)),
            marker: { color: colors.slice(0, 4) }
        }
    ],
        { yaxis: { title: 'Mean reported hours' } });
    const groups = xb.map((_, i) => rows.filter(r => Math.min(5, Math.floor(r.screen_hours / 2)) === i));
    chart('trend',
        [
        { x: xb,
            y: groups.map(r => mean(r, 'productivity')),
            name: 'Productivity',
            mode: 'lines+markers' }, { x: xb,
            y: groups.map(r => mean(r, 'focus_score')),
            name: 'Focus',
            mode: 'lines+markers' }
    ],
        { xaxis: { title: 'Phone hours (not time series)' }, yaxis: { title: 'Mean score /100' } });
    scatter('sleepChart', 'screen_hours', 'sleep_hours', 'Phone hours', 'Sleep hours');
    const a = rows.filter(r => r.screen_hours < 4),
        b = rows.filter(r => r.screen_hours >= 6),
        low = rows.filter(r => r.sleep_hours < 7),
        high = rows.filter(r => r.sleep_hours >= 7);
    const findings = rows.length ? [
        `Students below 4 phone hours averaged ${fmt(mean(a, 'productivity'))} productivity (n=${a.length}) versus ${fmt(mean(b, 'productivity'))} at 6+ hours (n=${b.length}).`,
        `Students reporting under 7 hours of sleep averaged ${fmt(mean(low, 'productivity'))} productivity versus ${fmt(mean(high, 'productivity'))} for 7+ hours.`,
        `Social media hours and productivity have a Pearson correlation of ${fmt(corr('social_hours', 'productivity'), 2)}.`,
        `Study hours and productivity have a Pearson correlation of ${fmt(corr('study_hours', 'productivity'), 2)}.`,
        `Stress and focus score have a Pearson correlation of ${fmt(corr('stress', 'focus_score'), 2)}.`
    ] : ['No records match the selected filters.'];
    $('insights').innerHTML = findings.map((s, i) => `<div class="finding"><b>${i + 1}.</b> ${esc(s)}</div>`).join('');
}
async function loadData() {
    const query = new URLSearchParams({
        age: $('filterAge').value,
        style: $('filterStyle').value,
        minimum: $('filterMin').value,
        maximum: $('filterMax').value
    });
    if (+$('filterMin').value > +$('filterMax').value)
        throw Error('Minimum must not exceed maximum.');
    const d = await api('/api/data?' + query);
    rows = d.rows;
    stats = d.stats;
    audit = d.audit;
    drawOverview();
    if (typeof drawDashboard === 'function') {
        drawDashboard();
        loadOverviewExtras();
    }
}
function drawAnalysis() {
    metric('audit', Object.entries(audit).map(([k, v]) => [k.replaceAll('_', ' '), v]));
    table('statsTable',
        ['Feature', 'Mean', 'Median', 'Variance', 'SD', 'Min', 'Max', 'IQR flags'],
        stats.map(r => [r.feature,
        fmt(r.mean),
        fmt(r.median),
        fmt(r.variance),
        fmt(r.std),
        fmt(r.min),
        fmt(r.max),
        r.outliers]));
    chart('hist', [{ x: rows.map(r => r.screen_hours), type: 'histogram', nbinsx: 20 }], { xaxis: { title: 'Screen hours' } });
    chart('boxplot',
        [[0, 4], [4, 6], [6, 25]].map(([lo, hi]) => ({
        type: 'box',
        name: `${lo}–${hi === 25 ? '24' : hi} h`,
        y: rows.filter(r => r.screen_hours >= lo && r.screen_hours < hi).map(r => r.productivity)
    })),
        { yaxis: { title: 'Productivity score' } });
    scatter('notifChart', 'social_hours', 'productivity', 'Social media hours');
    scatter('checkChart', 'gaming_hours', 'productivity', 'Gaming hours');
    const keys = [
        'screen_hours',
        'productivity',
        'sleep_hours',
        'social_hours',
        'gaming_hours',
        'study_hours',
        'stress',
        'focus_score'
    ];
    chart('corr',
        [
        {
            type: 'heatmap',
            x: keys.map(featureLabel),
            y: keys.map(featureLabel),
            z: keys.map(a => keys.map(b => corr(a, b))),
            zmin: -1,
            zmax: 1,
            colorscale: 'RdBu',
            reversescale: true,
            colorbar: { title: { text: 'Pearson r' } }
        }
    ],
        { height: 470, xaxis: { tickangle: -35 }, margin: { l: 145, r: 55, t: 35, b: 145 } });
    // The six unique pairs use SVG scatter traces instead of WebGL,
    // so they are visible even on computers without GPU acceleration.
    const pairKeys = keys.slice(0, 4);
    const sample = rows.filter((r, i) => i % Math.max(1, Math.ceil(rows.length / 450)) === 0);
    const pairTraces = [];
    const pairLayout = {
        grid: { rows: 3, columns: 2, pattern: 'independent', xgap: 0.22, ygap: 0.32 },
        height: 880,
        margin: { l: 85, r: 30, t: 25, b: 65 },
        showlegend: true, legend: { orientation: 'h', y: 1.05 }
    };
    let panel = 0;
    for (let i = 1; i < pairKeys.length; i++) {
        for (let j = 0; j < i; j++) {
            const x = pairKeys[j], y = pairKeys[i];
            const n = ++panel;
            const suffix = n === 1 ? '' : String(n);
            pairTraces.push(...ageTraces(sample, x, y, suffix, n === 1));
            pairLayout['xaxis' + suffix] = chartAxis({}, featureLabel(x));
            pairLayout['yaxis' + suffix] = chartAxis({}, featureLabel(y));
        }
    }
    chart('pairplot', pairTraces, pairLayout);
}
const fields = [
    ['screen_hours', 'Daily phone hours', 6, 24, .1],
    ['social_hours', 'Social media hours', 2, 24, .1],
    ['video_hours', 'YouTube hours', 1, 24, .1],
    ['gaming_hours', 'Gaming hours', .5, 24, .1],
    ['study_hours', 'Study hours', 5, 24, .1],
    ['sleep_hours', 'Sleep hours', 7, 24, .1],
    ['breaks', 'Breaks per day', 7, 30, 1],
    ['exercise', 'Exercise minutes', 60, 300, 1],
    ['stress', 'Stress (1–10)', 5, 10, 1],
    ['age', 'Age', 23, 29, 1],
    ['night_minutes', 'Night usage minutes (optional)', null, 180, 1],
    ['notifications', 'Notifications (optional)', null, 2000, 1],
    ['checks', 'Phone checks (optional)', null, 2000, 1]
];
$('habitFields').innerHTML = fields.map(([k, label, v, max, step]) => `<label>${label}<input name="${k}" type="number" value="${v ?? ''}" min="${k === 'age' ? 17 : k === 'stress' ? 1 : 0}" max="${max}" step="${step}" ${v === null ? '' : 'required'}></label>`).join('') + '<label>Gender<select name="gender"><option>Male</option><option>Female</option><option>Other</option></select></label>';
function drawProfile() {
    if (!current)
        return;
    for (const id of ['heroGauge', 'profileGauge'])
        $(id).style.setProperty('--score', current.wellness);
    $('heroScore').textContent = $('profileScore').textContent = current.wellness;
    $('heroPersona').textContent = $('persona').textContent = current.persona;
    $('profileResult').innerHTML = `<p>More screen time than <b>${fmt(current.percentile, 0)}%</b> of students in the uploaded dataset.</p><p>Estimated productivity: <b>${fmt(current.predicted_focus, 0)}/100</b> · Predicted class: <b>${esc(current.category)}</b> · Low / Normal / High: ${["Low","Normal","High"].map(k => pct(current.probabilities[k])).join(" / ")}</p><p>Distraction index: ${fmt(current.distraction, 0)}/100 · Prediction ID: ${current.prediction_id}</p><p class="note">Wellness score is a heuristic. Model estimates use the uploaded dataset; its provenance is unverified. Optional notification/night/check values are not model inputs.${current.out_of_range.length ? ' Outside training range: ' + esc(current.out_of_range.join(', ')) : ''}</p>`;
    for (const id of ['pdf', 'explain', 'simulate'])
        $(id).disabled = false;
    if (typeof syncDashboard === 'function')
        syncDashboard();
}
$('habits').onsubmit = e => {
    e.preventDefault();
    task(async () => {
        const v = Object.fromEntries(new FormData(e.target));
        fields.forEach(([k]) => v[k] = v[k] === '' ? null : +v[k]);
        const p = await api('/api/profile', 'POST', v);
        habits = v;
        current = p;
        drawProfile();
        $('scenarioResult').textContent = '';
        if ($('explanation')._fullLayout) Plotly.purge('explanation');
        $('explanation').innerHTML = '';
        $('explanation').classList.remove('plot-requested');
    });
};
$('pdf').onclick = () => task(async () => {
    const r = await fetch('/api/report', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(habits) });
    if (!r.ok)
        throw Error('PDF export failed');
    const u = URL.createObjectURL(await r.blob()), a = document.createElement('a');
    a.href = u;
    a.download = 'focusiq-report.pdf';
    a.click();
    setTimeout(() => URL.revokeObjectURL(u), 1000);
});
$('explain').onclick = () => task(async () => {
    const b = $('explain');
    b.disabled = true;
    b.textContent = 'Computing SHAP…';
    try {
        const x = await api('/api/explain', 'POST', habits);
        $('explanation').classList.add('plot-requested');
        chart('explanation',
            [
            {
                type: 'bar',
                orientation: 'h',
                x: x.map(r => r.contribution),
                y: x.map(r => featureLabel(r.feature)),
                marker: { color: x.map(r => r.contribution > 0 ? '#bd6e57' : '#197766') }
            }
        ],
            {
            margin: { l: 150, r: 30, t: 35, b: 85 },
            xaxis: { title: 'Contribution to low-productivity probability' },
            yaxis: { autorange: 'reversed' }
        });
    }
    finally {
        b.disabled = false;
        b.textContent = 'Explain prediction (SHAP)';
    }
});
$('simulate').onclick = () => task(async () => {
    const v = { ...habits }, cut = Math.min(v.social_hours, +$('cutSocial').value / 60);
    v.social_hours -= cut;
    v.screen_hours = Math.max(0, v.screen_hours - cut);
    if (v.night_minutes !== null)
        v.night_minutes = Math.max(0, Math.min(v.screen_hours * 60, v.night_minutes - +$('cutNight').value));
    const p = await api('/api/scenario', 'POST', v);
    $('scenarioResult').innerHTML = `<p>Wellness: <b>${current.wellness} → ${p.wellness}</b> · Estimated productivity: <b>${fmt(current.predicted_focus)} → ${fmt(p.predicted_focus)}</b> · Low-focus probability: ${pct(current.risk)} → ${pct(p.risk)}</p><p class="note">Scenario assumes saved social time reduces phone use; optional night usage changes wellness only, not the ML prediction.</p>`;
});
for (const id of ['cutSocial', 'cutNight'])
    $(id).oninput = () => $(id + 'Val').textContent = $(id).value;
function drawResearch() {
    let r = researchData, alpha = +$('sig').value;
    metric('researchKpis', [
        ['Student records', r.baseline_users],
        ['P(low productivity | ≥6h)', pct(r.conditional)],
        ['High usage group', r.conditional_n],
        ['Mean score 95% CI', r.bootstrap_ci.map(x => fmt(x)).join(' – ')]
    ]);
    chart('beforeAfter',
        [
        {
            type: 'bar',
            x: ['<4h', '4–<6h', '≥6h'],
            y: r.group_means,
            text: r.group_counts.map(n => 'n=' + n),
            textposition: 'auto'
        }
    ],
        { yaxis: { title: 'Mean productivity' } });
    table('researchTable',
        ['Test', 'Research question', 'Statistic', 'p-value', 'Decision at α'],
        r.tests.map(t => [
        t.test,
        t.question,
        fmt(t.stat, 3),
        t.p == null ? '—' : t.p.toExponential(3),
        t.p == null ? 'Not estimable' : t.p < alpha ? 'Reject H₀' : 'Do not reject H₀'
    ]));
    $('researchDetails').textContent = 'Exploratory tests assume independent student records; collection methodology unverified. No causal or before/after claims can be made from this dataset.';
}
$('sig').onchange = () => researchData && drawResearch();
function probHelp() {
    $('paramHelp').textContent = {
        Normal: 'Parameter = SD (mean fixed at 60).',
        Binomial: 'Parameter = probability percent (7 trials).',
        Poisson: 'Parameter = expected count per interval.',
        Uniform: 'Parameter = upper bound in minutes.'
    }[$('dist').value];
}
;
$('dist').onchange = probHelp;
probHelp();
async function drawProb() {
    const p = await api('/api/probability?' + new URLSearchParams({
        kind: $('dist').value,
        param: $('param').value,
        n: $('sample').value,
        alpha: $('alpha').value,
        effect: $('effect').value
    }));
    chart('probChart', [{ type: 'histogram', x: p.values }], { xaxis: { title: p.label } });
    metric('probKpis', [
        ['Sample mean', fmt(p.mean, 2)],
        ['Sample SD', fmt(p.sd, 2)],
        ['Type I error', pct(p.type1)],
        ['Type II error', pct(p.type2)],
        ['Power', pct(p.power)]
    ]);
}
$('runProb').onclick = () => task(drawProb);
function drawClusters() {
    const d = clusterData;
    const method = $('clusterMethod').value;
    const methodName = method === 'k' ? 'K-Means' : 'Hierarchical';
    const groups = d.profiles[method];
    const labels = groups.map(r => `Cluster ${r.cluster + 1}: ${r.name}`);
    $('clusterCards').innerHTML = groups.map((r, i) => `
        <div class="cluster-card" style="--cluster:${colors[i]}">
            <strong>Cluster ${r.cluster + 1} · ${esc(r.name)}</strong>
            <b>${r.count} students · ${pct(r.share)}</b>
            <span>Phone ${fmt(r.screen_hours)} h · Study ${fmt(r.study_hours)} h · Sleep ${fmt(r.sleep_hours)} h</span>
        </div>`).join('');
    chart('clusterPopulation', [{ type: 'bar', x: groups.map(r => 'Cluster ' + (r.cluster + 1)),
        y: groups.map(r => r.count), marker: { color: colors.slice(0, 4) },
        text: groups.map(r => `${r.count} (${pct(r.share)})`), textposition: 'auto',
        customdata: labels, hovertemplate: '%{customdata}<br>%{y} students<extra></extra>' }],
        { title: { text: methodName + ' — group sizes' }, showlegend: false });
    const keys = ['screen_hours','social_hours','video_hours','gaming_hours','study_hours','sleep_hours'];
    chart('clusterHabits', groups.map((r, i) => ({ type: 'bar', name: 'Cluster ' + (i+1),
        x: keys.map(featureLabel), y: keys.map(k => r[k]), marker: { color: colors[i] } })),
        { barmode: 'group', height: 460, margin: { t: 100, b: 100 }, xaxis: { tickangle: -20 } });
    table('clusterProfiles', ['Group', 'Behavioural profile', 'Students', 'Share', 'Phone h', 'Social h', 'YouTube h', 'Gaming h', 'Study h', 'Sleep h'],
        groups.map((r, i) => [i + 1, r.name, r.count, pct(r.share), ...keys.map(k => fmt(r[k]))]));
    metric('clusterQuality', [['K-Means silhouette', fmt(d.silhouette.k, 3)], ['Hierarchical silhouette', fmt(d.silhouette.h, 3)], ['Sample size', d.sample_size]]);
    $('pcaVariance').textContent = 'PCA: first two components retain ' + pct(d.variance.reduce((a,b) => a+b, 0)) + ' of the standardized variance.';
}
$('clusterMethod').onchange = () => clusterData && drawClusters();
let modelData = null;
function drawSelectedModel() {
    const m = modelData;
    if (!m) return;
    const r = m.models.find(x => x.model === $('modelChoice').value) || m.models.find(x => x.model === m.winner);
    metric('selectedMetrics', [['Test accuracy', pct(r.accuracy)], ['Macro F1', fmt(r.f1, 3)], ['Macro recall', fmt(r.recall, 3)], ['Macro precision', fmt(r.precision, 3)], ['Macro ROC AUC', fmt(r.roc_auc, 3)]]);
    chart('confusion', [{ type: 'heatmap', z: r.confusion, x: m.class_names, y: m.class_names,
        colorscale: 'Blues', colorbar: { title: { text: 'Students' } }, texttemplate: '%{z}',
        hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Students: %{z}<extra></extra>' }],
        { title: { text: r.model }, yaxis: { autorange: 'reversed' } });
    const traces = r.roc.map((c, i) => ({ type: 'scatter', mode: 'lines', x: c.fpr, y: c.tpr,
        name: `${c.class} (AUC ${fmt(c.auc, 3)})`, line: { color: colors[i], width: 2 } }));
    traces.push({ type: 'scatter', mode: 'lines', x: [0,1], y: [0,1], name: 'Random baseline', line: { color: '#64748b', dash: 'dash' } });
    chart('roc', traces, { title: { text: r.model }, height: 420, margin: { t: 100 },
        xaxis: { range: [0,1] }, yaxis: { range: [0,1.02] }, legend: { orientation: 'h', y: 1.16 } });
}
$('modelChoice').onchange = drawSelectedModel;
function drawModels(m) {
    modelData = m;
    $('modelSummary').innerHTML = `<p><b>Deployed: ${esc(m.winner)}</b> · Version ${esc(m.version)} · ${m.train_users} training / ${m.test_users} test students</p><p>Class counts: ${m.class_names.map((n,i) => esc(n) + ' ' + m.class_counts[i]).join(' · ')}</p><p>Linear regression: MAE ${fmt(m.regression.mae)}, RMSE ${fmt(m.regression.rmse)}, R² ${fmt(m.regression.r2, 3)}</p><p class="note">Grid search: ${esc(JSON.stringify(m.grid_params))} · Random search: ${esc(JSON.stringify(m.random_params))}</p>`;
    table('modelTable', ['Model', 'CV macro F1', 'Test accuracy', 'Macro precision', 'Macro recall', 'Macro F1', 'Macro ROC AUC'],
        m.models.map(r => [r.model + (r.model === m.winner ? ' ★' : ''), fmt(r.cv_f1,3), pct(r.accuracy), fmt(r.precision,3), fmt(r.recall,3), fmt(r.f1,3), fmt(r.roc_auc,3)]));
    const previous = $('modelChoice').value;
    $('modelChoice').replaceChildren(...m.models.map(r => new Option(r.model, r.model)));
    $('modelChoice').value = m.models.some(r => r.model === previous) ? previous : m.winner;
    drawSelectedModel();
    chart('bias', [
        { x: m.bias.map(x => x.depth), y: m.bias.map(x => x.train), name: 'Training macro F1', mode: 'lines+markers' },
        { x: m.bias.map(x => x.depth), y: m.bias.map(x => x.cv), name: 'Validation macro F1', mode: 'lines+markers' }
    ], { xaxis: { type: 'category' }, yaxis: { range: [0,1.05] } });
}
$('retrain').onclick = () => task(async () => {
    let b = $('retrain');
    b.disabled = true;
    b.textContent = 'Training models…';
    try {
        drawModels(await api('/api/retrain', 'POST'));
    }
    finally {
        b.disabled = false;
        b.textContent = 'Retrain and save new model version';
    }
});
let monitorPredictions = [];

function selectedPrediction() {
    return monitorPredictions.find(r => String(r.id) === $('predID').value);
}

function showPrediction() {
    const record = selectedPrediction();
    $('actualFocus').value = record?.actual ?? '';
    $('actualFocus').disabled = !record;
    $('saveActual').disabled = !record;
    $('predictionDetails').textContent = !record
        ? 'Choose a saved prediction to see its details.'
        : `Predicted productivity: ${fmt(record.predicted)}/100 · Predicted class: ${record.category || "Legacy prediction"} · Chance of low productivity: ${pct(record.risk)}. ` +
          (record.actual == null ? 'Actual score not recorded yet.' : `Recorded actual score: ${fmt(record.actual)}/100 · Absolute error: ${fmt(Math.abs(record.actual - record.predicted))} points.`);
}

$('predID').onchange = showPrediction;

async function drawMonitor() {
    const m = await api('/api/monitor');
    metric('monitorKpis', [
        ['Predictions (active model)', m.current_count],
        ['Actual scores recorded (active model)', m.labeled],
        ['Average absolute error (points)', m.mae == null ? 'Not available' : fmt(m.mae)],
        ['Saved model versions', m.versions.length]
    ]);
    chart('drift',
        [{ type: 'bar', x: Object.keys(m.drift).map(featureLabel), y: Object.values(m.drift),
           hovertemplate: '%{x}<br>Mean shift: %{y:.2f} training SD<extra></extra>' }],
        { xaxis: { tickangle: -35 },
          margin: { l: 90, r: 30, t: 40, b: 135 }, height: 420,
          ...(m.current_count ? {} : { annotations: [{ text: 'No saved predictions yet', x: .5, y: .5, xref: 'paper', yref: 'paper', showarrow: false }] }) });
    const daily = m.daily_errors || [];
    chart('errorTrend', [
        { type: 'scatter', mode: 'lines+markers', x: daily.map(r => r.day), y: daily.map(r => r.error), name: 'Daily MAE' },
        { type: 'scatter', mode: 'lines+markers', x: daily.map(r => r.day), y: daily.map(r => r.rolling), name: '3-observed-day rolling mean', line: { dash: 'dash' } }
    ], { xaxis: { type: 'date' }, yaxis: { rangemode: 'tozero' },
        ...(daily.length ? {} : { annotations: [{ text: 'No evaluated predictions yet', x: .5, y: .5, xref: 'paper', yref: 'paper', showarrow: false }] }) });
    $('errorTrendNote').textContent = daily.length < 3 ? 'Record actual outcomes on at least three different prediction dates to see the rolling mean.' : 'The rolling mean is a descriptive monitoring trend, not a forecast of student habits.';
    monitorPredictions = m.predictions;
    const previous = $('predID').value;
    $('predID').replaceChildren(new Option('Select a saved prediction', ''));
    for (const r of monitorPredictions) {
        $('predID').append(new Option(
            `#${r.id} · ${r.created ?? 'Date unavailable'} · Predicted ${fmt(r.predicted)}/100${r.actual == null ? ' (awaiting actual score)' : ' (actual recorded)'}`,
            String(r.id)
        ));
    }
    $('predID').disabled = !monitorPredictions.length;
    if (monitorPredictions.some(r => String(r.id) === previous)) {
        $('predID').value = previous;
    } else {
        const pending = monitorPredictions.find(r => r.actual == null) || monitorPredictions[0];
        if (pending) $('predID').value = String(pending.id);
    }
    showPrediction();
    $('monitorEmpty').textContent = monitorPredictions.length ? '' : 'No saved predictions yet. Open My wellness and submit the personal assessment to create your first prediction.';
    if (monitorPredictions.length) {
        table('monitorTable',
            ['Prediction ID', 'Created (UTC)', 'Predicted score /100', 'Predicted class', 'Chance of score below 40', 'Actual score /100', 'Absolute error (points)', 'Status', 'Model version'],
            monitorPredictions.map(r => [r.id, r.created, fmt(r.predicted), r.category || "Legacy", pct(r.risk),
                r.actual == null ? 'Not recorded' : fmt(r.actual),
                r.actual == null ? '—' : fmt(Math.abs(r.actual - r.predicted)),
                r.actual == null ? 'Awaiting actual score' : 'Evaluated', r.version]));
    } else {
        $('monitorTable').innerHTML = '';
    }
    $('versions').textContent = 'Active model: ' + m.current + ' | Saved versions: ' + m.versions.join(', ');
    $('driftNote').textContent = m.current_count
        ? 'The bar height measures how far the average input differs from its training average, in training standard deviations. A larger bar means a larger change, not necessarily a prediction error.'
        : 'Create a prediction in My wellness before input drift can be calculated.';
}

$('actualForm').onsubmit = e => {
    e.preventDefault();
    task(async () => {
        const record = selectedPrediction();
        if (!record) throw Error('Select a saved prediction first.');
        if ($('actualFocus').value === '') throw Error('Enter an actual productivity score.');
        await api('/api/predictions/' + record.id, 'PATCH', { actual: +$('actualFocus').value });
        await drawMonitor();
        $('monitorFeedback').textContent = `Actual score saved for prediction #${record.id}. The prediction error and average error have been updated.`;
    });
};
const titles = {
    overview: 'Your digital habits, decoded.',
    profile: 'Know your digital self.',
    analysis: 'Explore the evidence.',
    research: 'Test statistical relationships.',
    probability: 'Understand uncertainty.',
    clusters: 'Find behavioural patterns.',
    models: 'Compare, tune, explain.',
    monitor: 'Keep your models accountable.'
};
async function go(page) {
    document.querySelectorAll('.page').forEach(x => x.classList.toggle('active', x.id === page));
    document.querySelectorAll('nav button').forEach(x => x.classList.toggle('active', x.dataset.page === page));
    $('heading').textContent = titles[page];
    window.scrollTo(0, 0);
    await task(async () => {
        if (page === 'overview') {
            if (!rows.length)
                await loadData();
            else {
                drawOverview();
                if (typeof drawDashboard === 'function')
                    drawDashboard();
            }
        }
        if (page === 'profile')
            drawProfile();
        if (page === 'analysis') {
            if (!stats.length)
                await loadData();
            drawAnalysis();
        }
        if (page === 'research') {
            researchData = await api('/api/research');
            drawResearch();
        }
        if (page === 'probability')
            await drawProb();
        if (page === 'clusters') {
            clusterData = await api('/api/clusters');
            drawClusters();
        }
        if (page === 'models')
            drawModels(await api('/api/models'));
        if (page === 'monitor')
            await drawMonitor();
    });
}
document.querySelectorAll('[data-page],[data-go]').forEach(x => x.onclick = () => go(x.dataset.page || x.dataset.go));
$('applyFilter').onclick = () => task(loadData);
task(loadData);
