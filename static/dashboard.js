'use strict';
// Interactive overview. Scenario calls do not create monitored prediction records.
const simIds = ['simSocial', 'simSleep', 'simNotif', 'simNight'];
let simTimer, simSeq = 0, importanceVersion = null;
function simPayload() {
    const base = habits ? { ...habits } : {
        screen_hours: 6,
        social_hours: 2,
        video_hours: 1,
        gaming_hours: .5,
        study_hours: 5,
        sleep_hours: 7,
        breaks: 7,
        exercise: 60,
        stress: 5,
        age: 23,
        gender: 'Male',
        night_minutes: null,
        notifications: null,
        checks: null
    };
    const social = +$('simSocial').value;
    return {
        ...base,
        social_hours: social,
        screen_hours: Math.max(social, Math.min(24, base.screen_hours + social - base.social_hours)),
        sleep_hours: +$('simSleep').value,
        notifications: +$('simNotif').value,
        night_minutes: Math.min(+$('simNight').value, 180, Math.max(social, Math.min(24, base.screen_hours + social - base.social_hours)) * 60)
    };
}
function queueSim() {
    const seq = ++simSeq;
    clearTimeout(simTimer);
    const v = simPayload();
    $('simNight').max = Math.min(180, v.screen_hours * 60);
    $('simNight').value = v.night_minutes;
    simIds.forEach((id, i) => $(id + 'Out').textContent = fmt(+$(id).value, Number.isInteger(+$(id).value) ? 0 : 1) + [' h', ' h', '', ' min'][i]);
    const score = Math.round(100 * (.3 * (1 - Math.min(v.screen_hours / 12, 1)) + .3 * Math.max(0, 1 - Math.abs(v.sleep_hours - 8) / 4) + .2 * (1 - Math.min(v.notifications / 200, 1)) + .2 * (1 - Math.min(v.night_minutes / 180, 1))));
    $('phoneBuddy').textContent = v.screen_hours <= 3 ? 'Phone buddy: a lighter phone-use scenario.' : v.screen_hours <= 6 ? 'Phone buddy: consider a phone-free study break.' : 'Phone buddy: consider reducing recreational phone time.';
    $('simScore').textContent = score;
    $('simArc').setAttribute('stroke-dasharray', `${score} 100`);
    $('simArc').setAttribute('stroke', score >= 75 ? '#35bc99' : score >= 50 ? '#f0aa31' : '#f07b68');
    $('simPrediction').textContent = 'Updating model estimate…';
    $('simPercentile').textContent = `Scenario phone time: ${fmt(v.screen_hours)} h`;
    simTimer = setTimeout(async () => {
        try {
            const p = await api('/api/scenario', 'POST', v);
            if (seq !== simSeq)
                return;
            $('simScore').textContent = p.wellness;
            $('simArc').setAttribute('stroke-dasharray', `${p.wellness} 100`);
            $('simPercentile').textContent = `Phone time above ${fmt(p.percentile, 0)}% of dataset students (${fmt(v.screen_hours)} h)`;
            $('simPrediction').textContent = `Estimated productivity: ${fmt(p.predicted_focus)}/100 · ${p.category}${p.out_of_range.length ? ' · outside training range' : ''}`;
        }
        catch (e) {
            if (seq === simSeq)
                $('simPrediction').textContent = 'Prediction unavailable: ' + e.message;
        }
    }, 220);
}
simIds.forEach(id => $(id).addEventListener('input', queueSim));
queueSim();
function syncDashboard() {
    if (!habits)
        return;
    $('simSocial').value = Math.min(12, habits.social_hours);
    $('simSleep').value = Math.min(12, habits.sleep_hours);
    if (habits.notifications != null)
        $('simNotif').value = Math.min(200, habits.notifications);
    $('simNight').max = 180;
    if (habits.night_minutes != null)
        $('simNight').value = habits.night_minutes;
    $('simBase').textContent = 'Other model inputs use your saved assessment. Slider values are exploratory scenarios, not saved observations.';
    queueSim();
}
const bands = [[0, 2, 'Under 2h'], [2, 4, '2–4h'], [4, 6, '4–6h'], [6, 8, '6–8h'], [8, 25, '8h+']];
function drawDashboard() {
    if (!$('overview').classList.contains('active'))
        return;
    const groups = bands.map(([lo, hi]) => rows.filter(r => r.screen_hours >= lo && r.screen_hours < hi));
    const z = groups.map(g => [
        g.filter(r => r.productivity < 40).length, g.filter(r => r.productivity >= 40 && r.productivity < 70).length, g.filter(r => r.productivity >= 70).length
    ].map(n => g.length ? 100 * n / g.length : null));
    chart('heatmap',
        [
        {
            type: 'heatmap',
            x: ['Low (<40)', 'Medium (40–<70)', 'High (70+)'],
            y: bands.map(b => b[2]),
            z,
            zmin: 0,
            zmax: 100,
            colorscale: [[0, '#f0f6ff'], [.5, '#9ac3f0'], [1, '#2364aa']],
            text: z.map(a => a.map(v => v === null ? '—' : fmt(v, 0) + '%')),
            texttemplate: '%{text}',
            showscale: false,
            hovertemplate: '%{y} · %{x}: %{text}<extra></extra>'
        }
    ],
        { xaxis: { side: 'top' },
        yaxis: { autorange: 'reversed' },
        margin: { l: 115, r: 25, t: 100, b: 35 } });
    chart('overviewHist',
        [
        {
            type: 'histogram',
            x: rows.map(r => r.screen_hours),
            xbins: { start: 0, end: 24, size: 1 },
            marker: { color: '#3987df' },
            hovertemplate: '%{x} h: %{y} students<extra></extra>'
        }
    ],
        { bargap: .12, xaxis: { title: 'Hours per day' }, yaxis: { title: 'Students' } });
    const x = rows.map(r => r.screen_hours),
        y = rows.map(r => r.productivity),
        mx = avg(x),
        my = avg(y);
    let sxx = 0, sxy = 0;
    x.forEach((v, i) => {
        sxx += (v - mx) ** 2;
        sxy += (v - mx) * (y[i] - my);
    });
    const slope = sxx ? sxy / sxx : 0, ends = x.length ? [Math.min(...x), Math.max(...x)] : [];
    const step = Math.max(1, Math.ceil(x.length / 1800)),
        shown = x.map((v, i) => i).filter(i => i % step === 0);
    chart('overviewScatter',
        [
        {
            type: 'scatter',
            mode: 'markers',
            x: shown.map(i => x[i]),
            y: shown.map(i => y[i]),
            name: 'Students',
            marker: { color: '#3987df', size: 5, opacity: .26 }
        }, {
            type: 'scatter',
            mode: 'lines',
            x: ends,
            y: ends.map(v => my + slope * (v - mx)),
            name: 'Linear regression trend',
            line: { color: '#f07b48', dash: 'dash', width: 3 }
        }
    ],
        { showlegend: true, legend: {orientation: 'h', y: 1.15}, margin: {l: 65, r: 18, t: 55, b: 65},
        xaxis: { title: 'Screen time (hours)' },
        yaxis: { title: 'Productivity /100' } });
    chart('overviewBox',
        groups.map((g, i) => ({
        type: 'box',
        name: bands[i][2],
        y: g.map(r => r.productivity),
        boxpoints: 'outliers',
        marker: { color: '#3987df', size: 4 },
        line: { color: '#3987df' },
        fillcolor: 'rgba(57,135,223,.16)'
    })),
        { showlegend: false, xaxis: {title: 'Daily screen-time group'}, yaxis: { title: 'Productivity /100' } });
}
async function loadOverviewExtras() {
    try {
        const m = await api('/api/models'), winner = m.models.find(r => r.model === m.winner);
        if (winner) {
            const el = $('kpis');
            if (!el.querySelector('[data-accuracy]')) {
                const tile = document.createElement('div');
                tile.className = 'metric';
                tile.dataset.accuracy = '1';
                tile.innerHTML = `<small>Held-out accuracy</small><strong>${pct(winner.accuracy)}</strong>`;
                el.append(tile);
            }
        }
        if (importanceVersion === m.version)
            return;
        const r = await api('/api/importance');
        importanceVersion = m.version;
        const a = r.features.slice(0, 7).reverse();
        chart('globalImportance',
            [
            {
                type: 'bar',
                orientation: 'h',
                x: a.map(v => v.mean_abs_shap),
                y: a.map(v => featureLabel(v.feature)),
                marker: { color: '#3987df' }
            }
        ],
            { margin: { l: 150, r: 25, t: 35, b: 85 }, xaxis: { title: 'Mean |SHAP| (risk probability)' } });
        $('importanceNote').textContent = `${r.sample_size} held-out student records; gender held fixed per record. Notifications and night use are not model inputs.`;
    }
    catch (e) {
        $('importanceNote').textContent = 'Model importance unavailable: ' + e.message;
    }
}
// The original data loader may still be in flight. Refresh after each filter and when revisiting overview.
const originalGo = go;
go = async function (page) {
    await originalGo(page);
    if (page === 'overview') {
        drawDashboard();
        importanceVersion = null;
        loadOverviewExtras();
    }
};
if (rows.length) {
    drawDashboard();
    loadOverviewExtras();
}
