/* global Chart */
import { formatDay, withAlpha } from './format.js';

const MS_PER_DAY = 86400000;

/**
 * Turns the API rows into one point per calendar day for the last `days` days, oldest first.
 * The API returns newest first and omits days with no traffic, so the rows are reordered and the
 * gaps are filled with zeros; otherwise a quiet day would vanish and the x-axis would skip time.
 * The window ends at the later of today (UTC, which is how the database buckets days) and the
 * newest row, so a database clock ahead of the browser's does not push its latest day off the end.
 */
export function buildDailySeries(rows, days) {
    const byDay = new Map(rows.map((row) => [row.day, row]));
    const todayMs = Date.parse(`${new Date().toISOString().slice(0, 10)}T00:00:00Z`);
    const end = rows.reduce((max, row) => {
        const ms = Date.parse(`${row.day}T00:00:00Z`);
        return Number.isNaN(ms) ? max : Math.max(max, ms);
    }, todayMs);
    const series = [];
    for (let offset = days - 1; offset >= 0; offset -= 1) {
        const day = new Date(end - offset * MS_PER_DAY).toISOString().slice(0, 10);
        const row = byDay.get(day);
        series.push({
            day,
            queries: row ? row.total_queries : 0,
            up: row ? row.thumbs_up : 0,
            down: row ? row.thumbs_down : 0,
        });
    }
    return series;
}

/**
 * Draws the daily usage line chart on `canvas` and returns the Chart.js instance. `series` comes from
 * buildDailySeries. `colors` holds the CSS token values the chart paints with (accent, good, bad, text,
 * muted, border, surface) plus the page's fontFamily, which becomes the Chart.js default font.
 * `animate` turns the draw animation on or off.
 * Chart.js errors propagate to the caller, and a chart already on the canvas must be destroyed first,
 * because Chart.js refuses to draw twice on one canvas.
 */
export function createDailyChart(canvas, series, colors, animate) {
    const { accent, good, bad, text, muted, border, surface } = colors;

    Chart.defaults.font.family = colors.fontFamily;

    return new Chart(canvas, {
        type: 'line',
        data: {
            labels: series.map((point) => formatDay(point.day)),
            datasets: [
                {
                    label: 'Queries',
                    data: series.map((point) => point.queries),
                    fill: 'origin',
                    backgroundColor: withAlpha(accent, '1a'),
                    borderColor: accent,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: accent,
                    order: 1,
                },
                {
                    label: 'Thumbs up',
                    data: series.map((point) => point.up),
                    borderColor: good,
                    borderWidth: 1.5,
                    borderDash: [4, 3],
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    order: 2,
                },
                {
                    label: 'Thumbs down',
                    data: series.map((point) => point.down),
                    borderColor: bad,
                    borderWidth: 1.5,
                    borderDash: [2, 2],
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    order: 3,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: animate ? { duration: 500 } : false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: { boxWidth: 14, boxHeight: 2, color: muted, font: { size: 11 }, padding: 10 },
                },
                tooltip: {
                    backgroundColor: surface,
                    titleColor: text,
                    bodyColor: muted,
                    borderColor: border,
                    borderWidth: 1,
                    padding: 8,
                    titleFont: { weight: '600', size: 12 },
                    bodyFont: { size: 11 },
                },
            },
            scales: {
                x: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: {
                        color: muted,
                        font: { size: 10 },
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                    },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: withAlpha(border, '50') },
                    border: { display: false },
                    ticks: { color: muted, font: { size: 10 }, padding: 4, precision: 0 },
                },
            },
        },
    });
}
