/* global Vue, Chart */
import {
    isRecord,
    formatCount,
    formatLatency,
    formatSpan,
} from '../shared/format.js';

import {
    apiGet,
    isAuthError,
    describeError,
} from '../shared/api.js';

import { createDailyChart, buildDailySeries } from '../shared/chart.js';

const RANGE_OPTIONS = [30, 90];
const DEFAULT_DAYS = RANGE_OPTIONS[0];
const CITATIONS_LIMIT = 20;
const GAPS_LIMIT = 20;
const CONVERSATIONS_LIMIT = 50;

/**
 * Owns everything the analytics view shows: the five sections, the daily chart, and the requests
 * that fill them. `session` is the shell's plain `{ token, id }` object: the token authenticates each
 * request, and the id lets a response that arrives after a disconnect be dropped. `onAuthError(reason)`
 * is called when the API rejects the token, so the shell can sign the user out.
 * Must run inside a component's setup(), because its watcher is tied to that component.
 */
export function useAnalytics(session, onAuthError) {
    // Read the global here instead of at module level. app.js reports a failed CDN load only after
    // its imports have run, so a top-level `Vue` read in this file would throw first and leave a
    // blank page.
    const { ref, reactive, computed, watch, nextTick } = Vue;

    const days = ref(DEFAULT_DAYS);
    const overview = ref(null);
    const daily = ref(null);
    const citations = ref(null);
    const gaps = ref(null);
    const conversations = ref(null);
    const config = ref(null);
    // One message per section, '' when its last load worked.
    const errors = reactive({ overview: '', daily: '', citations: '', gaps: '', conversations: '' });
    const dailyCanvas = ref(null);

    // Plain variables on purpose: the template never reads them, and wrapping a Chart.js
    // instance in a Vue reactive proxy breaks the library.
    let chart = null;
    let dailyDays = DEFAULT_DAYS;

    const hasDaily = computed(() => Array.isArray(daily.value) && daily.value.length > 0);

    const maxCiteCount = computed(() => {
        return (citations.value || []).reduce((max, row) => Math.max(max, row.cite_count), 0);
    });

    const metrics = computed(() => {
        const stats = overview.value || {};
        return [
            { label: 'Interactions', value: formatCount(stats.total_interactions) },
            { label: 'Conversations', value: formatCount(stats.total_conversations) },
            { label: 'Unique visitors', value: formatCount(stats.unique_visitors) },
            { label: 'Avg latency', value: formatLatency(stats.avg_latency_ms) },
        ];
    });

    const sublineText = computed(() => {
        if (errors.overview) {
            return errors.overview;
        }
        return overview.value ? formatDateRange() : 'Loading...';
    });

    /** Text for a section with nothing to list: its load error, "Loading...", or an empty-state note. */
    function sectionMessage(key, data, emptyText) {
        if (errors[key]) {
            return errors[key];
        }
        if (data === null) {
            return 'Loading...';
        }
        return data.length === 0 ? emptyText : '';
    }

    const messages = computed(() => ({
        daily: sectionMessage('daily', daily.value, 'No usage recorded in this period.'),
        citations: sectionMessage('citations', citations.value, 'No cited sources yet.'),
        gaps: sectionMessage('gaps', gaps.value, 'No thumbs-down questions yet.'),
        conversations: sectionMessage('conversations', conversations.value, 'No conversations yet.'),
    }));

    /** Formats the overview's first and last interaction as the header date range. */
    function formatDateRange() {
        const stats = overview.value;
        return stats ? formatSpan(stats.first_interaction_at, stats.last_interaction_at) : '';
    }

    /** Width of a citation bar as a percentage of the most cited source. */
    function barWidth(count) {
        if (maxCiteCount.value <= 0) {
            return '0%';
        }
        return `${Math.round((count / maxCiteCount.value) * 100)}%`;
    }

    // ---- Chart ----

    /** Destroys the chart, releasing its canvas so a new one can be drawn on it. */
    function destroyChart() {
        if (chart) {
            chart.destroy();
            chart = null;
        }
    }

    /** True when the OS asks for reduced motion, in which case the chart skips its draw animation. */
    function prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    /**
     * Rebuilds the daily usage chart from the current rows. A chart that fails to build is reported
     * in the Daily usage panel; the tables do not depend on it.
     */
    function renderChart() {
        const canvas = dailyCanvas.value;
        const rows = daily.value;
        if (!session.token || !canvas || !Array.isArray(rows) || rows.length === 0) {
            destroyChart();
            return;
        }
        if (typeof Chart === 'undefined') {
            errors.daily = 'Chart library failed to load.';
            return;
        }

        // Only the first draw animates. The chart is rebuilt on every refresh and range change,
        // and replaying the animation each minute would look like a glitch.
        const animate = chart === null && !prefersReducedMotion();
        destroyChart();

        const series = buildDailySeries(rows, dailyDays);
        const css = getComputedStyle(document.documentElement);
        const color = (name) => css.getPropertyValue(name).trim();
        const colors = {
            accent: color('--ac'),
            good: color('--gd'),
            bad: color('--cr'),
            text: color('--tx'),
            muted: color('--tx2'),
            border: color('--bd'),
            surface: color('--bg'),
            fontFamily: getComputedStyle(document.body).fontFamily,
        };

        try {
            chart = createDailyChart(canvas, series, colors, animate);
        } catch (err) {
            console.error('Dashboard: chart render failed', err);
            errors.daily = 'Chart could not be rendered.';
        }
    }

    // ---- Loading data ----

    const neverOutdated = () => false;

    /** Returns a callback that stores a response in the given ref. */
    const assignTo = (target) => (data) => {
        target.value = data;
    };

    /**
     * Loads one endpoint into its section. A result is dropped when the session ended while the
     * request was in flight (disconnect, or disconnect then reconnect) or when `isOutdated` says
     * the answer no longer matches what the user asked for. A 401 or 403 signs the user out.
     */
    async function loadSection(key, url, isValid, apply, isOutdated) {
        const run = session.id;
        try {
            const data = await apiGet(url, session.token, isValid);
            if (run !== session.id || isOutdated()) {
                return;
            }
            apply(data);
            errors[key] = '';
        } catch (err) {
            if (run !== session.id || isOutdated()) {
                return;
            }
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error(`Dashboard: could not load ${key}`, err);
            errors[key] = `Could not load (${describeError(err)}).`;
        }
    }

    /**
     * Loads the daily series for the selected range. Both the response and the range it was fetched
     * for are kept, so the chart is always built over the window its rows belong to.
     */
    function loadDaily() {
        const requestedDays = days.value;
        return loadSection(
            'daily',
            `/analytics/daily?days=${requestedDays}`,
            Array.isArray,
            (rows) => {
                dailyDays = requestedDays;
                daily.value = rows;
            },
            () => requestedDays !== days.value
        );
    }

    /** Fetches the pipeline config once per session; skips on subsequent refreshes. */
    async function loadConfig() {
        if (config.value !== null) {
            return;
        }
        const run = session.id;
        try {
            config.value = await apiGet('/analytics/config', session.token, isRecord);
        } catch (err) {
            if (run !== session.id) {
                return;
            }
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
            }
        }
    }

    /** Loads all sections in parallel; one failing request does not stop the others. */
    function fetchAll() {
        return Promise.allSettled([
            loadConfig(),
            loadSection('overview', '/analytics/overview', isRecord, assignTo(overview), neverOutdated),
            loadDaily(),
            loadSection(
                'citations',
                `/analytics/citations?limit=${CITATIONS_LIMIT}`,
                Array.isArray,
                assignTo(citations),
                neverOutdated
            ),
            loadSection(
                'gaps',
                `/analytics/gaps?limit=${GAPS_LIMIT}`,
                Array.isArray,
                assignTo(gaps),
                neverOutdated
            ),
            loadSection(
                'conversations',
                `/analytics/conversations?limit=${CONVERSATIONS_LIMIT}`,
                Array.isArray,
                assignTo(conversations),
                neverOutdated
            ),
        ]);
    }

    /** Switches the chart range and reloads only the daily series. */
    function setDays(n) {
        if (n === days.value) {
            return;
        }
        days.value = n;
        loadDaily();
    }

    // ---- Lifecycle ----

    // Wait a tick so the canvas for the new rows (and the v-show that reveals it) is in the DOM.
    watch(daily, () => nextTick(renderChart));

    /** Clears all data and errors. Called by the shell on disconnect. */
    function reset() {
        overview.value = null;
        daily.value = null;
        citations.value = null;
        gaps.value = null;
        conversations.value = null;
        config.value = null;
        Object.keys(errors).forEach((key) => {
            errors[key] = '';
        });
    }

    /** Releases the chart canvas. Called by the shell on unmount and disconnect. */
    function cleanup() {
        destroyChart();
    }

    return {
        overview, daily, citations, gaps, conversations, config,
        errors, days, dailyCanvas,
        hasDaily, maxCiteCount, metrics, sublineText, messages,
        rangeOptions: RANGE_OPTIONS,
        fetchAll, setDays,
        barWidth, formatDateRange,
        reset,
        cleanup,
    };
}
