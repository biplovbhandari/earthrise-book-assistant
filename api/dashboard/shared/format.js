const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Satisfaction bands in whole percent. The API sends thumbs_up_pct as a 0-1 fraction.
const SAT_HIGH_MIN = 85;
const SAT_MID_MIN = 75;

// Retrieval scores at or below this are flagged as weak matches. For cross-encoder reranker logits,
// 0 is the natural boundary: negative means the model considers the passage irrelevant.
// For cosine similarity (0-1), set to ~0.25. For RRF sums (~0.03 max), set near 0.01.
const SCORE_LOW_MAX = 0;
const SCORE_HIGH_MIN = 1.5;

const COUNT_FORMAT = new Intl.NumberFormat('en-US');
const SHORT_DATE = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
const LONG_DATE = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
const DATE_TIME = new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
});

// ---- Formatting helpers (pure) ----

/** True for a finite number. API fields that are null, missing or malformed all fail this check. */
export function isNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

/** True for a plain JSON object, as opposed to null or an array. */
export function isRecord(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Formats a count with thousands separators, or "-" when the value is missing. */
export function formatCount(value) {
    return isNumber(value) ? COUNT_FORMAT.format(value) : '-';
}

/** Formats milliseconds as seconds with one decimal, for example 1830 becomes "1.8s". */
export function formatLatency(ms) {
    return isNumber(ms) ? `${(ms / 1000).toFixed(1)}s` : '-';
}

/**
 * Formats a duration as "Xm Ys". Under a minute it drops the minutes ("45s") and from an hour up it
 * switches to "Xh Ym", because a conversation left open overnight would otherwise read "1200m 5s".
 */
export function formatDuration(seconds) {
    if (!isNumber(seconds) || seconds < 0) {
        return '-';
    }
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${total % 60}s`;
    }
    return `${total}s`;
}

/** Parses an ISO timestamp into a Date, or returns null when it is missing or invalid. */
export function parseDate(value) {
    if (!value) {
        return null;
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
}

/** Formats an ISO timestamp in the viewer's time zone as "Oct 12, 2:23 PM". */
export function formatDate(isoString) {
    const date = parseDate(isoString);
    return date ? DATE_TIME.format(date) : '-';
}

/** Formats the span between two timestamps, leaving the year off the start when both share it. */
export function formatSpan(firstIso, lastIso) {
    const first = parseDate(firstIso);
    const last = parseDate(lastIso);
    if (!first || !last) {
        return 'No interactions recorded yet';
    }
    if (first.toDateString() === last.toDateString()) {
        return LONG_DATE.format(last);
    }
    if (first.getFullYear() === last.getFullYear()) {
        return `${SHORT_DATE.format(first)} - ${LONG_DATE.format(last)}`;
    }
    return `${LONG_DATE.format(first)} - ${LONG_DATE.format(last)}`;
}

/**
 * Formats a "YYYY-MM-DD" day as "Oct 5". The string is split rather than parsed with new Date(),
 * because Date reads a bare date as UTC midnight, which shows as the previous day west of UTC.
 */
export function formatDay(day) {
    const [, month, date] = day.split('-');
    return `${MONTHS[Number(month) - 1]} ${Number(date)}`;
}

/** Formats a retrieval score to two decimals, or "n/a" when the answer cited no chunk. */
export function formatScore(score) {
    return isNumber(score) ? score.toFixed(2) : 'n/a';
}

/** Picks the colour class for a score: red (miss), yellow (borderline), green (good match). */
export function scoreClass(score) {
    if (!isNumber(score)) {
        return 'score-na';
    }
    const rounded = Number(score.toFixed(2));
    if (rounded <= SCORE_LOW_MAX) {
        return 'score-lo';
    }
    return rounded >= SCORE_HIGH_MIN ? 'score-hi' : 'score-md';
}

/** Converts a 0-1 fraction to whole percent, so the label and the colour band always agree. */
export function toPercent(fraction) {
    return Math.round(fraction * 100);
}

/** Formats a 0-1 satisfaction fraction as a percentage, or "-" when no answer was rated. */
export function formatSatisfaction(fraction) {
    return isNumber(fraction) ? `${toPercent(fraction)}%` : '-';
}

/** Picks the colour class for a satisfaction fraction: 85% and up, 75-84%, or below 75%. */
export function satClass(fraction) {
    if (!isNumber(fraction)) {
        return 'sat-na';
    }
    const percent = toPercent(fraction);
    if (percent >= SAT_HIGH_MIN) {
        return 'sat-hi';
    }
    return percent >= SAT_MID_MIN ? 'sat-md' : 'sat-lo';
}

/** Shortens a UUID to its first 8 characters, enough to tell visitors apart in a table. */
export function shortId(id) {
    return id ? id.slice(0, 8) : '-';
}

/** The best label a conversation has, or '' when neither a title nor a topic was recorded. */
export function conversationTitle(conversation) {
    return conversation.title || conversation.topic || '';
}

/**
 * Appends an alpha channel to a 6-digit hex colour read from a CSS token. Any other colour format is
 * returned unchanged, so a token written as rgb() or a named colour still draws, just without the tint.
 */
export function withAlpha(color, alphaHex) {
    return /^#[0-9a-f]{6}$/i.test(color) ? color + alphaHex : color;
}
