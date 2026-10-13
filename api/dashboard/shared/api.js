const TOKEN_KEY = 'dashboard_token';
const REQUEST_TIMEOUT_MS = 15000;
const AUTH_STATUSES = [401, 403];

// ---- Token storage ----
// sessionStorage can throw (blocked storage, some private modes). The dashboard still works for the
// current page load in that case; it only asks for the token again after a reload.

/** Returns the saved admin token, or '' when there is none or storage is unavailable. */
export function readStoredToken() {
    try {
        return sessionStorage.getItem(TOKEN_KEY) || '';
    } catch (_e) {
        return '';
    }
}

/** Saves the admin token for this tab; sessionStorage clears it when the tab closes. */
export function storeToken(value) {
    try {
        sessionStorage.setItem(TOKEN_KEY, value);
    } catch (_e) {
        // Storage unavailable: the token lives in memory for this page load only.
    }
}

/** Removes the saved admin token. */
export function clearStoredToken() {
    try {
        sessionStorage.removeItem(TOKEN_KEY);
    } catch (_e) {
        // Nothing was stored, so there is nothing to clear.
    }
}

// ---- API access ----

/** An HTTP failure from the analytics API, keeping the status code and FastAPI's `detail` text. */
export class ApiError extends Error {
    constructor(status, detail) {
        super(detail ? `HTTP ${status}: ${detail}` : `HTTP ${status}`);
        this.name = 'ApiError';
        this.status = status;
        this.detail = detail;
    }
}

/** Reads FastAPI's {"detail": "..."} error body, or '' when the body is absent or shaped differently. */
export async function readErrorDetail(response) {
    try {
        const body = await response.json();
        return typeof body.detail === 'string' ? body.detail : '';
    } catch (_e) {
        return '';
    }
}

/**
 * GETs one analytics endpoint with the Bearer token and returns the parsed JSON.
 * Throws ApiError for a non-2xx status or a body that fails `isValid`. Validating the shape here
 * keeps a malformed response from reaching the templates, which assume arrays and objects.
 */
export async function apiGet(url, token, isValid) {
    const response = await fetch(url, {
        headers: { Accept: 'application/json', Authorization: `Bearer ${token}` },
        cache: 'no-store',
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) {
        throw new ApiError(response.status, await readErrorDetail(response));
    }
    const data = await response.json();
    if (!isValid(data)) {
        throw new ApiError(response.status, 'Unexpected response shape');
    }
    return data;
}

/** Turns a failed request into short text for the section that could not load. */
export function describeError(err) {
    if (err instanceof ApiError) {
        return err.message;
    }
    if (err && err.name === 'TimeoutError') {
        return 'request timed out';
    }
    if (err instanceof SyntaxError) {
        return 'invalid response';
    }
    return 'network error';
}

/** True when the API rejected the admin token (HTTP 401 or 403), which is the cue to sign the user out. */
export function isAuthError(err) {
    return err instanceof ApiError && AUTH_STATUSES.includes(err.status);
}
