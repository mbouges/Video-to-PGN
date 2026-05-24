/* ═══════════════════════════════════════════════════════════════════════════
   Video to PGN — Frontend Application
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ──────────────────── STATE ────────────────────
    let currentJobId = null;
    let eventSource = null;
    let lastUrl = '';

    // Stage ordering for overall progress computation
    const STAGE_ORDER = ['downloading', 'segmenting', 'analyzing', 'generating'];
    const STAGE_WEIGHT = { downloading: 0.15, segmenting: 0.15, analyzing: 0.55, generating: 0.15 };

    // ──────────────────── DOM REFERENCES ────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dom = {
        // Sections
        inputSection:       $('#input-section'),
        processingSection:  $('#processing-section'),
        errorSection:       $('#error-section'),
        resultsSection:     $('#results-section'),

        // Form
        urlForm:            $('#url-form'),
        urlInput:           $('#url-input'),
        urlClear:           $('#url-clear'),
        urlError:           $('#url-error'),
        convertBtn:         $('#convert-btn'),

        // Processing
        processingSubtitle: $('#processing-subtitle'),
        overallFill:        $('#overall-progress-fill'),
        overallLabel:       $('#overall-progress-label'),
        cancelBtn:          $('#cancel-btn'),

        // Error
        errorMessage:       $('#error-message'),
        retryBtn:           $('#retry-btn'),
        errorNewBtn:        $('#error-new-btn'),

        // Results
        resultsTitle:       $('#results-title'),
        resultsSubtitle:    $('#results-subtitle'),
        gamesGrid:          $('#games-grid'),
        downloadAllBtn:     $('#download-all-btn'),
        newConversionBtn:   $('#new-conversion-btn'),

        // Toast
        toastContainer:     $('#toast-container'),
    };

    // ──────────────────── YOUTUBE URL VALIDATION ────────────────────
    const YOUTUBE_PATTERNS = [
        /^https?:\/\/(www\.)?youtube\.com\/watch\?.*v=[\w-]{11}/,
        /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/,
        /^https?:\/\/(www\.)?youtube\.com\/live\/[\w-]{11}/,
        /^https?:\/\/youtu\.be\/[\w-]{11}/,
        /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]{11}/,
    ];

    function isValidYouTubeUrl(url) {
        const trimmed = url.trim();
        return YOUTUBE_PATTERNS.some((rx) => rx.test(trimmed));
    }

    // ──────────────────── SECTION MANAGEMENT ────────────────────
    function showSection(sectionEl) {
        [dom.inputSection, dom.processingSection, dom.errorSection, dom.resultsSection].forEach((s) => {
            if (s === sectionEl) {
                s.classList.remove('hidden');
                // Re-trigger entrance animation
                s.style.animation = 'none';
                // Force reflow
                void s.offsetHeight;
                s.style.animation = '';
            } else {
                s.classList.add('hidden');
            }
        });
    }

    // ──────────────────── TOAST NOTIFICATIONS ────────────────────
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast--${type}`;

        const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
        toast.innerHTML = `<span class="toast__icon">${icon}</span><span>${escapeHtml(message)}</span>`;

        dom.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toastOut 0.3s var(--ease-out) forwards';
            toast.addEventListener('animationend', () => toast.remove());
        }, 3000);
    }

    // ──────────────────── UTILITY ────────────────────
    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function setButtonLoading(btn, loading) {
        if (loading) {
            btn.classList.add('btn--loading');
            btn.disabled = true;
        } else {
            btn.classList.remove('btn--loading');
            btn.disabled = false;
        }
    }

    // ──────────────────── STAGE PROGRESS HELPERS ────────────────────
    function getStageEl(stageName) {
        return $(`#stage-${stageName}`);
    }

    function resetStages() {
        STAGE_ORDER.forEach((name) => {
            const el = getStageEl(name);
            if (!el) return;
            el.classList.remove('stage--active', 'stage--completed');
            const bar = $(`#stage-${name}-bar`);
            const pct = $(`#stage-${name}-percent`);
            const msg = $(`#stage-${name}-message`);
            if (bar) bar.style.width = '0%';
            if (pct) pct.textContent = '';
            if (msg) msg.textContent = 'Waiting...';
        });
        dom.overallFill.style.width = '0%';
        dom.overallLabel.textContent = '0%';
    }

    function activateStage(stageName) {
        STAGE_ORDER.forEach((name) => {
            const el = getStageEl(name);
            if (!el) return;
            const idx = STAGE_ORDER.indexOf(name);
            const activeIdx = STAGE_ORDER.indexOf(stageName);

            if (idx < activeIdx) {
                // Completed
                el.classList.remove('stage--active');
                el.classList.add('stage--completed');
            } else if (idx === activeIdx) {
                // Active
                el.classList.add('stage--active');
                el.classList.remove('stage--completed');
            } else {
                // Pending
                el.classList.remove('stage--active', 'stage--completed');
            }
        });
    }

    function updateStageProgress(stageName, progress, message) {
        const bar = $(`#stage-${stageName}-bar`);
        const pct = $(`#stage-${stageName}-percent`);
        const msg = $(`#stage-${stageName}-message`);

        if (bar) bar.style.width = `${Math.min(100, progress)}%`;
        if (pct) pct.textContent = progress != null ? `${Math.round(progress)}%` : '';
        if (msg && message) msg.textContent = message;
    }

    function computeOverallProgress(currentStage, stageProgress) {
        const idx = STAGE_ORDER.indexOf(currentStage);
        if (idx === -1) return 0;

        let total = 0;
        for (let i = 0; i < STAGE_ORDER.length; i++) {
            const w = STAGE_WEIGHT[STAGE_ORDER[i]];
            if (i < idx) {
                total += w * 100;
            } else if (i === idx) {
                total += w * (stageProgress || 0);
            }
        }
        return Math.min(100, Math.round(total));
    }

    function updateOverallProgress(pct) {
        dom.overallFill.style.width = `${pct}%`;
        dom.overallLabel.textContent = `${pct}%`;
    }

    function completeAllStages() {
        STAGE_ORDER.forEach((name) => {
            const el = getStageEl(name);
            if (!el) return;
            el.classList.remove('stage--active');
            el.classList.add('stage--completed');
            const bar = $(`#stage-${name}-bar`);
            if (bar) bar.style.width = '100%';
            const pct = $(`#stage-${name}-percent`);
            if (pct) pct.textContent = '100%';
        });
        updateOverallProgress(100);
    }

    // ──────────────────── API CALLS ────────────────────
    async function submitJob(url) {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || err.message || `Server error (${response.status})`);
        }

        const data = await response.json();
        return data.job_id;
    }

    function connectSSE(jobId) {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        eventSource = new EventSource(`/api/status/${jobId}`);

        eventSource.onmessage = (event) => {
            let data;
            try {
                data = JSON.parse(event.data);
            } catch {
                return;
            }
            handleSSEEvent(data);
        };

        eventSource.onerror = () => {
            // SSE connection lost — don't show error if we already completed or explicitly errored
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        };
    }

    function handleSSEEvent(data) {
        const { stage, progress, message, current_game, total_games } = data;

        if (stage === 'error') {
            closeSSE();
            showError(message || 'An unexpected error occurred during processing.');
            return;
        }

        if (stage === 'complete') {
            completeAllStages();
            dom.processingSubtitle.textContent = message || 'Processing complete!';
            closeSSE();
            // Brief delay so the user sees 100% before switching view
            setTimeout(() => fetchAndShowResults(), 600);
            return;
        }

        // Map to our stage names (backend might send 'generating' for PGN generation)
        const mappedStage = STAGE_ORDER.includes(stage) ? stage : null;
        if (!mappedStage) return;

        activateStage(mappedStage);
        updateStageProgress(mappedStage, progress || 0, message);

        const overall = computeOverallProgress(mappedStage, progress || 0);
        updateOverallProgress(overall);

        if (message) {
            dom.processingSubtitle.textContent = message;
        }
    }

    function closeSSE() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    async function fetchAndShowResults() {
        try {
            const response = await fetch(`/api/games/${currentJobId}`);
            if (!response.ok) throw new Error(`Failed to load results (${response.status})`);
            const data = await response.json();
            renderResults(data.games || []);
        } catch (err) {
            showError(err.message);
        }
    }

    // ──────────────────── RENDER RESULTS ────────────────────
    function renderResults(games) {
        if (games.length === 0) {
            dom.resultsTitle.textContent = `0 games extracted`;
            dom.resultsSubtitle.textContent = 'No complete games could be identified from the video. The positions may not have been recognized accurately enough to reconstruct moves.';
            dom.gamesGrid.innerHTML = '';
            dom.downloadAllBtn.style.display = 'none';
            showSection(dom.resultsSection);
            return;
        }

        dom.resultsTitle.textContent = `${games.length} game${games.length !== 1 ? 's' : ''} extracted successfully`;
        dom.resultsSubtitle.textContent = 'Your PGN files are ready to download.';
        dom.downloadAllBtn.style.display = 'inline-flex';

        dom.gamesGrid.innerHTML = '';

        games.forEach((game, i) => {
            const card = createGameCard(game, i);
            dom.gamesGrid.appendChild(card);
        });

        showSection(dom.resultsSection);
    }

    function createGameCard(game, index) {
        const { game_num, result, pgn, moves_count } = game;

        const resultClass = getResultClass(result);
        const pgnPreview = (pgn || '').slice(0, 200);

        const card = document.createElement('div');
        card.className = 'game-card';
        card.style.animationDelay = `${index * 0.06}s`;

        card.innerHTML = `
            <div class="game-card__header">
                <div class="game-card__info">
                    <span class="game-card__number">Game ${game_num}</span>
                    <span class="game-card__moves">${moves_count} move${moves_count !== 1 ? 's' : ''}</span>
                </div>
                <span class="result-badge result-badge--${resultClass}">${escapeHtml(result || '?')}</span>
            </div>
            <div class="game-card__pgn">${escapeHtml(pgnPreview)}</div>
            <div class="game-card__actions">
                <button class="btn btn--primary btn--sm btn-download" data-game="${game_num}" title="Download PGN">
                    <span class="btn__icon">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="7 10 12 15 17 10"/>
                            <line x1="12" y1="15" x2="12" y2="3"/>
                        </svg>
                    </span>
                    <span class="btn__text">Download</span>
                </button>
                <button class="btn btn--outline btn--sm btn--copy btn-copy" data-pgn="${escapeAttr(pgn || '')}" title="Copy PGN">
                    <span class="btn__text">Copy</span>
                </button>
                <a class="btn btn--ghost btn--sm btn-lichess" href="https://lichess.org/paste" target="_blank" rel="noopener" title="Open in Lichess">
                    <span class="btn__text">Lichess ↗</span>
                </a>
            </div>
        `;

        // Event listeners
        card.querySelector('.btn-download').addEventListener('click', () => downloadGame(game_num));
        card.querySelector('.btn-copy').addEventListener('click', (e) => copyPgn(e.currentTarget, pgn));
        card.querySelector('.btn-lichess').addEventListener('click', () => openInLichess(pgn));

        return card;
    }

    function escapeAttr(str) {
        return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function getResultClass(result) {
        if (!result) return 'draw';
        if (result === '1-0') return 'white';
        if (result === '0-1') return 'black';
        return 'draw';
    }

    // ──────────────────── GAME ACTIONS ────────────────────
    function downloadGame(gameNum) {
        const a = document.createElement('a');
        a.href = `/api/download/${currentJobId}/${gameNum}`;
        a.download = `game_${gameNum}.pgn`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    function downloadAll() {
        const a = document.createElement('a');
        a.href = `/api/download/${currentJobId}`;
        a.download = 'all_games.pgn';
        document.body.appendChild(a);
        a.click();
        a.remove();
    }

    async function copyPgn(btn, pgn) {
        try {
            await navigator.clipboard.writeText(pgn);
            btn.classList.add('copied');
            btn.querySelector('.btn__text').textContent = 'Copied!';
            showToast('PGN copied to clipboard');
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.querySelector('.btn__text').textContent = 'Copy';
            }, 2000);
        } catch {
            showToast('Failed to copy', 'error');
        }
    }

    function openInLichess(pgn) {
        // Copy PGN to clipboard first so user can paste at Lichess
        navigator.clipboard.writeText(pgn).catch(() => {});
        window.open('https://lichess.org/paste', '_blank', 'noopener');
    }

    // ──────────────────── ERROR HANDLING ────────────────────
    function showError(message) {
        dom.errorMessage.textContent = message;
        showSection(dom.errorSection);
    }

    function resetToInput() {
        closeSSE();
        currentJobId = null;
        resetStages();
        dom.urlInput.value = lastUrl || '';
        updateClearButton();
        dom.urlError.textContent = '';
        dom.urlError.classList.remove('visible');
        setButtonLoading(dom.convertBtn, false);
        showSection(dom.inputSection);
        dom.urlInput.focus();
    }

    // ──────────────────── FORM HANDLING ────────────────────
    function updateClearButton() {
        dom.urlClear.classList.toggle('visible', dom.urlInput.value.length > 0);
    }

    async function handleSubmit(e) {
        e.preventDefault();

        const url = dom.urlInput.value.trim();
        if (!url) {
            showValidationError('Please enter a YouTube URL.');
            return;
        }

        if (!isValidYouTubeUrl(url)) {
            showValidationError('Please enter a valid YouTube URL (youtube.com or youtu.be).');
            return;
        }

        clearValidationError();
        lastUrl = url;
        setButtonLoading(dom.convertBtn, true);

        try {
            currentJobId = await submitJob(url);
            resetStages();
            showSection(dom.processingSection);
            connectSSE(currentJobId);
        } catch (err) {
            setButtonLoading(dom.convertBtn, false);
            showValidationError(err.message || 'Failed to start processing.');
        }
    }

    function showValidationError(msg) {
        dom.urlError.textContent = msg;
        dom.urlError.classList.add('visible');
        dom.urlInput.focus();
    }

    function clearValidationError() {
        dom.urlError.textContent = '';
        dom.urlError.classList.remove('visible');
    }

    // ──────────────────── EVENT LISTENERS ────────────────────
    function init() {
        // Form
        dom.urlForm.addEventListener('submit', handleSubmit);

        dom.urlInput.addEventListener('input', () => {
            updateClearButton();
            if (dom.urlError.classList.contains('visible')) {
                clearValidationError();
            }
        });

        dom.urlClear.addEventListener('click', () => {
            dom.urlInput.value = '';
            updateClearButton();
            clearValidationError();
            dom.urlInput.focus();
        });

        // Cancel processing
        dom.cancelBtn.addEventListener('click', () => {
            closeSSE();
            resetToInput();
        });

        // Error actions
        dom.retryBtn.addEventListener('click', () => {
            if (lastUrl) {
                dom.urlInput.value = lastUrl;
                showSection(dom.inputSection);
                // Auto-submit
                setTimeout(() => dom.urlForm.requestSubmit(), 100);
            } else {
                resetToInput();
            }
        });

        dom.errorNewBtn.addEventListener('click', () => {
            lastUrl = '';
            resetToInput();
        });

        // Results actions
        dom.downloadAllBtn.addEventListener('click', downloadAll);

        dom.newConversionBtn.addEventListener('click', () => {
            lastUrl = '';
            resetToInput();
        });

        // Focus the input on load
        dom.urlInput.focus();
    }

    // ──────────────────── BOOT ────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
