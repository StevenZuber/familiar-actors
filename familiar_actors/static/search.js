function initSearch() {
    const searchInput = document.getElementById('actor-search');
    const suggestions = document.getElementById('suggestions');
    const tabs = document.querySelectorAll('.search-tab');

    if (!searchInput || !suggestions) return;

    let debounceTimer;
    // Detect current mode from active tab
    let searchMode = document.querySelector('.search-tab.active')?.dataset.mode || 'actor';

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            searchMode = this.dataset.mode;
            searchInput.value = '';
            suggestions.innerHTML = '';
            suggestions.classList.remove('active');
            document.getElementById('results').innerHTML = '';
            searchInput.placeholder = searchMode === 'actor'
                ? 'Search for an actor...'
                : 'Search for a movie or show...';
            searchInput.focus();
        });
    });

    searchInput.addEventListener('input', function() {
        clearTimeout(debounceTimer);
        const query = this.value.trim();

        if (query.length < 2) {
            suggestions.innerHTML = '';
            suggestions.classList.remove('active');
            return;
        }

        debounceTimer = setTimeout(async () => {
            const endpoint = searchMode === 'actor'
                ? `/api/search?q=${encodeURIComponent(query)}`
                : `/api/search-titles?q=${encodeURIComponent(query)}`;
            const response = await fetch(endpoint);
            const results = await response.json();

            if (results.length === 0) {
                const noResultsText = searchMode === 'actor'
                    ? 'No actors found'
                    : 'No movies or shows found';
                suggestions.innerHTML = `<div class="suggestion-item no-results">${noResultsText}</div>`;
                suggestions.classList.add('active');
                return;
            }

            if (searchMode === 'actor') {
                suggestions.innerHTML = results.map(actor => `
                    <div class="suggestion-item"
                         hx-get="/search?actor_id=${actor.id}"
                         hx-target="#results"
                         hx-swap="innerHTML"
                         hx-push-url="true">
                        <img src="${escapeHtml(actor.tmdb_image_url || '')}"
                             alt="${escapeHtml(actor.name)}"
                             class="suggestion-thumb"
                             onerror="this.style.display='none'">
                        <span>${escapeHtml(actor.name)}</span>
                    </div>
                `).join('');
            } else {
                suggestions.innerHTML = results.map(title => `
                    <div class="suggestion-item"
                         hx-get="/cast?title_id=${title.tmdb_id}&source=${escapeHtml(title.source)}"
                         hx-target="#results"
                         hx-swap="innerHTML"
                         hx-push-url="true">
                        <span>${escapeHtml(title.title)}${title.year ? ` (${escapeHtml(title.year)})` : ''}</span>
                    </div>
                `).join('');
            }

            htmx.process(suggestions);
            suggestions.classList.add('active');
        }, 300);
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.search-wrapper')) {
            suggestions.classList.remove('active');
        }
    });

    // Close suggestions and update input when selecting
    suggestions.addEventListener('click', function(e) {
        const item = e.target.closest('.suggestion-item');
        if (item && !item.classList.contains('no-results')) {
            searchInput.value = item.querySelector('span').textContent;
            suggestions.classList.remove('active');
        }
    });
}

// Initialize on page load
initSearch();

// Re-initialize after HTMX history restore (back/forward navigation)
document.addEventListener('htmx:historyRestore', function() {
    initSearch();
});
