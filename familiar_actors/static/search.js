function initSearch() {
    const searchInput = document.getElementById('actor-search');
    const suggestions = document.getElementById('suggestions');
    const searchWrapper = document.querySelector('.search-wrapper');
    const tabs = document.querySelectorAll('.search-tab');
    const photoSection = document.getElementById('photo-upload');
    const photoInput = document.getElementById('photo-input');
    const photoButton = document.getElementById('photo-button');
    const photoPreview = document.getElementById('photo-preview');

    if (!searchInput || !suggestions) return;

    let debounceTimer;
    // Detect current mode from active tab
    let searchMode = document.querySelector('.search-tab.active')?.dataset.mode || 'actor';

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function setMode(mode) {
        searchMode = mode;
        tabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
        const isPhoto = mode === 'photo';
        if (searchWrapper) searchWrapper.hidden = isPhoto;
        if (photoSection) photoSection.hidden = !isPhoto;
        if (!isPhoto) {
            searchInput.placeholder = mode === 'actor'
                ? 'Search for an actor...'
                : 'Search for a movie or show...';
        }
    }
    // Result cards switch back to actor mode when tapped (see _results_grid.html)
    window.faSetMode = setMode;

    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            setMode(this.dataset.mode);
            searchInput.value = '';
            suggestions.innerHTML = '';
            suggestions.classList.remove('active');
            document.getElementById('results').innerHTML = '';
            if (photoPreview) photoPreview.hidden = true;
            if (searchMode !== 'photo') searchInput.focus();
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

    // --- Photo upload ---

    const MAX_PHOTO_DIM = 1024;
    let previewUrl = null;

    // Downscale + re-encode as JPEG before upload: a 12MP phone photo
    // becomes ~100KB (fast on cellular), HEIC becomes JPEG, and EXIF
    // rotation gets baked into the pixels by the browser's decoder.
    async function resizeToJpeg(file) {
        const url = URL.createObjectURL(file);
        try {
            const img = new Image();
            img.src = url;
            await img.decode();
            const scale = Math.min(1, MAX_PHOTO_DIM / Math.max(img.naturalWidth, img.naturalHeight));
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
            canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
            return await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    async function handlePhoto(file) {
        if (!file) return;
        const results = document.getElementById('results');
        results.innerHTML = '<div class="upload-status">Matching your photo&hellip;</div>';

        let blob = null;
        try {
            blob = await resizeToJpeg(file);
        } catch (e) {
            // Browser couldn't decode it (e.g. HEIC outside Safari) —
            // send the original and let the server try.
        }
        const payload = blob || file;

        if (photoPreview) {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
            previewUrl = URL.createObjectURL(payload);
            photoPreview.src = previewUrl;
            photoPreview.hidden = false;
        }

        const formData = new FormData();
        formData.append('photo', payload, blob ? 'photo.jpg' : (file.name || 'photo'));

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            results.innerHTML = await response.text();
            htmx.process(results);
        } catch (e) {
            results.innerHTML = '<div class="results-section"><p class="no-results">Upload failed. Check your connection and try again.</p></div>';
        }
    }

    if (photoButton && photoInput) {
        photoButton.addEventListener('click', () => photoInput.click());
        photoInput.addEventListener('change', function() {
            const file = this.files[0];
            // Reset so picking the same photo again still fires `change`
            this.value = '';
            handlePhoto(file);
        });
    }
}

// Initialize on page load
initSearch();

// Re-initialize after HTMX history restore (back/forward navigation)
document.addEventListener('htmx:historyRestore', function() {
    initSearch();
});
