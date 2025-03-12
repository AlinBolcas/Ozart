let isProcessing = false;

async function processSong() {
    const input = document.getElementById('song-input');
    const btn = document.getElementById('process-btn');
    const status = document.getElementById('status-message');
    
    if (!input.value) return;
    
    // Get the application prefix 
    const prefix = window.APP_CONFIG ? APP_CONFIG.prefix : '';
    
    // Get API key values directly from inputs
    const openaiKey = document.getElementById('openai-key').value;
    const replicateKey = document.getElementById('replicate-key').value;
    
    if (!openaiKey || !replicateKey) {
        status.innerHTML = '<div class="error">⚠️ Please add both API keys before processing songs.</div>';
        return;
    }
    
    btn.disabled = true;
    isProcessing = true;
    status.innerHTML = '<div class="info">Please wait about 1 minute, your song is processing...</div>';
    
    try {
        const response = await fetch(`${prefix}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                song: input.value
            })
        });

        if (!response.ok) {
            throw new Error('Failed to process song');
        }
        
        // Start polling for updates
        pollStatus();
        
    } catch (error) {
        showError(error);
        btn.disabled = false;
        isProcessing = false;
    }
}

async function pollStatus() {
    if (!isProcessing) return;
    
    // Get the application prefix
    const prefix = window.APP_CONFIG ? APP_CONFIG.prefix : '';
    
    try {
        const response = await fetch(`${prefix}/status`);
        const data = await response.json();
        
        // Update counts
        document.getElementById('artwork-count').textContent = data.artwork_count;
        document.getElementById('sidebar-artwork-count').textContent = data.artwork_count;
        
        // Show any errors
        if (data.errors && data.errors.length > 0) {
            showError(data.errors[0]);
        }
        
        // Update gallery
        if (data.artworks && data.artworks.length > 0) {
            updateGallery(data.artworks);
        }
        
        // Continue polling if still processing
        if (data.processing) {
            setTimeout(pollStatus, 2000);
        } else {
            isProcessing = false;
            document.getElementById('process-btn').disabled = false;
            document.getElementById('status-message').innerHTML = '';
            document.getElementById('song-input').value = '';
        }
        
    } catch (error) {
        console.error('Polling error:', error);
        showError(error);
        isProcessing = false;
        document.getElementById('process-btn').disabled = false;
    }
}

function updateGallery(artworks) {
    console.log("Updating gallery with artworks:", artworks);
    const gallery = document.getElementById('artwork-gallery');
    
    if (!artworks || artworks.length === 0) {
        gallery.innerHTML = '<p>No artworks available yet.</p>';
        return;
    }
    
    gallery.innerHTML = artworks.map(artwork => {
        console.log("Processing artwork:", artwork);
        
        // Get the application prefix
        const prefix = window.APP_CONFIG ? APP_CONFIG.prefix : '';
        
        // Make sure image path is properly formatted
        let imageUrl;
        if (!artwork.image_path) {
            imageUrl = `${prefix}/static/data/fallback.png`;
        } else {
            // Always use /output/ prefix for artwork images
            imageUrl = `${prefix}/output/${artwork.image_path}`;
        }
        
        console.log("Image URL:", imageUrl);
        
        return `
            <div class="artwork-card">
                <img src="${imageUrl}" alt="${artwork.song_title || 'Untitled'}" 
                     onerror="this.onerror=null; this.src='/static/data/fallback.png'">
                <div class="artwork-info">
                    <h3>${artwork.song_title || 'Untitled'}</h3>
                    <p>${artwork.artist || 'Unknown artist'}</p>
                    <details>
                        <summary>Show Details</summary>
                        <div class="artwork-details">
                            <h4>Song Analysis</h4>
                            <p>${artwork.interpretation || 'No analysis available'}</p>
                            <h4>Image Prompt</h4>
                            <p><code>${artwork.prompt || 'No prompt available'}</code></p>
                            ${artwork.audio_path ? `
                                <h4>Listen to the song</h4>
                                <audio controls src="/output/${artwork.audio_path}"></audio>
                            ` : ''}
                            <p><small>Created: ${artwork.human_time || 'Unknown time'}</small></p>
                        </div>
                    </details>
                </div>
            </div>
        `;
    }).join('');
}

function showError(error) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = `Error: ${error}`;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function showSettingsModal() {
    // Fetch current prompts
    loadPrompts();
    
    // Show modal
    document.getElementById('settings-modal').style.display = 'block';
}

async function loadPrompts() {
    try {
        const response = await fetch('/get-prompts');
        const data = await response.json();
        
        if (data.success) {
            // Fill the form fields
            document.getElementById('music-system-prompt').value = data.prompts.music_analysis_system_prompt || '';
            document.getElementById('music-user-prompt').value = data.prompts.music_analysis_user_prompt || '';
            document.getElementById('image-system-prompt').value = data.prompts.image_prompt_system_prompt || '';
            document.getElementById('image-user-prompt').value = data.prompts.image_prompt_user_prompt || '';
        } else {
            showError(data.error || 'Failed to load prompts');
        }
    } catch (error) {
        console.error('Error loading prompts:', error);
        showError('Failed to load prompts');
    }
}

async function resetDefaultPrompts() {
    // Clear session storage for prompts to revert to defaults
    try {
        await fetch('/save-prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}) // Empty object to clear custom prompts
        });
        
        // Reload prompts from defaults
        loadPrompts();
        showMessage('Prompts reset to defaults', 'success');
    } catch (error) {
        console.error('Error resetting prompts:', error);
        showError('Failed to reset prompts');
    }
}

async function saveSettings() {
    // Save both API keys and prompts
    try {
        // Save API keys
        const openaiKey = document.getElementById('openai-key').value;
        const replicateKey = document.getElementById('replicate-key').value;
        
        if (openaiKey || replicateKey) {
            await saveApiKeys(openaiKey, replicateKey);
        }
        
        // Save prompts
        const musicSystemPrompt = document.getElementById('music-system-prompt').value;
        const musicUserPrompt = document.getElementById('music-user-prompt').value;
        const imageSystemPrompt = document.getElementById('image-system-prompt').value;
        const imageUserPrompt = document.getElementById('image-user-prompt').value;
        
        const promptResponse = await fetch('/save-prompts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                music_analysis_system_prompt: musicSystemPrompt,
                music_analysis_user_prompt: musicUserPrompt,
                image_prompt_system_prompt: imageSystemPrompt,
                image_prompt_user_prompt: imageUserPrompt
            })
        });
        
        const promptData = await promptResponse.json();
        
        if (promptData.success) {
            showMessage('Settings saved successfully!', 'success');
            document.getElementById('settings-modal').style.display = 'none';
        } else {
            showError('Failed to save prompts');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showError('Failed to save settings');
    }
}

async function saveApiKeys(openaiKey, replicateKey) {
    const response = await fetch('/save-keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            openai_key: openaiKey,
            replicate_key: replicateKey
        })
    });
    
    const data = await response.json();
    if (!data.success) {
        throw new Error('Failed to save API keys');
    }
}

function openTab(evt, tabName) {
    // Hide all tab content
    const tabContent = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContent.length; i++) {
        tabContent[i].style.display = 'none';
    }
    
    // Remove 'active' class from all tab buttons
    const tabButtons = document.getElementsByClassName('tab-btn');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].className = tabButtons[i].className.replace(' active', '');
    }
    
    // Show current tab and add 'active' class to button
    document.getElementById(tabName).style.display = 'block';
    evt.currentTarget.className += ' active';
}

// Load initial artworks when page loads
window.addEventListener('load', async () => {
    try {
        // Check API key status
        const keyResponse = await fetch('/check-keys');
        const keyData = await keyResponse.json();
        
        if (keyData.keys_required) {
            // Show API key form if keys are missing
            document.getElementById('api-key-modal').style.display = 'block';
            showMessage('Please enter your API keys to continue', 'info');
        }
        
        // Get artworks as before
        const response = await fetch('/status');
        const data = await response.json();
        if (data.artworks && data.artworks.length > 0) {
            updateGallery(data.artworks);
        }
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
});

// Add event listener for Enter key on the input field
document.getElementById('song-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        document.getElementById('process-btn').click();
    }
});

// Load user data on page load
window.addEventListener('load', function() {
    fetch('/get-user-data')
        .then(response => response.json())
        .then(data => {
            // Fill in API keys if available
            if (data.api_keys) {
                document.getElementById('openai-key').value = data.api_keys.openai_key || '';
                document.getElementById('replicate-key').value = data.api_keys.replicate_key || '';
            }
            
            // Fill in custom prompts if available
            if (data.custom_prompts) {
                document.getElementById('music-system-prompt').value = data.custom_prompts.music_analysis_system_prompt || '';
                document.getElementById('music-user-prompt').value = data.custom_prompts.music_analysis_user_prompt || '';
                document.getElementById('image-system-prompt').value = data.custom_prompts.image_prompt_system_prompt || '';
                document.getElementById('image-user-prompt').value = data.custom_prompts.image_prompt_user_prompt || '';
            }
        });
});