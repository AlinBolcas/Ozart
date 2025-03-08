// AJAX for updating progress information
document.addEventListener('DOMContentLoaded', function() {
    // Variables
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    
    // Function to update the status
    function updateStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Check if there are any active processes
                const processes = Object.values(data);
                
                if (processes.length === 0) {
                    // If no processes, refresh the page
                    window.location.reload();
                    return;
                }
                
                // Find the most recent process
                const activeProcess = processes.find(p => p.status === 'processing');
                
                if (activeProcess) {
                    // Update progress bar
                    progressBar.style.width = activeProcess.progress + '%';
                    
                    // Update status message
                    statusMessage.textContent = activeProcess.message;
                } else {
                    // Check if all processes are complete
                    const allComplete = processes.every(p => 
                        p.status === 'complete' || p.status === 'error');
                    
                    if (allComplete) {
                        // All done, refresh the page to show results
                        window.location.reload();
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
            });
    }
    
    // Update every 2 seconds
    setInterval(updateStatus, 2000);
    
    // Initial update
    updateStatus();
}); 