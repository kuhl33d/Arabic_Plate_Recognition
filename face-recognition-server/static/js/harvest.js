(() => {
    // Configuration
    const CONFIG = {
        imageCount: 10,
        captureInterval: 500, // ms between captures
        videoWidth: 640,
        videoHeight: 480
    };

    // DOM Elements
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startTraining');
    const usernameInput = document.getElementById('username');
    const captureStatus = document.getElementById('captureStatus');
    const imageCount = document.getElementById('imageCount');
    const progressBar = document.querySelector('.progress-bar');
    const trainingStatus = document.getElementById('trainingStatus');

    let captureCount = 0;
    let isCapturing = false;
    let ws = null;

    // Set up canvas
    canvas.width = CONFIG.videoWidth;
    canvas.height = CONFIG.videoHeight;

    // Initialize webcam
    async function initializeCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: CONFIG.videoWidth,
                    height: CONFIG.videoHeight
                }
            });
            video.srcObject = stream;
            await video.play();
            updateStatus('Camera initialized successfully', 'success');
        } catch (error) {
            updateStatus('Failed to access camera: ' + error.message, 'error');
        }
    }

    // WebSocket setup
    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/harvesting`);

        ws.onopen = () => {
            updateStatus('Connected to server', 'success');
        };

        ws.onmessage = (event) => {
            const response = JSON.parse(event.data);
            if (response.status === 'success') {
                captureCount++;
                updateProgress();
                
                if (captureCount >= CONFIG.imageCount) {
                    stopCapturing();
                    startTraining();
                }
            }
        };

        ws.onerror = (error) => {
            updateStatus('WebSocket error: ' + error.message, 'error');
        };

        ws.onclose = () => {
            updateStatus('Connection closed', 'info');
        };
    }

    function updateStatus(message, type) {
        captureStatus.textContent = message;
        captureStatus.className = `status ${type}`;
    }

    function updateProgress() {
        const progress = (captureCount / CONFIG.imageCount) * 100;
        progressBar.style.width = `${progress}%`;
        imageCount.textContent = `Images captured: ${captureCount}/${CONFIG.imageCount}`;
    }

    function captureImage() {
        if (!isCapturing) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(blob);
            }
        }, 'image/jpeg', 0.9);
    }

    function startCapturing() {
        const username = usernameInput.value.trim();
        if (!username) {
            updateStatus('Please enter a username', 'error');
            return;
        }

        // Send username to server
        fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `label=${encodeURIComponent(username)}`
        })
        .then(response => {
            if (!response.ok) throw new Error('Failed to register username');
            
            isCapturing = true;
            captureCount = 0;
            setupWebSocket();
            updateProgress();
            
            // Start capture loop
            const captureLoop = setInterval(() => {
                if (captureCount >= CONFIG.imageCount || !isCapturing) {
                    clearInterval(captureLoop);
                } else {
                    captureImage();
                }
            }, CONFIG.captureInterval);
        })
        .catch(error => {
            updateStatus('Error: ' + error.message, 'error');
        });
    }

    function stopCapturing() {
        isCapturing = false;
        if (ws) ws.close();
    }

    function startTraining() {
        updateStatus('Starting model training...', 'info');
        
        fetch('/train', {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) throw new Error('Training failed');
            updateStatus('Training completed successfully!', 'success');
            trainingStatus.textContent = 'Model trained and ready for use';
        })
        .catch(error => {
            updateStatus('Training error: ' + error.message, 'error');
        });
    }

    // Event Listeners
    startButton.addEventListener('click', () => {
        if (!isCapturing) {
            startCapturing();
            startButton.disabled = true;
        }
    });

    // Initialize
    initializeCamera();
})();