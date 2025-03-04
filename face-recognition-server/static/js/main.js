(() => {
    // Configuration
    const CONFIG = {
        imageCount: 10,
        captureInterval: 500,
        videoWidth: 640,
        videoHeight: 480,
        confidenceThreshold: 100  // Maximum confidence value for recognition
    };

    // State
    let currentVideo = null;
    let currentMode = null;
    let isCapturing = false;
    let captureCount = 0;
    let ws = null;

    // DOM Elements
    const modeButtons = document.querySelectorAll('.mode-button');
    const sections = document.querySelectorAll('.section');
    
    // Initialize mode selection
    modeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const mode = button.dataset.mode;
            switchMode(mode);
        });
    });

    function switchMode(mode) {
        // Update buttons
        modeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Update sections
        sections.forEach(section => {
            section.style.display = section.id === `${mode}Section` ? 'block' : 'none';
        });

        // Stop previous video if any
        if (currentVideo) {
            stopVideo(currentVideo);
        }

        // Initialize new mode
        currentMode = mode;
        if (mode === 'train') {
            initializeTraining();
        } else {
            initializePrediction();
        }
    }

    async function initializeCamera(videoElement) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: CONFIG.videoWidth,
                    height: CONFIG.videoHeight
                }
            });
            videoElement.srcObject = stream;
            await videoElement.play();
            currentVideo = videoElement;
            return true;
        } catch (error) {
            console.error("Camera access error:", error);
            updateStatus(`Camera error: ${error.message}`, 'error');
            return false;
        }
    }

    function stopVideo(videoElement) {
        if (videoElement && videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
    }

    function setupWebSocket(path) {
        if (ws) {
            ws.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/${path}`);

        ws.onopen = () => {
            console.log(`WebSocket connected to ${path}`);
            updateStatus('Connected to server', 'success');
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('Connection error', 'error');
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            updateStatus('Connection closed', 'info');
        };

        return ws;
    }

    // Training Mode Functions
    function initializeTraining() {
        const video = document.querySelector('#videoElement');
        const startButton = document.querySelector('#startTraining');
        const usernameInput = document.querySelector('#username');

        initializeCamera(video);

        startButton.onclick = () => {
            const username = usernameInput.value.trim();
            if (username) {
                startTraining(username);
            } else {
                updateStatus('Please enter a name', 'error');
            }
        };
    }

    function startTraining(username) {
        isCapturing = true;
        captureCount = 0;
        updateStatus('Starting training...', 'info');

        // Register username
        fetch('/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `label=${encodeURIComponent(username)}`
        })
        .then(response => {
            if (!response.ok) throw new Error('Failed to register username');
            
            // Setup WebSocket for image capture
            const ws = setupWebSocket('harvesting');
            ws.onmessage = (event) => {
                const response = JSON.parse(event.data);
                
                if (response.status === 'success') {
                    captureCount++;
                    updateProgress(captureCount);
                    
                    if (captureCount >= CONFIG.imageCount) {
                        isCapturing = false;
                        ws.close();
                        startModelTraining();
                    }
                } else if (response.status === 'error') {
                    updateStatus(response.message, 'error');
                }
            };

            // Start capturing images
            const captureInterval = setInterval(() => {
                if (!isCapturing || captureCount >= CONFIG.imageCount) {
                    clearInterval(captureInterval);
                    return;
                }
                captureImage();
            }, CONFIG.captureInterval);
        })
        .catch(error => {
            updateStatus(`Error: ${error.message}`, 'error');
            isCapturing = false;
        });
    }

    function captureImage() {
        if (!currentVideo || !ws || ws.readyState !== WebSocket.OPEN) return;

        const canvas = document.createElement('canvas');
        canvas.width = CONFIG.videoWidth;
        canvas.height = CONFIG.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(currentVideo, 0, 0);
        
        canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.9);
    }

    function updateProgress(count) {
        const progress = (count / CONFIG.imageCount) * 100;
        document.querySelector('.progress-bar').style.width = `${progress}%`;
        document.querySelector('.progress-bar').textContent = `${Math.round(progress)}%`;
        document.querySelector('#imageCount').textContent = 
            `Images captured: ${count}/${CONFIG.imageCount}`;
    }

    function startModelTraining() {
        updateStatus('Training model...', 'info');
        fetch('/train', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateStatus('Training completed successfully!', 'success');
                    setTimeout(() => switchMode('predict'), 2000);
                } else {
                    throw new Error(data.message || 'Training failed');
                }
            })
            .catch(error => {
                updateStatus(`Training error: ${error.message}`, 'error');
            });
    }

    // Prediction Mode Functions
    function initializePrediction() {
        const video = document.querySelector('#videoElement2');
        initializeCamera(video).then(success => {
            if (success) {
                startPrediction();
            }
        });
    }

    function startPrediction() {
        const ws = setupWebSocket('predict');
        const canvas = document.querySelector('#canvasOverlay2');
        const ctx = canvas.getContext('2d');
        
        canvas.width = CONFIG.videoWidth;
        canvas.height = CONFIG.videoHeight;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Clear previous drawing
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.status === 'success' && data.result && data.result.face) {
                const face = data.result.face;
                const confidence = Math.max(0, 100 - face.confidence);
                
                // Draw rectangle around face
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.strokeRect(
                    face.coords.x,
                    face.coords.y,
                    face.coords.width,
                    face.coords.height
                );

                // Draw name and confidence
                ctx.fillStyle = '#00ff00';
                ctx.font = '16px Arial';
                ctx.fillText(
                    `${face.name} (${confidence.toFixed(1)}%)`,
                    face.coords.x,
                    face.coords.y - 5
                );

                // Update prediction result
                document.querySelector('#predictionResult').innerHTML = `
                    <div class="prediction-box">
                        <h3>Detected: ${face.name}</h3>
                        <p>Confidence: ${confidence.toFixed(1)}%</p>
                    </div>
                `;
                document.querySelector('#predictionResult').style.display = 'block';
            } else {
                document.querySelector('#predictionResult').style.display = 'none';
            }
        };

        // Start sending frames for prediction
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                captureImage();
            }
        }, 100);
    }

    function updateStatus(message, type) {
        const status = document.querySelector(
            currentMode === 'train' ? '#trainingStatus' : '#predictStatus'
        );
        status.textContent = message;
        status.className = `status ${type}`;
    }

    // Start with training mode by default
    switchMode('train');
})();