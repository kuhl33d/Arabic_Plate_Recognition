(() => {
    // Configuration
    const CONFIG = {
        updateInterval: 250,
        canvasWidth: 320,
        canvasHeight: 240
    };

    // DOM Elements
    const video = document.querySelector('video');
    const canvas = document.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const statusDiv = document.querySelector('#status');

    // Setup canvas
    ctx.strokeStyle = '#ff0';
    ctx.lineWidth = 2;

    // WebSocket setup
    let ws = null;

    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/facedetector`);

        ws.onopen = () => {
            console.log("WebSocket connection opened");
            updateStatus("Connected to face detection server", "success");
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed");
            updateStatus("Connection closed. Reconnecting...", "error");
            setTimeout(setupWebSocket, 1000);
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            updateStatus("Connection error. Please refresh the page.", "error");
        };

        ws.onmessage = (e) => {
            try {
                const openCvCoords = JSON.parse(e.data)[0];
                drawFaceRect(openCvCoords);
            } catch (error) {
                console.error("Error processing message:", error);
            }
        };
    }

    function drawFaceRect(coords) {
        // Clear previous drawing
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw current frame
        ctx.drawImage(video, 0, 0, CONFIG.canvasWidth, CONFIG.canvasHeight);
        // Draw face rectangle
        ctx.strokeRect(coords[0], coords[1], coords[2], coords[3]);
    }

    function updateFrame() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        ctx.drawImage(video, 0, 0, CONFIG.canvasWidth, CONFIG.canvasHeight);
        canvas.toBlob(blob => ws.send(blob), 'image/jpeg');
    }

    function updateStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }

    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: CONFIG.canvasWidth,
                    height: CONFIG.canvasHeight
                },
                audio: false
            });

            video.srcObject = stream;
            await video.play();

            // Start updating frames
            setInterval(updateFrame, CONFIG.updateInterval);
            updateStatus("Camera initialized successfully", "success");

        } catch (error) {
            console.error("Camera access error:", error);
            updateStatus("Failed to access camera. Please ensure camera permissions are granted.", "error");
        }
    }

    // Initialize application
    function initialize() {
        setupWebSocket();
        initCamera().catch(error => {
            console.error("Initialization error:", error);
            updateStatus("Failed to initialize application", "error");
        });
    }

    // Start the application
    initialize();
})();