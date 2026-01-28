/**
 * Playing Card Detection PWA
 * Uses local YOLOv8 ONNX model for real-time card detection
 */

class CardDetectionApp {
    constructor() {
        // DOM Elements
        this.video = document.getElementById('video');
        this.overlay = document.getElementById('overlay');
        this.cameraContainer = document.getElementById('cameraContainer');
        this.resultContainer = document.getElementById('resultContainer');
        this.errorContainer = document.getElementById('errorContainer');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.statusBadge = document.getElementById('statusBadge');
        this.detectionIndicator = document.getElementById('detectionIndicator');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.cardDisplay = document.getElementById('detectedCard');
        this.cardName = document.getElementById('cardName');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.setupWizard = document.getElementById('setupWizard');

        // Settings
        this.settings = {
            confidenceThreshold: 75,
            vibrationEnabled: true,
            vibrationDuration: 200,
            facingMode: 'environment'
        };

        // State
        this.stream = null;
        this.isDetecting = false;
        this.animationFrameId = null;
        this.lastFrameTime = 0;
        this.consecutiveDetections = 0;
        this.requiredConsecutive = 1; // Immediate detection on first match
        this.lastDetectedClass = null;
        this.processingFrame = false;
        this.modelLoaded = false;
        this.session = null;

        // Model configuration
        this.modelPath = 'models/yolov8m_synthetic.onnx';
        this.inputSize = 320; // Smaller input for faster inference
        this.confThreshold = 0.5; // Higher threshold for faster filtering

        // YOLOv8 class names from the synthetic model
        this.classNames = [
            '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s',
            '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
            '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
            '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s',
            '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As',
            'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
            'Qc', 'Qd', 'Qh', 'Qs'
        ];

        // These will be initialized after model loads (to get actual input size)
        this.inputCanvas = null;
        this.inputCtx = null;
        this.tensorBuffer = null;
        this.pixelCount = 0;

        // Card display mapping
        this.suitSymbols = {
            'c': '\u2663', 'd': '\u2666', 'h': '\u2665', 's': '\u2660'
        };
        this.suitNames = {
            'c': 'Clubs', 'd': 'Diamonds', 'h': 'Hearts', 's': 'Spades'
        };
        this.rankNames = {
            '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
            '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
            '10': 'Ten', 'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'A': 'Ace'
        };

        this.init();
    }

    async init() {
        this.loadSettings();
        this.setupEventListeners();
        this.registerServiceWorker();

        // Hide setup wizard (no API key needed for local model)
        this.hideSetupWizard();

        // Load model and camera in parallel for faster startup
        const modelPromise = this.loadModel();
        const cameraPromise = this.startCamera();

        // Wait for both to complete
        await Promise.all([modelPromise, cameraPromise]);

        // Start detection if both are ready
        if (this.modelLoaded && this.stream) {
            this.startDetection();
        }
    }

    async loadModel() {
        this.showLoading();
        const loadingText = document.getElementById('loadingText');
        loadingText.textContent = 'Loading AI model...';

        try {
            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime Web not loaded. Please check your internet connection.');
            }

            // Configure ONNX Runtime for maximum performance
            ort.env.wasm.wasmPaths = 'https://unpkg.com/onnxruntime-web/dist/';
            ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
            ort.env.wasm.simd = true;

            loadingText.textContent = 'Downloading model (~99MB)...';
            console.log('Fetching model from:', this.modelPath);

            // Fetch model with progress tracking
            const response = await fetch(this.modelPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            }

            const contentLength = response.headers.get('content-length');
            const total = contentLength ? parseInt(contentLength, 10) : 0;
            console.log('Model size:', total, 'bytes');

            // Read the response as array buffer with progress
            const reader = response.body.getReader();
            const chunks = [];
            let received = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                chunks.push(value);
                received += value.length;

                if (total > 0) {
                    const percent = Math.round((received / total) * 100);
                    loadingText.textContent = `Downloading model... ${percent}%`;
                }
            }

            // Combine chunks into single ArrayBuffer
            const modelData = new Uint8Array(received);
            let offset = 0;
            for (const chunk of chunks) {
                modelData.set(chunk, offset);
                offset += chunk.length;
            }

            loadingText.textContent = 'Initializing AI model...';
            console.log('Model downloaded, creating session...');

            // Create inference session from array buffer
            // Try WebGL (GPU) first, fallback to WASM (CPU)
            this.session = await ort.InferenceSession.create(modelData.buffer, {
                executionProviders: ['webgl', 'wasm'],
                graphOptimizationLevel: 'all'
            });
            console.log('Using execution provider:', this.session.handler?.name || 'unknown');

            // Initialize preprocessing buffers
            this.initPreprocessing();

            this.modelLoaded = true;
            console.log('Model loaded successfully');
            console.log('Input size:', this.inputSize);

        } catch (error) {
            console.error('Failed to load model:', error);
            this.showError(
                'Model Loading Failed',
                `Could not load the AI model: ${error.message}`
            );
        }
    }

    initPreprocessing() {
        // Initialize canvas and buffers for the input size
        this.inputCanvas = document.createElement('canvas');
        this.inputCanvas.width = this.inputSize;
        this.inputCanvas.height = this.inputSize;
        this.inputCtx = this.inputCanvas.getContext('2d', {
            willReadFrequently: true,
            alpha: false
        });
        this.tensorBuffer = new Float32Array(3 * this.inputSize * this.inputSize);
        this.pixelCount = this.inputSize * this.inputSize;
    }

    hideSetupWizard() {
        if (this.setupWizard) {
            this.setupWizard.classList.add('hidden');
        }
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('cardDetectionSettings');
            if (saved) {
                const parsed = JSON.parse(saved);
                // Don't load apiKey since we don't need it
                this.settings = {
                    ...this.settings,
                    confidenceThreshold: parsed.confidenceThreshold || 75,
                    vibrationEnabled: parsed.vibrationEnabled !== false,
                    vibrationDuration: parsed.vibrationDuration || 200,
                    facingMode: parsed.facingMode || 'environment'
                };
            }

            // Update UI
            document.getElementById('confidenceThreshold').value = this.settings.confidenceThreshold;
            document.getElementById('thresholdValue').textContent = `${this.settings.confidenceThreshold}%`;
            document.getElementById('vibrationToggle').checked = this.settings.vibrationEnabled;
            document.getElementById('vibrationDuration').value = this.settings.vibrationDuration;
            document.getElementById('vibrationDurationValue').textContent = `${(this.settings.vibrationDuration / 1000).toFixed(1)}s`;

            this.updateVibrationDurationVisibility();
        } catch (e) {
            console.error('Failed to load settings:', e);
        }
    }

    updateVibrationDurationVisibility() {
        const vibrationDurationItem = document.getElementById('vibrationDurationItem');
        const vibrationEnabled = document.getElementById('vibrationToggle').checked;
        if (vibrationDurationItem) {
            vibrationDurationItem.style.display = vibrationEnabled ? 'block' : 'none';
        }
    }

    saveSettings() {
        try {
            this.settings.confidenceThreshold = parseInt(document.getElementById('confidenceThreshold').value);
            this.settings.vibrationEnabled = document.getElementById('vibrationToggle').checked;
            this.settings.vibrationDuration = parseInt(document.getElementById('vibrationDuration').value);

            localStorage.setItem('cardDetectionSettings', JSON.stringify(this.settings));
            this.hideSettings();
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
    }

    setupEventListeners() {
        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => this.showSettings());
        document.getElementById('closeSettingsBtn').addEventListener('click', () => this.hideSettings());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());

        // Threshold slider
        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
        });

        // Vibration toggle
        document.getElementById('vibrationToggle').addEventListener('change', () => {
            this.updateVibrationDurationVisibility();
        });

        // Vibration duration slider
        document.getElementById('vibrationDuration').addEventListener('input', (e) => {
            document.getElementById('vibrationDurationValue').textContent = `${(e.target.value / 1000).toFixed(1)}s`;
        });

        // Camera switch button
        document.getElementById('cameraSwitchBtn').addEventListener('click', () => this.switchCamera());

        // Scan again button
        document.getElementById('scanAgainBtn').addEventListener('click', () => this.resetToCamera());

        // Retry button
        document.getElementById('retryBtn').addEventListener('click', () => this.startCamera());

        // Handle visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isDetecting) {
                this.stopDetection();
            } else if (!document.hidden && this.cameraContainer &&
                !this.cameraContainer.classList.contains('hidden') &&
                !this.isDetecting && this.modelLoaded) {
                this.startDetection();
            }
        });
    }

    async startCamera() {
        try {
            this.stopCamera();

            // Optimized constraints for faster initialization
            const constraints = {
                video: {
                    facingMode: { ideal: this.settings.facingMode },
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            // Use canplay event for faster response than loadedmetadata
            await new Promise((resolve, reject) => {
                this.video.oncanplay = () => {
                    this.video.play().then(resolve).catch(reject);
                };
                this.video.onerror = reject;
                // Timeout fallback
                setTimeout(resolve, 2000);
            });

            // Set overlay canvas size
            this.overlay.width = this.video.videoWidth;
            this.overlay.height = this.video.videoHeight;

            this.hideLoading();
            this.showCamera();
            this.updateStatus('Detecting', 'detecting');

            // Start detection if model is already loaded
            if (this.modelLoaded) {
                this.startDetection();
            }

        } catch (error) {
            console.error('Camera error:', error);
            this.hideLoading();

            let message = 'Please allow camera access to detect playing cards.';
            if (error.name === 'NotAllowedError') {
                message = 'Camera access was denied. Please enable it in your browser settings.';
            } else if (error.name === 'NotFoundError') {
                message = 'No camera found on this device.';
            } else if (error.name === 'NotReadableError') {
                message = 'Camera is in use by another application.';
            }

            this.showError('Camera Access Required', message);
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.stopDetection();
    }

    async switchCamera() {
        this.settings.facingMode = this.settings.facingMode === 'environment' ? 'user' : 'environment';
        localStorage.setItem('cardDetectionSettings', JSON.stringify(this.settings));
        this.stopCamera();
        await this.startCamera();
    }

    startDetection() {
        if (this.isDetecting || !this.modelLoaded) return;
        this.isDetecting = true;
        this.lastFrameTime = 0;
        this.detectLoop();
    }

    stopDetection() {
        this.isDetecting = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    detectLoop() {
        if (!this.isDetecting) return;

        // Run detection as fast as possible
        if (!this.processingFrame && this.video.videoWidth > 0) {
            this.detectCard().then(() => {
                if (this.isDetecting) {
                    this.animationFrameId = requestAnimationFrame(() => this.detectLoop());
                }
            });
        } else {
            this.animationFrameId = requestAnimationFrame(() => this.detectLoop());
        }
    }

    async detectCard() {
        if (!this.modelLoaded || this.processingFrame) return;

        this.processingFrame = true;
        const startTime = performance.now();

        try {
            // Preprocess image
            const inputTensor = this.preprocessImage();

            // Run inference
            const outputs = await this.session.run({ images: inputTensor });

            // Get output tensor and process
            const output = outputs[this.session.outputNames[0]];
            const detections = this.processOutput(output);

            const inferenceTime = Math.round(performance.now() - startTime);
            if (inferenceTime > 100) {
                console.log(`Inference: ${inferenceTime}ms`);
            }

            if (detections.length > 0) {
                const best = detections[0]; // Already sorted by confidence
                const confidence = best.confidence * 100;
                this.updateConfidenceBar(confidence);

                // Immediate detection if threshold met
                if (confidence >= this.settings.confidenceThreshold) {
                    this.cardDetected(best);
                    return;
                }
            } else {
                this.updateConfidenceBar(0);
            }

        } catch (error) {
            console.error('Detection error:', error);
        } finally {
            this.processingFrame = false;
        }
    }

    preprocessImage() {
        const videoWidth = this.video.videoWidth;
        const videoHeight = this.video.videoHeight;

        // Calculate scaling to fit image in input size while maintaining aspect ratio
        const scale = Math.min(this.inputSize / videoWidth, this.inputSize / videoHeight);
        const scaledWidth = Math.round(videoWidth * scale);
        const scaledHeight = Math.round(videoHeight * scale);

        // Calculate padding to center the image
        const padX = (this.inputSize - scaledWidth) / 2;
        const padY = (this.inputSize - scaledHeight) / 2;

        // Store for coordinate conversion
        this.scale = scale;
        this.padX = padX;
        this.padY = padY;

        // Clear canvas and fill with gray (letterbox)
        this.inputCtx.fillStyle = '#808080';
        this.inputCtx.fillRect(0, 0, this.inputSize, this.inputSize);

        // Draw scaled video frame
        this.inputCtx.drawImage(
            this.video,
            0, 0, videoWidth, videoHeight,
            padX, padY, scaledWidth, scaledHeight
        );

        // Get image data and convert to tensor
        const imageData = this.inputCtx.getImageData(0, 0, this.inputSize, this.inputSize);
        const data = imageData.data;

        // Reuse pre-allocated buffer and optimize loop
        const float32Data = this.tensorBuffer;
        const pixelCount = this.pixelCount;

        // Optimized NCHW conversion with single loop
        for (let i = 0; i < pixelCount; i++) {
            const srcIdx = i * 4;
            float32Data[i] = data[srcIdx] * 0.00392156862745098;                 // R: /255
            float32Data[pixelCount + i] = data[srcIdx + 1] * 0.00392156862745098; // G
            float32Data[pixelCount * 2 + i] = data[srcIdx + 2] * 0.00392156862745098; // B
        }

        return new ort.Tensor('float32', float32Data, [1, 3, this.inputSize, this.inputSize]);
    }

    processOutput(outputTensor) {
        // YOLOv8 output shape: [1, 56, N] where 56 = 4 (bbox) + 52 (classes)
        const data = outputTensor.data;
        const numClasses = this.classNames.length;
        const numDetections = outputTensor.dims[2];
        const confThreshold = this.confThreshold;
        const classOffset = 4 * numDetections;

        // Find single best detection (skip NMS for speed)
        let bestScore = confThreshold;
        let bestIdx = -1;
        let bestClassIdx = 0;

        for (let i = 0; i < numDetections; i++) {
            // Find max class score for this detection
            for (let c = 0; c < numClasses; c++) {
                const score = data[classOffset + c * numDetections + i];
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = i;
                    bestClassIdx = c;
                }
            }
        }

        if (bestIdx === -1) return [];

        // Get bounding box for best detection
        const cx = data[bestIdx];
        const cy = data[numDetections + bestIdx];
        const w = data[numDetections * 2 + bestIdx];
        const h = data[numDetections * 3 + bestIdx];
        const halfW = w * 0.5;
        const halfH = h * 0.5;

        return [{
            x1: cx - halfW,
            y1: cy - halfH,
            x2: cx + halfW,
            y2: cy + halfH,
            confidence: bestScore,
            classIdx: bestClassIdx,
            class: this.classNames[bestClassIdx]
        }];
    }

    drawDetection(detection) {
        const ctx = this.overlay.getContext('2d');
        ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);

        // Convert coordinates from model space to video space
        const x1 = (detection.x1 - this.padX) / this.scale;
        const y1 = (detection.y1 - this.padY) / this.scale;
        const x2 = (detection.x2 - this.padX) / this.scale;
        const y2 = (detection.y2 - this.padY) / this.scale;

        const x = x1;
        const y = y1;
        const width = x2 - x1;
        const height = y2 - y1;

        // Draw bounding box with glow effect
        ctx.shadowColor = '#6366f1';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        ctx.shadowBlur = 0;

        // Draw corner accents
        const cornerLength = Math.min(width, height) * 0.2;
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#10b981';

        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerLength);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerLength, y + height);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + width - cornerLength, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerLength);
        ctx.stroke();

        // Draw label
        const cardInfo = this.parseCardClass(detection.class);
        const label = `${cardInfo.displayName} ${Math.round(detection.confidence * 100)}%`;
        ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const labelPadding = 8;
        const labelHeight = 28;
        const labelY = Math.max(0, y - labelHeight - 4);

        ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
        ctx.beginPath();
        ctx.roundRect(x, labelY, textWidth + labelPadding * 2, labelHeight, 4);
        ctx.fill();

        ctx.fillStyle = 'white';
        ctx.fillText(label, x + labelPadding, labelY + 19);
    }

    clearOverlay() {
        const ctx = this.overlay.getContext('2d');
        ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    }

    updateConfidenceBar(confidence) {
        this.confidenceFill.style.width = `${Math.min(confidence, 100)}%`;

        if (confidence >= this.settings.confidenceThreshold) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
        } else if (confidence >= 50) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
        } else {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #6366f1, #4f46e5)';
        }
    }

    resetDetectionState() {
        this.updateConfidenceBar(0);
        this.consecutiveDetections = 0;
        this.lastDetectedClass = null;
        this.clearOverlay();
    }

    cardDetected(detection) {
        this.stopDetection();
        this.stopCamera();
        this.vibrate();

        const cardInfo = this.parseCardClass(detection.class);
        const confidence = Math.round(detection.confidence * 100);

        this.displayCard(cardInfo);
        this.cardName.textContent = cardInfo.displayName;
        this.confidenceValue.textContent = `${confidence}%`;

        this.showResult();
        this.updateStatus('Detected', 'ready');
    }

    parseCardClass(className) {
        // Parse class names like "10c", "Ac", "Kh", etc.
        const match = className.match(/^(10|[2-9]|[AJQK])([cdhs])$/i);

        if (match) {
            const rank = match[1].toUpperCase();
            const suit = match[2].toLowerCase();

            return {
                rank: rank,
                suit: suit,
                suitName: this.suitNames[suit],
                rankName: this.rankNames[rank],
                displayName: `${this.rankNames[rank]} of ${this.suitNames[suit]}`
            };
        }

        return {
            rank: '?',
            suit: 's',
            suitName: 'Unknown',
            rankName: className,
            displayName: className
        };
    }

    displayCard(cardInfo) {
        const suitSymbol = this.suitSymbols[cardInfo.suit] || '?';
        const rank = cardInfo.rank;
        const isRed = cardInfo.suit === 'h' || cardInfo.suit === 'd';

        this.cardDisplay.className = 'detected-card ' + (isRed ? 'red-suit' : 'black-suit');

        this.cardDisplay.innerHTML = `
            <span class="card-rank">${rank}</span>
            <span class="card-suit">${suitSymbol}</span>
            <span class="card-rank-bottom">${rank}</span>
        `;
    }

    vibrate() {
        if (this.settings.vibrationEnabled && 'vibrate' in navigator) {
            const duration = this.settings.vibrationDuration || 200;
            navigator.vibrate(duration);
        }
    }

    // UI State Management
    showCamera() {
        this.cameraContainer.classList.remove('hidden');
        this.resultContainer.classList.add('hidden');
        this.errorContainer.classList.add('hidden');
    }

    showResult() {
        this.cameraContainer.classList.add('hidden');
        this.resultContainer.classList.remove('hidden');
        this.errorContainer.classList.add('hidden');
    }

    showError(title, message) {
        this.cameraContainer.classList.add('hidden');
        this.resultContainer.classList.add('hidden');
        this.errorContainer.classList.remove('hidden');

        document.getElementById('errorTitle').textContent = title;
        document.getElementById('errorMessage').textContent = message;
        this.updateStatus('Error', 'error');
    }

    showLoading() {
        this.loadingOverlay.classList.remove('hidden');
    }

    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }

    showSettings() {
        this.settingsPanel.classList.remove('hidden');
        setTimeout(() => this.settingsPanel.classList.add('visible'), 10);
    }

    hideSettings() {
        this.settingsPanel.classList.remove('visible');
        setTimeout(() => this.settingsPanel.classList.add('hidden'), 300);
    }

    updateStatus(text, type) {
        this.statusBadge.textContent = text;
        this.statusBadge.className = 'status-badge ' + (type || '');
    }

    resetToCamera() {
        this.consecutiveDetections = 0;
        this.lastDetectedClass = null;
        this.showCamera();
        this.startCamera();
    }

    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('service-worker.js');
                console.log('ServiceWorker registered:', registration.scope);
            } catch (error) {
                console.error('ServiceWorker registration failed:', error);
            }
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (typeof ort === 'undefined') {
        const loadingText = document.getElementById('loadingText');
        if (loadingText) {
            loadingText.textContent = 'Error: AI library failed to load. Please refresh.';
        }
        return;
    }

    window.cardApp = new CardDetectionApp();
});
