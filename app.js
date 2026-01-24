/**
 * Playing Card Detection PWA
 * Detects playing cards using camera with Roboflow API or local TensorFlow.js inference
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
            apiKey: ''
        };

        // State
        this.stream = null;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.lastDetections = [];
        this.consecutiveDetections = 0;
        this.requiredConsecutive = 2; // Require 2 consecutive detections for stability
        this.lastDetectedClass = null;
        this.processingFrame = false;

        // Canvas for frame capture
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d');

        // Card mapping
        this.cardClasses = this.initializeCardClasses();
        this.suitSymbols = {
            'hearts': '\u2665',
            'diamonds': '\u2666',
            'clubs': '\u2663',
            'spades': '\u2660',
            'h': '\u2665',
            'd': '\u2666',
            'c': '\u2663',
            's': '\u2660',
            'H': '\u2665',
            'D': '\u2666',
            'C': '\u2663',
            'S': '\u2660'
        };

        this.init();
    }

    initializeCardClasses() {
        // Standard 52-card deck mapping
        const suits = ['C', 'D', 'H', 'S'];
        const suitNames = { 'C': 'Clubs', 'D': 'Diamonds', 'H': 'Hearts', 'S': 'Spades' };
        const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
        const rankNames = {
            '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five',
            '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
            '10': 'Ten', 'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'A': 'Ace'
        };

        const classes = {};

        // Handle various naming conventions from different models
        for (const suit of suits) {
            for (const rank of ranks) {
                // Format: "10C", "AS", "KH", etc.
                const key1 = `${rank}${suit}`;
                // Format: "10-C", "A-S", "K-H", etc.
                const key2 = `${rank}-${suit}`;
                // Format: "10c", "as", "kh", etc.
                const key3 = `${rank}${suit.toLowerCase()}`;
                // Format with 'of'
                const key4 = `${rank} of ${suitNames[suit]}`;
                // Lowercase format
                const key5 = key1.toLowerCase();
                // Format: "10-of-clubs"
                const key6 = `${rank}-of-${suitNames[suit].toLowerCase()}`;

                const cardInfo = {
                    rank: rank,
                    suit: suit.toLowerCase(),
                    suitName: suitNames[suit],
                    rankName: rankNames[rank],
                    displayName: `${rankNames[rank]} of ${suitNames[suit]}`
                };

                classes[key1] = cardInfo;
                classes[key2] = cardInfo;
                classes[key3] = cardInfo;
                classes[key4] = cardInfo;
                classes[key5] = cardInfo;
                classes[key6] = cardInfo;
            }
        }

        return classes;
    }

    async init() {
        this.loadSettings();
        this.setupEventListeners();
        this.registerServiceWorker();

        // Check if API key is set
        if (!this.settings.apiKey) {
            this.showSetupWizard();
        } else {
            this.hideSetupWizard();
            await this.startCamera();
        }
    }

    showSetupWizard() {
        this.setupWizard.classList.remove('hidden');
        this.hideLoading();
    }

    hideSetupWizard() {
        this.setupWizard.classList.add('hidden');
    }

    async completeSetup() {
        const apiKeyInput = document.getElementById('setupApiKeyInput');
        const apiKey = apiKeyInput.value.trim();

        if (!apiKey) {
            apiKeyInput.focus();
            apiKeyInput.style.borderColor = '#ef4444';
            return;
        }

        // Save the API key
        this.settings.apiKey = apiKey;
        localStorage.setItem('cardDetectionSettings', JSON.stringify(this.settings));

        // Update settings panel too
        document.getElementById('apiKeyInput').value = apiKey;

        // Hide wizard and start camera
        this.hideSetupWizard();
        await this.startCamera();
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('cardDetectionSettings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }

            // Update UI
            document.getElementById('confidenceThreshold').value = this.settings.confidenceThreshold;
            document.getElementById('thresholdValue').textContent = `${this.settings.confidenceThreshold}%`;
            document.getElementById('vibrationToggle').checked = this.settings.vibrationEnabled;
            document.getElementById('apiKeyInput').value = this.settings.apiKey;
        } catch (e) {
            console.error('Failed to load settings:', e);
        }
    }

    saveSettings() {
        try {
            const oldApiKey = this.settings.apiKey;

            this.settings.confidenceThreshold = parseInt(document.getElementById('confidenceThreshold').value);
            this.settings.vibrationEnabled = document.getElementById('vibrationToggle').checked;
            this.settings.apiKey = document.getElementById('apiKeyInput').value.trim();

            localStorage.setItem('cardDetectionSettings', JSON.stringify(this.settings));
            this.hideSettings();

            // Restart camera if API key changed
            if (oldApiKey !== this.settings.apiKey && this.settings.apiKey) {
                this.resetToCamera();
            }
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
    }

    setupEventListeners() {
        // Settings
        document.getElementById('settingsBtn').addEventListener('click', () => this.showSettings());
        document.getElementById('closeSettingsBtn').addEventListener('click', () => this.hideSettings());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());

        // Setup wizard
        document.getElementById('startDetectionBtn').addEventListener('click', () => this.completeSetup());
        document.getElementById('setupApiKeyInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.completeSetup();
        });
        document.getElementById('setupApiKeyInput').addEventListener('input', (e) => {
            e.target.style.borderColor = ''; // Reset error state on input
        });

        // Threshold slider
        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            document.getElementById('thresholdValue').textContent = `${e.target.value}%`;
        });

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
                       !this.isDetecting) {
                this.startDetection();
            }
        });
    }

    async startCamera() {
        this.showLoading();

        try {
            // Stop any existing stream
            this.stopCamera();

            // Request camera with optimal settings for card detection
            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' }, // Prefer back camera
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 }
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            await new Promise((resolve, reject) => {
                this.video.onloadedmetadata = () => {
                    this.video.play()
                        .then(resolve)
                        .catch(reject);
                };
                this.video.onerror = reject;
            });

            // Set canvas sizes to match video
            const width = this.video.videoWidth;
            const height = this.video.videoHeight;

            this.overlay.width = width;
            this.overlay.height = height;
            this.captureCanvas.width = width;
            this.captureCanvas.height = height;

            this.hideLoading();
            this.showCamera();
            this.updateStatus('Detecting', 'detecting');
            this.startDetection();

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

    startDetection() {
        if (this.isDetecting) return;
        this.isDetecting = true;

        // Run detection every 150ms for good balance of speed and performance
        this.detectionInterval = setInterval(() => this.detectCard(), 150);
    }

    stopDetection() {
        this.isDetecting = false;
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }

    async detectCard() {
        if (!this.isDetecting || !this.video.videoWidth || this.processingFrame) return;

        this.processingFrame = true;

        try {
            // Capture frame from video
            this.captureCtx.drawImage(this.video, 0, 0);

            // Convert to base64 JPEG (lower quality for faster transfer)
            const imageData = this.captureCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];

            // Send to Roboflow API
            const predictions = await this.callRoboflowAPI(imageData);

            if (predictions && predictions.length > 0) {
                // Find best prediction above minimum threshold
                const validPredictions = predictions.filter(p => p.confidence >= 0.3);

                if (validPredictions.length > 0) {
                    const bestPrediction = validPredictions.reduce((best, current) =>
                        current.confidence > best.confidence ? current : best
                    );

                    const confidence = bestPrediction.confidence * 100;
                    this.updateConfidenceBar(confidence);

                    // Check if same card detected consecutively
                    if (this.lastDetectedClass === bestPrediction.class) {
                        this.consecutiveDetections++;
                    } else {
                        this.consecutiveDetections = 1;
                        this.lastDetectedClass = bestPrediction.class;
                    }

                    // Draw bounding box
                    this.drawDetection(bestPrediction);

                    // Check threshold with consecutive requirement for stability
                    if (confidence >= this.settings.confidenceThreshold &&
                        this.consecutiveDetections >= this.requiredConsecutive) {
                        this.cardDetected(bestPrediction);
                        return;
                    }
                } else {
                    this.resetDetectionState();
                }
            } else {
                this.resetDetectionState();
            }

        } catch (error) {
            console.error('Detection error:', error);
            // Don't reset on API errors, just continue
        } finally {
            this.processingFrame = false;
        }
    }

    resetDetectionState() {
        this.updateConfidenceBar(0);
        this.consecutiveDetections = 0;
        this.lastDetectedClass = null;
        this.clearOverlay();
    }

    async callRoboflowAPI(imageBase64) {
        // Check for API key
        if (!this.settings.apiKey) {
            throw new Error('No API key configured');
        }

        // Using Roboflow's playing cards model
        const model = 'playing-cards-ow27d';
        const version = '4';

        const url = `https://detect.roboflow.com/${model}/${version}?api_key=${this.settings.apiKey}&confidence=30&overlap=30`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: imageBase64
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('API error:', response.status, errorText);

            if (response.status === 401 || response.status === 403) {
                throw new Error('Invalid API key');
            }
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();
        return result.predictions || [];
    }

    drawDetection(prediction) {
        const ctx = this.overlay.getContext('2d');
        ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);

        // Coordinates from Roboflow are center-based
        const x = prediction.x - prediction.width / 2;
        const y = prediction.y - prediction.height / 2;
        const width = prediction.width;
        const height = prediction.height;

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

        // Draw label background
        const label = `${prediction.class} ${Math.round(prediction.confidence * 100)}%`;
        ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const labelPadding = 8;
        const labelHeight = 28;
        const labelY = y - labelHeight - 4;

        ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
        ctx.beginPath();
        ctx.roundRect(x, labelY, textWidth + labelPadding * 2, labelHeight, 4);
        ctx.fill();

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + labelPadding, labelY + 19);
    }

    clearOverlay() {
        const ctx = this.overlay.getContext('2d');
        ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    }

    updateConfidenceBar(confidence) {
        this.confidenceFill.style.width = `${Math.min(confidence, 100)}%`;

        // Update color based on confidence
        if (confidence >= this.settings.confidenceThreshold) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
        } else if (confidence >= 50) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
        } else {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #6366f1, #4f46e5)';
        }
    }

    cardDetected(prediction) {
        this.stopDetection();
        this.stopCamera();

        // Vibrate device
        this.vibrate();

        // Parse card info
        const cardInfo = this.parseCardClass(prediction.class);
        const confidence = Math.round(prediction.confidence * 100);

        // Update result display
        this.displayCard(cardInfo);
        this.cardName.textContent = cardInfo.displayName;
        this.confidenceValue.textContent = `${confidence}%`;

        // Show result
        this.showResult();
        this.updateStatus('Detected', 'ready');
    }

    parseCardClass(className) {
        // Try direct lookup first
        if (this.cardClasses[className]) {
            return this.cardClasses[className];
        }

        // Normalize the class name
        let normalized = className.toUpperCase().replace(/[-_\s]/g, '');

        // Try normalized lookup
        if (this.cardClasses[normalized]) {
            return this.cardClasses[normalized];
        }

        // Match patterns like "10H", "AS", "KD", etc.
        const match = normalized.match(/^(10|[2-9]|J|Q|K|A)([HDCS])$/);
        if (match) {
            const rank = match[1];
            const suit = match[2];
            const key = `${rank}${suit}`;
            if (this.cardClasses[key]) {
                return this.cardClasses[key];
            }
        }

        // Try reverse pattern (suit first)
        const reverseMatch = normalized.match(/^([HDCS])(10|[2-9]|J|Q|K|A)$/);
        if (reverseMatch) {
            const suit = reverseMatch[1];
            const rank = reverseMatch[2];
            const key = `${rank}${suit}`;
            if (this.cardClasses[key]) {
                return this.cardClasses[key];
            }
        }

        // Fallback: return raw class name
        return {
            rank: '?',
            suit: 'spades',
            suitName: 'Unknown',
            rankName: className,
            displayName: className
        };
    }

    displayCard(cardInfo) {
        const suitSymbol = this.suitSymbols[cardInfo.suit] || '?';
        const rank = cardInfo.rank;
        const isRed = cardInfo.suit === 'h' || cardInfo.suit === 'hearts' ||
                      cardInfo.suit === 'd' || cardInfo.suit === 'diamonds';

        // Add suit class for color
        this.cardDisplay.className = 'detected-card ' + (isRed ? 'red-suit' : 'black-suit');

        this.cardDisplay.innerHTML = `
            <span class="card-rank">${rank}</span>
            <span class="card-suit">${suitSymbol}</span>
            <span class="card-rank-bottom">${rank}</span>
        `;
    }

    vibrate() {
        if (this.settings.vibrationEnabled && 'vibrate' in navigator) {
            // Short vibration pattern: vibrate 100ms, pause 50ms, vibrate 100ms
            navigator.vibrate([100, 50, 100]);
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
        this.lastDetections = [];
        this.showCamera();
        this.startCamera();
    }

    // PWA Support
    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('service-worker.js');
                console.log('ServiceWorker registered:', registration.scope);

                // Check for updates
                registration.addEventListener('updatefound', () => {
                    console.log('New service worker available');
                });
            } catch (error) {
                console.error('ServiceWorker registration failed:', error);
            }
        }

        // Handle PWA install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            this.deferredPrompt = e;
            this.showInstallBanner();
        });
    }

    showInstallBanner() {
        // Check if already shown recently
        const lastShown = localStorage.getItem('installBannerShown');
        if (lastShown && Date.now() - parseInt(lastShown) < 86400000) { // 24 hours
            return;
        }

        const banner = document.createElement('div');
        banner.className = 'install-banner';
        banner.innerHTML = `
            <div class="install-banner-text">
                <strong>Install App</strong>
                <span>Add to home screen for quick access</span>
            </div>
            <button class="btn btn-primary install-btn">Install</button>
            <button class="close-btn dismiss-btn">&times;</button>
        `;

        document.body.appendChild(banner);

        banner.querySelector('.install-btn').addEventListener('click', async () => {
            if (this.deferredPrompt) {
                this.deferredPrompt.prompt();
                const result = await this.deferredPrompt.userChoice;
                console.log('Install prompt result:', result.outcome);
                this.deferredPrompt = null;
            }
            banner.remove();
        });

        banner.querySelector('.dismiss-btn').addEventListener('click', () => {
            localStorage.setItem('installBannerShown', Date.now().toString());
            banner.remove();
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.cardApp = new CardDetectionApp();
});
