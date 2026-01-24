# Playing Card Detection PWA

A Progressive Web App that detects playing cards in real-time using your device's camera.

## Demo

Visit the live demo: **https://[your-username].github.io/card_detection_app2/**

## Features

- Real-time playing card detection using YOLOv8 model via Roboflow API
- PWA support - installable on mobile devices
- Vibration feedback when a card is detected (on supported devices)
- Configurable confidence threshold (default: 75%)
- Clean, mobile-first UI
- Works offline after initial load (for UI assets)

## Setup

### 1. Get a Roboflow API Key

1. Sign up for a free account at [app.roboflow.com](https://app.roboflow.com)
2. Go to Settings → API Keys
3. Copy your API key

### 2. Deploy to GitHub Pages

1. Fork or clone this repository
2. Go to repository Settings → Pages
3. Set Source to "GitHub Actions"
4. Push to main branch to trigger deployment

The app will be available at `https://[your-username].github.io/[repo-name]/`

### 3. Local Development

You need to serve the app via HTTPS (required for camera access). Options:

**Option A: Using Python**
```bash
python -m http.server 8000
```

**Option B: Using Node.js**
```bash
npx serve
```

**Option C: Using VS Code Live Server**
Install the "Live Server" extension and click "Go Live"

### 4. Enter Your API Key

On first launch, you'll be prompted to enter your Roboflow API key. The key is stored locally in your browser.

## Generating PWA Icons

Open `generate-icons.html` in a browser to generate PWA icons, then save them to the `icons/` folder.

## Usage

1. Allow camera access when prompted
2. Point your camera at a playing card
3. Hold steady until the card is detected with >75% confidence
4. The app will vibrate and display the detected card
5. Tap "Scan Another Card" to continue

## Configuration

Click the Settings button to:
- Adjust the detection confidence threshold
- Enable/disable vibration feedback
- Update your API key

## Technical Details

- Uses [Roboflow](https://roboflow.com) inference API with the `playing-cards-ow27d` model
- Detection runs every 150ms for responsive real-time detection
- Requires 2 consecutive detections of the same card for stability
- Camera defaults to back-facing (environment) for best results

## Browser Support

- Chrome (Desktop & Mobile)
- Safari (iOS 11+)
- Firefox
- Edge

Requires HTTPS for camera access (except localhost).

## Model Information

The app uses a YOLOv8 model trained on playing cards from [Roboflow Universe](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d). It can detect all 52 cards in a standard deck.

## License

MIT
