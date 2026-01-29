# Playing Card Detection PWA

A Progressive Web App that detects playing cards in real-time using your device's camera with local AI inference.

## Demo

Visit the live demo: **https://[your-username].github.io/card_detection_app2/**

## Features

- Real-time playing card detection using YOLOv8 model running locally in browser
- No API key required - model runs entirely on your device
- PWA support - installable on mobile devices
- Works offline after initial model download
- Vibration feedback when a card is detected (on supported devices)
- Configurable vibration duration (up to 3 seconds)
- Configurable confidence threshold (default: 75%)
- Front/back camera switching
- **API Auto-Update**: Automatically send detected card data to a configured API endpoint
- Clean, mobile-first UI

## Setup

### 1. Deploy to GitHub Pages

1. Fork or clone this repository
2. Go to repository Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select "main" branch and "/ (root)" folder
5. Click Save

The app will be available at `https://[your-username].github.io/[repo-name]/`

### 2. Local Development

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

## Usage

1. Allow camera access when prompted
2. Wait for the model to load (first time may take a moment)
3. Point your camera at a playing card
4. Hold steady until the card is detected with >75% confidence
5. The app will vibrate and display the detected card
6. Tap "Scan Another Card" to continue

## Configuration

Click the Settings button to:
- Adjust the detection confidence threshold
- Enable/disable vibration feedback
- Adjust vibration duration
- **Auto Update API**: Enable automatic sending of detected cards to an external API
  - API Endpoint: URL to receive POST requests
  - API Key: Authentication key sent via `x-api-key` header

### API Auto-Update Format

When enabled, detected cards are sent as POST requests with:
- Header: `x-api-key` (your configured API key)
- Header: `Content-Type: application/json`
- Body: `{"suit": "1-4", "rank": "1-9,a-d"}`
  - Suit: 1=Diamonds, 2=Clubs, 3=Hearts, 4=Spades
  - Rank: 1=Ace, 2-9, a=10, b=Jack, c=Queen, d=King

## Technical Details

- Uses ONNX Runtime Web for local model inference
- YOLOv8m model trained on synthetic playing card data
- Detection runs continuously for real-time responsiveness
- Immediate detection when confidence threshold is met
- Model size: ~99MB (cached after first load for offline use)
- Input resolution: 320x320 pixels (optimized for speed)

## Browser Support

- Chrome (Desktop & Mobile)
- Safari (iOS 11+)
- Firefox
- Edge

Requires HTTPS for camera access (except localhost).

## Model Information

The app uses a YOLOv8m model from the [Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection) project, trained on synthetic playing card images. It can detect all 52 cards in a standard deck.

## License

MIT
