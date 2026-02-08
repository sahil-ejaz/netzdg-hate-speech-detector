# NetzDG Hate Speech Detector

A Streamlit application for detecting hate speech in German texts in compliance with the Network Enforcement Act (NetzDG).


## Features

- Analysis of German texts for hate speech content
- Rule-based scoring system
- Detection of xenophobia and misogyny
- Comparison with LLM classification (optional)
- Detailed explanations for each dimension
- Score visualizations

## Deployment on Streamlit Community Cloud

### Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free): https://share.streamlit.io/

### Steps

1. **Create a GitHub Repository:**
   - Go to https://github.com/new
   - Create a new repository (e.g., "netzdg-hate-speech-detector")
   - Upload all files:
     - `app.py`
     - `requirements.txt`
     - `.streamlit/config.toml` (if present)

2. **Deploy on Streamlit Community Cloud:**
   - Go to https://share.streamlit.io/
   - Click on "New app"
   - Select your GitHub repository
   - Select the branch (usually `main` or `master`)
   - Select `app.py` as Main file
   - Click on "Deploy"

3. **App is now publicly accessible!**
   - You will receive a URL like: `https://your-app-name.streamlit.app`
   - You can use this URL for your QR code

### Alternative: Local Deployment

If you want to host the app locally:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Usage

1. Enter German text in the text field
2. Select an LLM model for translation
3. Click on "Start Detection"
4. View the detailed analysis

## Technical Details

- Python 3.11+
- Streamlit for the web interface
- NLTK for text processing
- Plotly for visualizations
- AWS Bedrock API for LLM translations

## Browser Compatibility

**Important:** This app works best with Chrome, Firefox, or Edge. Safari on older versions like macOS Monterey 12.5.1 is not fully supported due to JavaScript compatibility issues.
