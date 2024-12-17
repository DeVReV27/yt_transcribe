# YouTube Video Transcriber

A Streamlit application that transcribes YouTube videos into text. This tool allows users to easily convert YouTube video audio into written text format.

## Features

- YouTube video URL input
- Automatic transcription
- Clean and user-friendly interface
- Streamlit-based web application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DeVReV27/yt_transcribe.git
cd yt_transcribe
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your required API keys and configurations

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser and navigate to the provided local URL (typically http://localhost:8501).

## Requirements

See `requirements.txt` for a full list of dependencies.

## Configuration

The application uses environment variables for configuration. Create a `.env` file and a `.streamlit/secrets.toml` file with the necessary credentials and settings.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome!
