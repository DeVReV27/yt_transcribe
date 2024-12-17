# YouTube Video Transcriber

A Streamlit application that transcribes YouTube videos into text and allows users to rewrite/edit the transcriptions. This tool provides both automatic transcription and the ability to modify the generated text to improve clarity or accuracy.

## Features

- YouTube video URL input
- Automatic speech-to-text transcription
- Text editing capabilities to modify and improve transcripts
- Clean and user-friendly interface
- Streamlit-based web application

## Key Functionalities

1. **Video Transcription**: Convert YouTube video audio into written text automatically
2. **Transcript Editing**: Edit, rewrite, or refine the generated transcript
3. **User-Friendly Interface**: Simple and intuitive design for easy interaction

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

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)
3. Enter a YouTube video URL
4. Wait for the transcription to complete
5. Review and edit the transcript as needed

## Requirements

See `requirements.txt` for a full list of dependencies.

## Configuration

The application uses environment variables for configuration. Create a `.env` file and a `.streamlit/secrets.toml` file with the necessary credentials and settings.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome!
