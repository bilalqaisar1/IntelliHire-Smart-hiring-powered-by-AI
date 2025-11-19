# 🎤 Resume ↔ AI Voice Interview

An AI-powered application that performs intelligent resume screening and conducts interactive voice interviews using Streamlit, OpenAI APIs, and speech recognition.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Voice Features & Faster-Whisper](#voice-features--faster-whisper)
- [Usage Guide](#usage-guide)
- [Application Workflow](#application-workflow)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Resume Parsing**: Automatically extract text from PDF, DOCX, and TXT files
- **Job Matching**: AI-powered analysis of resume vs. job description alignment
- **Eligibility Assessment**: Intelligent screening based on qualifications
- **Voice Interview**: Interactive AI-conducted interviews with speech recognition
- **Real-time Feedback**: Instant evaluation and scoring of candidate responses
- **Detailed Analysis**: Comprehensive candidate evaluation reports
- **Session Persistence**: Multi-step workflow with state management

---

## 📁 Project Structure

```
Resume-U/
├── app.py                 # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── modules/
│   ├── __init__.py
│   ├── file_utils.py           # Resume text extraction (PDF, DOCX, TXT)
│   ├── openai_utils.py         # OpenAI API integration for analysis
│   ├── eligibility_manager.py  # Screening logic & eligibility checks
│   ├── enhanced_evaluation.py  # Answer evaluation & reference comparison
│   └── agent/
│       ├── __init__.py
│       └── orchestrator.py     # Main AI orchestration (resume vs job matching)
├── voice/
│   ├── __init__.py
│   ├── voice_interview_manager.py  # Interview flow & management
│   ├── speech_recognizer.py        # Faster-Whisper integration
│   ├── audio_processor.py          # Audio recording & processing
│   └── text_to_speech.py          # Voice output (TTS)
└── models/
    └── faster-whisper/         # Downloaded Faster-Whisper model files
```

---

## 🔧 Prerequisites

Before installation, ensure you have:

- **Python 3.9+** (recommended: 3.10 or 3.11)
- **pip** (Python package manager)
- **CUDA 11.8+** (optional, for GPU acceleration of Whisper model)
- **FFmpeg** (required for audio processing)
- **OpenAI API Key** (for resume analysis and interview questions)
- **Microphone** (for voice interview feature)

### Install FFmpeg

**Windows (via Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (via Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

---

## 📦 Installation & Setup

### 1. Clone or Navigate to Project

```bash
cd /home/ubuntu/Desktop/Resume-U
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Faster-Whisper Model

The Faster-Whisper model is automatically downloaded on first use, but you can pre-download it:

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

**Model Options:**
- `tiny` (~39MB) - Fastest, lowest accuracy
- `base` (~140MB) - **Recommended** - Good balance
- `small` (~440MB) - Better accuracy
- `medium` (~1.4GB) - High accuracy
- `large` (~3.1GB) - Best accuracy, slowest

**Model Download Location:**
- Linux/macOS: `~/.cache/huggingface/hub/models--openai--whisper-*/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\models--openai--whisper-*/`

---

## 🔐 Configuration

### 1. Set OpenAI API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
```

Or set as environment variable:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"  # Linux/macOS
set OPENAI_API_KEY=sk-your-api-key-here       # Windows
```

### 2. Configure Streamlit Secrets (Optional)

Create `.streamlit/secrets.toml`:

```toml
openai_api_key = "sk-your-api-key-here"
whisper_model = "base"
enable_gpu = true
```

### 3. Voice Settings Configuration

Edit `voice/voice_interview_manager.py` to customize:

```python
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_DURATION = 30        # seconds per question
WHISPER_MODEL = "base"     # faster-whisper model
USE_GPU = True             # Enable CUDA if available
```

---

## 🎤 Voice Features & Faster-Whisper

### What is Faster-Whisper?

**Faster-Whisper** is an optimized implementation of OpenAI's Whisper speech recognition model:

- **4x-5x faster** than original Whisper
- **Lower memory footprint**
- **Optimized for CPU and GPU** (CUDA support)
- **Batch processing** capability
- **Streaming audio** support

**Reference**: https://github.com/guillaumekln/faster-whisper

### Voice Processing Pipeline

```
Audio Input (Microphone)
    ↓
Audio Preprocessing (noise reduction, normalization)
    ↓
Faster-Whisper Speech Recognition (audio → text)
    ↓
OpenAI GPT (evaluate response)
    ↓
Text-to-Speech Output (feedback to candidate)
```

### Key Voice Components

#### **1. Speech Recognizer (`voice/speech_recognizer.py`)**

Converts audio to text using Faster-Whisper:

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda")  # GPU
segments, info = model.transcribe("audio.wav")
transcript = " ".join([s.text for s in segments])
```

**Features:**
- Automatic language detection
- Word-level timestamps
- Confidence scores per segment
- VAD (Voice Activity Detection) to skip silence

#### **2. Audio Processor (`voice/audio_processor.py`)**

Handles recording and audio file management:

```python
# Recording settings
- Sample Rate: 16kHz (optimal for Whisper)
- Channels: Mono (1 channel)
- Duration: Configurable per question
- Format: WAV (lossless)
```

**Process:**
1. Record audio from microphone
2. Apply noise gate (remove silence)
3. Normalize audio levels
4. Save as temporary WAV file
5. Pass to Faster-Whisper for transcription

#### **3. Text-to-Speech (`voice/text_to_speech.py`)**

Generates AI voice responses:

```python
# Options:
- OpenAI TTS (natural, multiple voices)
- Google TTS (free alternative)
- gTTS (offline capable)
```

### Voice Interview Flow

```
1. Display Question
   ↓
2. Record Candidate Answer (30s limit)
   ↓
3. Transcribe with Faster-Whisper
   ↓
4. Evaluate Answer (GPT-4)
   ↓
5. Generate Feedback
   ↓
6. Convert Feedback to Speech (TTS)
   ↓
7. Play Audio Feedback
   ↓
8. Store Results & Scoring
```

### GPU Acceleration Setup

**Enable CUDA for Faster-Whisper:**

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**In code:**

```python
from faster_whisper import WhisperModel

# Automatic GPU detection
model = WhisperModel("base", device="auto", compute_type="float16")
# device="cpu" for CPU-only
# compute_type="int8" for lower memory usage
```

---

## 🚀 Usage Guide

### Starting the Application

```bash
streamlit run final_app.py
```

The app opens at `http://localhost:8501`

### Step-by-Step Usage

#### **Step 1: Paste Job Description**
- Copy the job posting text
- Paste into the "Job Description" text area
- Min. 50 characters recommended

#### **Step 2: Upload Resume**
- Click "Upload Your Resume"
- Select PDF, DOCX, or TXT file
- File size: max. 50MB

#### **Step 3: Analyze Resume**
- Click "🚀 Analyze Resume" button
- Wait for AI analysis (10-30s)
- View candidate overview and match score

#### **Step 4: Review Eligibility**
- Check eligibility status (Pass/Fail)
- View reasons for decision
- Proceed to voice interview if eligible

#### **Step 5: Voice Interview**
- Grant microphone permission
- Listen to AI questions
- Click "Record Answer" for each question
- Speak your response (30s limit)
- Wait for transcription and evaluation
- Continue to next question

#### **Step 6: Review Results**
- View detailed scoring
- Check AI feedback per question
- Download evaluation report (optional)

---

## 🔄 Application Workflow

### **Workflow Diagram**

```
START
  ↓
[User Input: Job Description + Resume]
  ↓
[extract_text_from_file()]
  → Resume text extraction
  ↓
[orchestrator.analyze()]
  → Resume parsing (name, experience, education)
  → Job description parsing
  → Skills matching (GPT-4)
  → Match score calculation
  ↓
[eligibility_manager.get_eligibility_status()]
  → Threshold-based screening
  → Generate eligibility decision
  ↓
[Display Eligibility Result]
  ↓
IF Eligible:
  ↓
  [voice_interview_manager.start_voice_interview()]
    → Load interview questions
    → Initialize voice components
    ↓
  [display_voice_interview_progress()]
    → Record audio
    → Transcribe (Faster-Whisper)
    → Evaluate (GPT-4)
    → TTS response
    → Next question loop
  ↓
[display_interview_results()]
  → Final scores
  → Feedback summary
  ↓
ELSE Not Eligible:
  ↓
[display_detailed_analysis()]
  → Show gaps
  → Recommendations
  ↓
END
```

---

## 📊 Key Modules Explained

### **modules/agent/orchestrator.py**
- **Purpose**: Main AI coordination
- **Key Method**: `analyze(job_description, resume_text)`
- **Returns**: Dictionary with candidate info, skills, match score

### **modules/eligibility_manager.py**
- **Purpose**: Screening and decision logic
- **Key Method**: `get_eligibility_status(result)`
- **Threshold**: Configurable match percentage (default: 60%)

### **modules/enhanced_evaluation.py**
- **Purpose**: Answer evaluation and grading
- **Key Method**: `evaluate_with_reference_answers()`
- **Returns**: Score (0-100) and feedback per answer

### **voice/voice_interview_manager.py**
- **Purpose**: Interview orchestration and UI
- **Key Method**: `start_voice_interview(result)`
- **Manages**: Question flow, audio recording, result storage

### **voice/speech_recognizer.py**
- **Purpose**: Faster-Whisper integration
- **Key Method**: `transcribe_audio(audio_file)`
- **Features**: Language detection, confidence scores

---

## 🐛 Troubleshooting

### **Issue: "No module named 'faster_whisper'"**

```bash
pip install faster-whisper
```

### **Issue: Whisper Model Download Fails**

```bash
# Manual download
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

# Check download location
ls ~/.cache/huggingface/hub/
```

### **Issue: Microphone Not Detected**

```bash
# Check audio devices (Linux)
arecord -L

# Install audio utilities
sudo apt-get install alsa-utils pulseaudio
```

### **Issue: Slow Transcription (CPU-only)**

**Install CUDA to enable GPU acceleration:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Issue: "No CUDA devices found"**

```bash
# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### **Issue: OpenAI API Errors**

- Verify API key is correct
- Check API quota and billing
- Ensure model name is valid (`gpt-4`, `gpt-3.5-turbo`)

### **Issue: Audio Quality Issues**

Adjust in `voice/audio_processor.py`:

```python
# Reduce background noise
NOISE_GATE_THRESHOLD = -40  # dB (lower = more sensitive)
GAIN = 2.0  # Amplification
```

---

## 📈 Performance Tips

1. **Use "base" or "small" Whisper model** for faster transcription
2. **Enable GPU** for 4-5x speed improvement
3. **Cache interview questions** to reduce API calls
4. **Use streaming Whisper** for real-time transcription
5. **Compress audio** before sending to APIs

---

## 🔐 Security Notes

- **Never commit** `.env` or API keys to version control
- **Use `.gitignore`**:
  ```
  .env
  .streamlit/secrets.toml
  venv/
  __pycache__/
  *.pyc
  ```
- **Rotate API keys** regularly
- **Use environment variables** for production

---

## 📝 Requirements.txt

```
streamlit>=1.28.0
openai>=1.3.0
faster-whisper>=0.10.0
pydub>=0.25.1
python-dotenv>=1.0.0
PyPDF2>=3.0.0
python-docx>=0.8.11
torch>=2.0.0
librosa>=0.10.0
sounddevice>=0.4.6
requests>=2.31.0
```

---

## 📧 Support & Contribution

For issues or contributions:
1. Check troubleshooting section
2. Review logs in `.streamlit/logs/`
3. Check OpenAI API documentation
4. Review Faster-Whisper GitHub: https://github.com/guillaumekln/faster-whisper

---

## 📄 License

[Your License Here]

---

## 🎯 Next Steps

- Customize interview questions in `voice/voice_interview_manager.py`
- Adjust eligibility thresholds in `modules/eligibility_manager.py`
- Fine-tune Whisper model selection based on accuracy needs
- Add database integration for candidate history
- Deploy to Streamlit Cloud or self-hosted server

---

**Last Updated**: 2024
**Version**: 1.0.0


