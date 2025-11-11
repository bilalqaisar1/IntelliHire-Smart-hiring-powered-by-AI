import streamlit as st
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import asyncio
import edge_tts
from pydub import AudioSegment
from faster_whisper import WhisperModel
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from typing import Dict, Any, List
from modules.file_utils import extract_text_from_file
from modules.agent.orchestrator import AgentOrchestrator
from modules.openai_utils import evaluate_answers
from modules.eligibility_manager import EligibilityManager
from modules.interview_manager import InterviewManager
from modules.enhanced_evaluation import generate_correct_answers, evaluate_with_reference_answers, display_detailed_evaluation_results

# -----------------------------
# VOICE CONFIGURATION
# -----------------------------
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION = 10
MODEL_NAME = "tiny"
OUTPUT_DIR = "recordings"
VOICE = "en-IN-NeerjaNeural"   # Indian accent female voice
TTS_OUTPUT_DIR = "tts_out"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module="edge_tts")

# -----------------------------
# INIT ASR (Faster-Whisper)
# -----------------------------
_model_lock = Lock()
_model_singleton = None
_preload_pool = ThreadPoolExecutor(max_workers=1)
_preload_future = None


def _get_asr_model():
    """Thread-safe singleton loader for Faster-Whisper model"""
    global _model_singleton
    if _model_singleton is None:
        with _model_lock:
            if _model_singleton is None:
                print("🔄 Loading Faster-Whisper model...")
                _model_singleton = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
                print("✅ Model loaded.")
    return _model_singleton


def preload_asr_model():
    """Preload ASR model in background"""
    global _preload_future
    if _preload_future is None:
        _preload_future = _preload_pool.submit(_get_asr_model)


def wait_for_asr_ready(timeout=None):
    """Wait until model is fully loaded"""
    preload_asr_model()
    try:
        _preload_future.result(timeout=timeout)
    except Exception:
        pass


# -----------------------------
# TTS FUNCTION WITH INDIAN ACCENT
# -----------------------------
def play_audio_file(filepath):
    """Play audio file using sounddevice (more stable than pydub.playback)"""
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(filepath)
        
        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Play audio
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until playback is finished
        
    except Exception as e:
        print(f"❌ Audio playback error: {e}")


async def speak_text(text):
    """Convert text to speech using Indian accent voice and play it"""
    print(f"\n🗣️ {text}")
    
    # Generate unique filename for this TTS output
    timestamp = int(time.time())
    output_mp3 = os.path.join(TTS_OUTPUT_DIR, f"q_{timestamp}.mp3")
    output_wav = os.path.join(TTS_OUTPUT_DIR, f"q_{timestamp}.wav")
    
    try:
        # Generate speech with Indian accent
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_mp3)
        
        # Convert MP3 → WAV for playback compatibility
        if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 1000:
            sound = AudioSegment.from_file(output_mp3, format="mp3")
            sound.export(output_wav, format="wav")
            
            # Play audio using sounddevice
            play_audio_file(output_wav)
            
            # Clean up temporary files
            try:
                os.remove(output_mp3)
                os.remove(output_wav)
            except:
                pass
        else:
            print("❌ Error: Generated audio file invalid or empty!")
            
    except Exception as e:
        print(f"❌ TTS Error: {e}")


def speak(text):
    """Synchronous wrapper for speak_text"""
    asyncio.run(speak_text(text))


# -----------------------------
# CALIBRATE NOISE FLOOR ONCE
# -----------------------------
def calibrate_noise_floor(calibration_duration=10.0, block_duration=0.5):
    """
    Capture a few seconds of background noise to estimate baseline RMS.
    """
    print("🎙️ Calibrating background noise... Please stay silent for a few seconds.")

    try:
        noise_samples = []
        blocks = int(calibration_duration / block_duration)
        for _ in range(blocks):
            noise_block = sd.rec(int(block_duration * SAMPLE_RATE),
                                 samplerate=SAMPLE_RATE,
                                 channels=1,
                                 dtype='float32')
            sd.wait()
            noise_samples.append(noise_block)

        noise_audio = np.concatenate(noise_samples, axis=0)
        noise_floor = np.sqrt(np.mean(noise_audio ** 2))
        dynamic_threshold = max(noise_floor * 1.5, SILENCE_THRESHOLD)

        print(f"🔊 Noise floor RMS: {noise_floor:.6f}")
        print(f"🔊 Using silence threshold: {dynamic_threshold:.6f}")

        return noise_floor, dynamic_threshold
        
    except Exception as e:
        print(f"❌ Calibration error: {e}")
        print("🔄 Using default thresholds...")
        return 0.01, SILENCE_THRESHOLD


# -----------------------------
# RECORD AUDIO FUNCTION
# -----------------------------
def record_audio(filename, noise_floor, silence_threshold,
                 silence_duration=5.0, block_duration=0.5, max_recording_duration=180):
    """
    Record voice until silence detected, using pre-calibrated noise settings.
    """
    print("🎙️ Recording... Speak now.")

    try:
        recording = []
        silence_counter = 0
        total_duration = 0

        while True:
            try:
                audio_block = sd.rec(int(block_duration * SAMPLE_RATE),
                                     samplerate=SAMPLE_RATE,
                                     channels=1,
                                     dtype='float32')
                sd.wait()
            except Exception as e:
                print(f"❌ Microphone error: {e}")
                return False

            rms = np.sqrt(np.mean(audio_block ** 2))
            adjusted_rms = max(0, rms - noise_floor)

            print(f"RMS: {rms:.5f}, Adjusted RMS: {adjusted_rms:.5f}")

            recording.append(audio_block)

            if adjusted_rms < silence_threshold:
                silence_counter += block_duration
            else:
                silence_counter = 0

            total_duration += block_duration

            if silence_counter >= silence_duration:
                print(f"🛑 Silence detected for {silence_duration} seconds. Ending recording.")
                break

            if total_duration >= max_recording_duration:
                print("⏰ Max recording time reached. Ending recording.")
                break

        if recording:
            audio = np.concatenate(recording, axis=0)
            sf.write(filename, audio, SAMPLE_RATE)
            print(f"✅ Audio saved: {filename}")
            return True
        else:
            print("❌ No audio recorded")
            return False
            
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False


# -----------------------------
# PARALLEL TRANSCRIPTION
# -----------------------------
def transcribe_audio_parallel(filename, question_idx):
    """Transcribe a single audio file - designed for parallel execution"""
    try:
        print(f"🧠 Transcribing Question {question_idx}: {filename}")
        model = _get_asr_model()
        segments, info = model.transcribe(
            filename,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join([segment.text.strip() for segment in segments])
        print(f"📝 Question {question_idx} transcribed: {text}")
        return question_idx, text
    except Exception as e:
        print(f"❌ Transcription error for Question {question_idx}: {e}")
        return question_idx, "[Transcription Failed]"


def transcribe_all_answers_parallel(recorded_files):
    """Transcribe all recorded files in parallel using threading"""
    
    # Create a thread pool for parallel transcription
    with ThreadPoolExecutor(max_workers=min(4, len(recorded_files))) as executor:
        # Submit all transcription tasks
        future_to_idx = {
            executor.submit(transcribe_audio_parallel, filename, idx): idx 
            for idx, filename in recorded_files.items()
        }
        
        # Collect results as they complete
        transcriptions = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                question_idx, transcription = future.result()
                transcriptions[question_idx] = transcription
            except Exception as e:
                print(f"❌ Error transcribing Question {idx}: {e}")
                transcriptions[idx] = "[Transcription Failed]"
    
    print("✅ All transcriptions completed!")
    return transcriptions


# -----------------------------
# VOICE INTERVIEW MANAGER
# -----------------------------
class VoiceInterviewManager(InterviewManager):
    """Enhanced Interview Manager with voice capabilities"""
    
    def __init__(self):
        super().__init__()
        self.noise_floor = None
        self.silence_threshold = None
        self.interview_started = False
    
    def start_voice_interview(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Start the voice interview process with background noise calibration.
        
        Args:
            analysis_result: The analysis result from orchestrator
            
        Returns:
            True if interview started successfully, False otherwise
        """
        try:
            with st.spinner("🧩 Generating interview questions..."):
                questions = self.orchestrator.prepare_interview(analysis_result)
            
            if not questions:
                st.warning("No interview questions could be generated. Please try with a different resume.")
                return False
            
            # Initialize voice interview session
            st.session_state["questions"] = questions
            st.session_state["answers"] = {}
            st.session_state["current_q"] = 0
            st.session_state["voice_interview_started"] = True
            st.session_state["interview_scores"] = {}
            st.session_state["recorded_files"] = {}
            st.session_state["interview_completed"] = False
            
            # Store matched skills for reference answer generation
            matched_skills = self.orchestrator.get_matched_skills_from_result(analysis_result)
            st.session_state["matched_skills"] = matched_skills
            
            # Preload ASR model
            preload_asr_model()
            
            # Calibrate background noise once at the start
            with st.spinner("🎙️ Calibrating background noise... Please stay silent for a few seconds."):
                self.noise_floor, self.silence_threshold = calibrate_noise_floor(calibration_duration=3.0)
                st.session_state["noise_floor"] = self.noise_floor
                st.session_state["silence_threshold"] = self.silence_threshold
            
            st.success("🎤 Microphone calibrated! Starting voice interview...")
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start voice interview: {e}")
            return False
    
    def display_voice_interview_progress(self, analysis_result: Dict[str, Any]):
        """Display voice interview progress and current question - optimized for minimal latency."""
        questions = st.session_state["questions"]
        current_q = st.session_state["current_q"]
        recorded_files = st.session_state.get("recorded_files", {})
        
        # Show compact header during interview
        st.markdown("---")
        st.header("🎤 Voice AI Interview in Progress")
        
        # Progress indicator
        progress = (current_q + 1) / len(questions)
        st.progress(progress)
        st.caption(f"Question {current_q + 1} of {len(questions)}")
        
        # Show current question
        st.markdown(f"### 🧠 Question {current_q + 1}")
        st.info(questions[current_q])
        
        # Check if this question has been processed
        if current_q not in recorded_files and current_q not in st.session_state.get("skipped_questions", set()):
            # Auto-start voice interview for this question
            if not st.session_state.get(f"question_{current_q}_started", False):
                st.session_state[f"question_{current_q}_started"] = True
                
                # Get calibrated noise settings
                noise_floor = st.session_state.get("noise_floor", self.noise_floor)
                silence_threshold = st.session_state.get("silence_threshold", self.silence_threshold)
                
                # Auto-speak the question
                with st.spinner("🔊 Playing question..."):
                    speak(questions[current_q])
                
                # Auto-start recording (no transcription yet)
                filename = os.path.join(OUTPUT_DIR, f"answer_{current_q}.wav")
                with st.spinner("🎙️ Recording your answer... Speak now!"):
                    success = record_audio(filename, noise_floor, silence_threshold, silence_duration=5)
                
                if success:
                    recorded_files[current_q] = filename
                    st.session_state["recorded_files"] = recorded_files
                    st.success(f"✅ Answer {current_q + 1} recorded!")
                    
                    # Auto-move to next question or finish
                    if current_q + 1 < len(questions):
                        st.session_state["current_q"] += 1
                        st.rerun()
                    else:
                        # All questions recorded, now transcribe in parallel
                        self.complete_voice_interview_with_parallel_transcription(questions, recorded_files)
                else:
                    st.error("❌ Recording failed. Please try again.")
                    # Reset the question state to allow retry
                    del st.session_state[f"question_{current_q}_started"]
                    st.rerun()
        
        # Show skip option if question hasn't been processed yet
        if current_q not in recorded_files and current_q not in st.session_state.get("skipped_questions", set()):
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⏭️ Skip This Question", use_container_width=True, key=f"skip_{current_q}"):
                    # Mark question as skipped
                    if "skipped_questions" not in st.session_state:
                        st.session_state["skipped_questions"] = set()
                    st.session_state["skipped_questions"].add(current_q)
                    
                    st.warning(f"⚠️ Question {current_q + 1} skipped")
                    
                    # Auto-move to next question or finish
                    if current_q + 1 < len(questions):
                        st.session_state["current_q"] += 1
                        st.rerun()
                    else:
                        # All questions processed, now transcribe in parallel
                        self.complete_voice_interview_with_parallel_transcription(questions, recorded_files)
            
            with col2:
                if st.button("🔄 Retry Recording", use_container_width=True, key=f"retry_{current_q}"):
                    # Reset the question state to allow retry
                    if f"question_{current_q}_started" in st.session_state:
                        del st.session_state[f"question_{current_q}_started"]
                    st.rerun()
        
        # Show recording status
        st.markdown("### 📹 Recording Status:")
        skipped_questions = st.session_state.get("skipped_questions", set())
        for i, question in enumerate(questions):
            if i in recorded_files:
                st.success(f"✅ Question {i + 1}: Recorded")
            elif i in skipped_questions:
                st.warning(f"⏭️ Question {i + 1}: Skipped")
            elif i == current_q:
                st.info(f"🎙️ Question {i + 1}: Currently recording...")
            else:
                st.info(f"⏳ Question {i + 1}: Pending")
        
        # Cancel interview option
        st.markdown("---")
        if st.button("❌ Cancel Interview", use_container_width=True, key="cancel_interview"):
            if st.session_state.get("confirm_cancel"):
                self.cancel_interview()
            else:
                st.session_state["confirm_cancel"] = True
                st.rerun()
        
        # Show cancel confirmation
        if st.session_state.get("confirm_cancel"):
            st.warning("⚠️ Are you sure you want to cancel the interview? All progress will be lost.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, Cancel", use_container_width=True, key="confirm_cancel_yes"):
                    self.cancel_interview()
            with col2:
                if st.button("❌ No, Continue", use_container_width=True, key="confirm_cancel_no"):
                    del st.session_state["confirm_cancel"]
                    st.rerun()
    
    def complete_voice_interview_with_parallel_transcription(self, questions: List[str], recorded_files: Dict[int, str]):
        """Complete the voice interview with parallel transcription - optimized for minimal latency."""
        try:
            # First, speak completion message
            with st.spinner("🎤 Interview completed! Thank you so much for your time..."):
                completion_message = "Your interview is completed. Thank you so much for your time. We really appreciate your participation."
                speak(completion_message)
            
            # Get skipped questions
            skipped_questions = st.session_state.get("skipped_questions", set())
            
            # First, transcribe all answers in parallel
            with st.spinner("🧠 Transcribing all answers in parallel..."):
                transcriptions = transcribe_all_answers_parallel(recorded_files)
            
            # Convert transcriptions to answers format, including skipped questions
            answers = {}
            for idx, question in enumerate(questions):
                if idx in skipped_questions:
                    answers[idx] = "[Question Skipped]"  # Mark skipped questions
                else:
                    answers[idx] = transcriptions.get(idx, "[Transcription Failed]")
            
            # Now proceed with evaluation
            with st.spinner("🧾 Evaluating your answers..."):
                # Get matched skills for generating reference answers
                matched_skills = st.session_state.get("matched_skills", [])
                
                # Generate reference answers
                with st.spinner("📚 Generating reference answers..."):
                    correct_answers = generate_correct_answers(questions, matched_skills)
                
                # Evaluate with reference answers
                evaluation_result = evaluate_with_reference_answers(questions, answers, correct_answers)
                
                # Store results
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = evaluation_result.get("overall_score", 0.0)
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.session_state["evaluation_result"] = evaluation_result
                st.session_state["correct_answers"] = correct_answers
                st.session_state["skipped_questions"] = skipped_questions
                
                # Clear interview state
                if "voice_interview_started" in st.session_state:
                    del st.session_state["voice_interview_started"]
                if "current_q" in st.session_state:
                    del st.session_state["current_q"]
                if "confirm_cancel" in st.session_state:
                    del st.session_state["confirm_cancel"]
                if "recorded_files" in st.session_state:
                    del st.session_state["recorded_files"]
                
                st.rerun()
                
        except Exception as e:
            st.error(f"Failed to complete interview: {e}")
            # Fallback to original scoring
            try:
                interview_score = self.orchestrator.score_interview(questions, answers)
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = interview_score
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.rerun()
            except Exception as e2:
                st.error(f"Fallback scoring also failed: {e2}")
    
    def complete_voice_interview(self, questions: List[str], answers: Dict[int, str]):
        """Complete the voice interview and calculate scores using enhanced evaluation."""
        try:
            with st.spinner("🧾 Evaluating your answers..."):
                # Get matched skills for generating reference answers
                matched_skills = st.session_state.get("matched_skills", [])
                
                # Generate reference answers
                with st.spinner("📚 Generating reference answers..."):
                    correct_answers = generate_correct_answers(questions, matched_skills)
                
                # Evaluate with reference answers
                evaluation_result = evaluate_with_reference_answers(questions, answers, correct_answers)
                
                # Store results
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = evaluation_result.get("overall_score", 0.0)
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.session_state["evaluation_result"] = evaluation_result
                st.session_state["correct_answers"] = correct_answers
                
                # Clear interview state
                if "voice_interview_started" in st.session_state:
                    del st.session_state["voice_interview_started"]
                if "current_q" in st.session_state:
                    del st.session_state["current_q"]
                if "confirm_cancel" in st.session_state:
                    del st.session_state["confirm_cancel"]
                if "recorded_files" in st.session_state:
                    del st.session_state["recorded_files"]
                
                st.rerun()
                
        except Exception as e:
            st.error(f"Failed to complete interview: {e}")
            # Fallback to original scoring
            try:
                interview_score = self.orchestrator.score_interview(questions, answers)
                st.session_state["interview_completed"] = True
                st.session_state["interview_score"] = interview_score
                st.session_state["interview_questions"] = questions
                st.session_state["interview_answers"] = answers
                st.rerun()
            except Exception as e2:
                st.error(f"Fallback scoring also failed: {e2}")
    
    def cancel_interview(self):
        """Cancel the interview and return to eligibility screen."""
        # Clear interview state
        keys_to_clear = [
            'voice_interview_started', 'questions', 'answers', 'current_q',
            'confirm_cancel', 'interview_scores', 'recorded_files',
            'noise_floor', 'silence_threshold', 'interview_completed',
            'skipped_questions'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear question state variables
        for key in list(st.session_state.keys()):
            if key.startswith('question_') and key.endswith('_started'):
                del st.session_state[key]
        
        st.rerun()


# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Resume ↔ AI Voice Interview",
    page_icon="🎤",
    layout="wide"
)

# Initialize managers
orchestrator = AgentOrchestrator()
eligibility_manager = EligibilityManager()
voice_interview_manager = VoiceInterviewManager()

# -------------------------------
# Header & Sidebar
# -------------------------------
st.title("🎤 Resume ↔ AI Voice Interview")
st.caption("AI-powered Resume Screening + Voice Interview Simulation")

with st.sidebar:
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Paste the **job description** below.  
    2. Upload your **resume (PDF/DOCX/TXT)**.  
    3. Click **Analyze Resume** to get candidate overview.  
    4. Review match score and eligibility assessment.  
    5. If eligible (≥60%), start your **AI Voice Interview**.
    6. Complete voice interview and get your final score.
    """)
    
    st.markdown("### 🎯 Eligibility Criteria")
    st.markdown(f"""
    - **Minimum Match Score:** {eligibility_manager.threshold}%
    - **Interview Required:** For eligible candidates only
    - **Final Score:** Combined resume + interview performance
    """)

# -------------------------------
# Input Section
# -------------------------------
st.markdown("### 💼 Paste Job Description")
job_post = st.text_area(
    "Paste job description here...",
    height=250,
    label_visibility="collapsed",
    placeholder="Paste the full job posting text here...",
    key="job_description_text_area"
)

st.markdown("### 📂 Upload Your Resume")
uploaded_resume = st.file_uploader(
    "Choose file",
    type=["pdf", "docx", "doc", "txt"],
    label_visibility="collapsed",
    key="resume_file_uploader"
)

analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True, key="analyze_resume_button")

# -------------------------------
# Step 1: Resume & Job Matching Logic
# -------------------------------
if analyze_btn:
    if not job_post.strip():
        st.error("❌ Please paste the job posting text.")
    elif not uploaded_resume:
        st.error("❌ Please upload a resume file first.")
    else:
        try:
            with st.spinner("📄 Extracting text from resume..."):
                resume_text = extract_text_from_file(uploaded_resume)

            if not resume_text.strip():
                st.error("⚠️ Could not extract text from this resume. Try another format.")
            else:
                with st.spinner("🧮 Computing AI-based match score..."):
                    result = orchestrator.analyze(job_post, resume_text)

                # Store result in session for persistence
                st.session_state["analysis_result"] = result
                
                # Get eligibility status
                eligibility_status = eligibility_manager.get_eligibility_status(result)
                st.session_state["eligibility_status"] = eligibility_status

                st.success("🎯 Resume analysis complete!")
                
                # Display Candidate Overview first
                #st.markdown("---")
                #st.header("👤 Candidate Overview")
                
                # Candidate basic information
                #col1, col2, col3 = st.columns(3)
                #with col1:
                #    st.metric("Name", result.get('candidate_name') or 'N/A')
                #with col2:
                #    st.metric("Current Title", result.get('candidate_current_title') or 'N/A')
                #with col3:
                #    st.metric("Job Applied For", result.get('job_title') or 'N/A')
                
                # Education section
                #st.subheader("🎓 Education")
                edu = result.get('education') or []
                if edu:
                    for e in edu:
                        degree = e.get('degree') or 'Degree'
                        inst = e.get('institution') or 'Institution'
                        sd = e.get('start_date') or ''
                        ed = e.get('end_date') or ''
                        dates = f" ({sd} – {ed})" if (sd or ed) else ""
                #        st.markdown(f"• {degree}, {inst}{dates}")
                else:
                    pass
                #    st.info("No education information found")
                
                # Experience section
                #st.subheader("💼 Work Experience")
                exps = result.get('experiences') or []
                if exps:
                    for exp in exps:
                        pos = exp.get('position') or 'Role'
                        comp = exp.get('company') or 'Company'
                        sd = exp.get('start_date') or ''
                        ed = exp.get('end_date') or ''
                        dates = f" ({sd} – {ed})" if (sd or ed) else ""
                #        st.markdown(f"• {pos} at {comp}{dates}")
                else:
                    pass
                #    st.info("No work experience found")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -------------------------------
# Step 2: Eligibility Assessment & Voice Interview Flow
# -------------------------------
if "analysis_result" in st.session_state and "voice_interview_started" not in st.session_state and "interview_completed" not in st.session_state:
    result = st.session_state["analysis_result"]
    eligibility_status = st.session_state.get("eligibility_status")
    
    if eligibility_status:
        # Display eligibility result
        should_start_interview = eligibility_manager.display_eligibility_result(eligibility_status)
        
        if should_start_interview:
            # Start the voice interview
            if voice_interview_manager.start_voice_interview(result):
                st.rerun()
    else:
        # Fallback: create eligibility status if missing
        eligibility_status = eligibility_manager.get_eligibility_status(result)
        st.session_state["eligibility_status"] = eligibility_status
        st.rerun()

# -------------------------------
# Step 3: Conduct Voice Interview (Interactive)
# -------------------------------
if st.session_state.get("voice_interview_started"):
    result = st.session_state["analysis_result"]
    voice_interview_manager.display_voice_interview_progress(result)

# -------------------------------
# Step 4: Display Interview Results
# -------------------------------
if st.session_state.get("interview_completed"):
    result = st.session_state["analysis_result"]
    voice_interview_manager.display_interview_results(result)

# -------------------------------
# Step 5: Detailed Analysis (for ineligible candidates)
# -------------------------------
if st.session_state.get("show_detailed_analysis"):
    result = st.session_state["analysis_result"]
    eligibility_manager.display_detailed_analysis(result)
