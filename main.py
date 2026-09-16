import os
import time
import shutil
import json
import numpy as np
import tempfile
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Agno and LLM client imports
from agno.agent import Agent
from agno.models.groq import Groq as AgnoGroq
from groq import Groq

# -----------------------------
# Setup Environment
# -----------------------------
load_dotenv()

# Set default active provider if not specified
if not os.getenv("ACTIVE_PROVIDER"):
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        os.environ["ACTIVE_PROVIDER"] = "gemini"
    elif os.getenv("OPENAI_API_KEY"):
        os.environ["ACTIVE_PROVIDER"] = "openai"
    else:
        os.environ["ACTIVE_PROVIDER"] = "groq"

KNOWLEDGE_BASE_FILE = "knowledge_base.txt"

# If knowledge base does not exist, create a sample one
if not os.path.exists(KNOWLEDGE_BASE_FILE):
    default_kb = (
        "The company name is Agno AI.\n"
        "The developer is Ayub Alam, who is a Machine Learning Engineer.\n"
        "The project is a Real-Time Audio Validation Agent.\n"
        "The system is built using Python, Agno, and Groq Llama 3.3.\n"
        "The capital of France is Paris.\n"
    )
    with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
        f.write(default_kb)
    print(f"Created default '{KNOWLEDGE_BASE_FILE}'. You can customize it as needed.")

# Load Knowledge Base cache
with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
    knowledge_base = f.read()

# Define Validation Output Schema
class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the transcript facts are correct according to the knowledge base, False otherwise.")
    reason: str = Field(description="A brief, one-sentence explanation of why the transcript was validated or why it failed based on the knowledge base.")
    confidence_score: int = Field(description="An integer between 0 and 100 representing the validation confidence level (0 means completely contradicted or unsure, 100 means a perfect certain match).")
    has_factual_claim: bool = Field(description="True if the statement contains a factual claim (e.g. names, dates, capitals, scientific facts) that can be verified. False for greetings, commands, opinions, general conversation, or meta-talk.")

def update_env_vars(updates: dict):
    existing = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        existing[parts[0].strip()] = parts[1].strip().strip('"').strip("'")
            
    for k, v in updates.items():
        existing[k] = v
        
    with open(".env", "w", encoding="utf-8") as f:
        f.write("# Environment Configuration\n")
        for k, v in existing.items():
            f.write(f'{k}="{v}"\n')
            
    load_dotenv(override=True)

# -----------------------------
# Agent Core & API Logic Helpers
# -----------------------------
def get_kb_text() -> str:
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return ""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return f.read()

def update_kb_text(new_text: str):
    global knowledge_base
    with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
        f.write(new_text)
    knowledge_base = new_text

def get_api_keys() -> dict:
    return {
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
        "active_provider": os.getenv("ACTIVE_PROVIDER", "gemini")
    }

def update_api_keys(payload: dict):
    env_updates = {"ACTIVE_PROVIDER": payload["active_provider"]}
    if payload.get("gemini_key") and payload["gemini_key"].strip():
        env_updates["GEMINI_API_KEY"] = payload["gemini_key"].strip()
        env_updates["GOOGLE_API_KEY"] = payload["gemini_key"].strip()
    if payload.get("groq_key") and payload["groq_key"].strip():
        env_updates["GROQ_API_KEY"] = payload["groq_key"].strip()
    if payload.get("openai_key") and payload["openai_key"].strip():
        env_updates["OPENAI_API_KEY"] = payload["openai_key"].strip()
    if payload.get("tavily_key") and payload["tavily_key"].strip():
        env_updates["TAVILY_API_KEY"] = payload["tavily_key"].strip()
    update_env_vars(env_updates)

def validate_via_web_search(transcript: str, provider: str) -> ValidationResult:
    import wikipedia
    # Set custom User-Agent to avoid Wikipedia API rate limits / JSONDecodeError
    wikipedia.set_user_agent("AudioValidator/1.0 (contact@audiovalidator.com)")
    from agno.tools.wikipedia import WikipediaTools
    
    if provider == "gemini":
        from agno.models.google import Gemini as AgnoGemini
        validation_model = AgnoGemini(
            id="gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "openai":
        from agno.models.openai import OpenAIChat
        validation_model = OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_completion_tokens=100
        )
    else:
        validation_model = AgnoGroq(
            id="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
    wiki_tools = WikipediaTools()
    
    # Custom tool function to simplify parameter schema and prevent Groq tool validation errors
    def search_wikipedia(query: str) -> str:
        """Use this function to search Wikipedia for a given query.
        This function uses Wikipedia to provide factual, encyclopedia information about the query.

        Args:
            query (str): Query to search for on Wikipedia.

        Returns:
            str: JSON string of results related to the query.
        """
        return wiki_tools.search_wikipedia(query=query)
        
    search_agent = Agent(
        name="Wikipedia Search Validation Agent",
        model=validation_model,
        tools=[search_wikipedia],
        description="Searches Wikipedia to validate the accuracy of a statement.",
        instructions=[
            "Search Wikipedia to verify the facts in the provided transcript statement, including historical facts, scientific concepts, geography, public figures, and general knowledge.",
            "Compare the transcript claims against Wikipedia search results to determine correctness.",
            "REAL-TIME INTERRUPTION HANDLING (LIVE CALL PRIORITY #1):",
            "- Wait until the user says at least 3 words OR stops speaking for 0.5s.",
            "- If the user interrupts you → stop generating audio immediately. Pause 0.3–0.5s, then listen for next query.",
            "- Ignore background noise; focus on user voice.",
            "Work smartly, intelligently, fast, and in real-time.",
            "Format your output EXACTLY as follows:",
            "VALID: <True or False>",
            "CONFIDENCE: <An integer between 0 and 100 indicating the confidence score of matching or contradicting facts>",
            "REASON: <Your detailed explanation of why the statement is true or false based on the search findings.>"
        ],
    )
    
    response = search_agent.run(
        f"Validate the following statement using Wikipedia search: '{transcript}'"
    )
    
    text = response.content if isinstance(response.content, str) else str(response.content)
    
    is_valid = False
    confidence_score = 80
    reason = text
    
    for line in text.splitlines():
        line_strip = line.strip()
        if line_strip.upper().startswith("VALID:"):
            val_str = line_strip.split(":", 1)[1].strip().lower()
            is_valid = "true" in val_str
        elif line_strip.upper().startswith("CONFIDENCE:"):
            try:
                confidence_score = int(line_strip.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line_strip.upper().startswith("REASON:"):
            reason = line_strip.split(":", 1)[1].strip()
            
    return ValidationResult(is_valid=is_valid, reason=reason, confidence_score=confidence_score, has_factual_claim=True)

def get_history_logs() -> list:
    history_file = os.path.join("history", "logs.json")
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def get_audio_filepath(filename: str) -> str:
    safe_filename = os.path.basename(filename)
    audio_path = os.path.join("history", safe_filename)
    if os.path.exists(audio_path):
        return audio_path
    return ""

def should_validate_transcript(transcript: str) -> bool:
    """
    Filters out background noise, empty/punctuation-only segments, and queries 
    that contain 3 or fewer words. Returns True only if the transcript contains
    more than 3 words (i.e. >= 4 words) and is not noise.
    """
    import re
    if not transcript:
        return False
        
    clean_text = transcript.strip()
    
    # 1. Skip standard punctuation and empty/blank transcripts
    if not clean_text or clean_text in [".", "...", "?", "!", ",", ".\n"]:
        return False
        
    # 2. Filter out background noise markers commonly generated by Whisper or Gemini
    # Examples: [music], (laughter), [coughing], [background noise], etc.
    filtered_text = re.sub(r'\[.*?\]|\(.*?\)', '', clean_text).strip()
    
    # Common text phrases representing noise if they are the entire transcript
    noise_phrases = {
        "music", "laughter", "coughing", "silence", "snicker", 
        "background noise", "cough", "clears throat", "sigh", "whispering"
    }
    if filtered_text.lower() in noise_phrases:
        return False
        
    # 3. Check word count: only validate if it contains more than 3 words (i.e., >= 4 words)
    words = filtered_text.split()
    if len(words) <= 3:
        return False
        
    # 4. Filter out common conversational fillers, commands, questions, and meta-talk
    # Convert to lowercase and strip punctuation for match checks
    norm_text = re.sub(r"[^\w\s\']", "", filtered_text.lower()).strip()
    
    # Common phrases that contain no validateable factual claims (conversational patterns)
    conversational_patterns = [
        r"^i'm going to",
        r"^i am going to",
        r"^let's go",
        r"^let us go",
        r"^go to the next",
        r"^let me check",
        r"^let me see",
        r"^let's see",
        r"^let's start",
        r"^let's stop",
        r"^can you hear",
        r"^is it working",
        r"^is this working",
        r"^hello hello",
        r"^test test",
        r"^testing testing",
        r"^one two three",
        r"^1 2 3",
        r"^thank you",
        r"^thanks for",
        r"^no problem",
        r"^you're welcome",
        r"^you are welcome",
        r"^i will go",
        r"^we will go",
        r"^hang on a",
        r"^wait a minute",
        r"^just a moment",
        r"^okay let's",
        r"^ok let's",
        r"^so basically",
        r"^what i mean",
        r"^i think that",
        r"^i'm not sure",
        r"^i don't know",
        r"^i do not know",
        r"^let's talk about",
        r"^i want to",
        r"^i need to",
        r"^can you show",
        r"^can you tell",
        r"^please show",
        r"^please tell",
    ]
    
    for pattern in conversational_patterns:
        if re.search(pattern, norm_text):
            return False
            
    # Also skip if it's very short conversational acknowledgements or phrases
    filler_phrases = {
        "how are you", "how's it going", "what's up", "good morning", "good afternoon",
        "good evening", "have a good day", "have a nice day", "see you later", "catch you later"
    }
    if norm_text in filler_phrases:
        return False
        
    return True

async def process_and_validate_uploaded_audio(audio) -> dict:
    os.makedirs("history", exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    ext = os.path.splitext(audio.filename)[1] or ".webm"
    filename = f"audio_{timestamp}{ext}"
    audio_path = os.path.join("history", filename)
    
    # Save uploaded file
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    provider = os.getenv("ACTIVE_PROVIDER", "gemini").lower()
    transcript = ""
    
    try:
        if provider == "gemini":
            from google import genai
            from google.genai import types
            
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                raise ValueError("Gemini API Key is not set. Please configure it in settings.")
                
            client = genai.Client(api_key=gemini_key)
            
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                
            mime_type = "audio/webm"
            if ext == ".ogg":
                mime_type = "audio/ogg"
            elif ext == ".wav":
                mime_type = "audio/wav"
                
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    "Transcribe this audio file exactly as spoken. If there is only silence, noise, or music, return an empty string. Output only the transcript text without any introductory remarks."
                ]
            )
            transcript = response.text.strip() if response.text else ""
        elif provider == "openai":
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OpenAI API Key is not set. Please configure it in settings.")
                
            from openai import OpenAI
            openai_client = OpenAI(api_key=openai_key)
            with open(audio_path, "rb") as audio_file:
                transcription = openai_client.audio.transcriptions.create(
                    file=(audio_path, audio_file),
                    model="whisper-1",
                    response_format="text"
                )
            transcript = transcription.strip()
        else:
            # Groq Provider
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                raise ValueError("Groq API Key is not set. Please configure it in settings.")
                
            groq_client = Groq(api_key=groq_key)
            with open(audio_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_path, audio_file),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
            transcript = transcription.strip()
    except Exception as e:
        print(f"Transcription failed: {e}")
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
        raise Exception(f"Transcription failed: {str(e)}")
        
    # Skip noise, empty, or short transcripts (less than or equal to 3 words)
    if not should_validate_transcript(transcript):
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
        return {"status": "skipped", "transcript": transcript}
        
    # Reload KB
    kb_content = get_kb_text()
            
    is_valid = False
    reason = ""
    
    try:
        if provider == "gemini":
            from agno.models.google import Gemini as AgnoGemini
            validation_model = AgnoGemini(
                id="gemini-2.5-flash",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        elif provider == "openai":
            from agno.models.openai import OpenAIChat
            validation_model = OpenAIChat(
                id="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_completion_tokens=100
            )
        else:
            validation_model = AgnoGroq(
                id="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY")
            )
            
        validation_agent = Agent(
            name="Knowledge Validation Agent",
            model=validation_model,
            description="Validates transcript against a knowledge base.",
            instructions=[
                "Use ONLY the provided knowledge base.",
                "Compare the transcript with the knowledge base.",
                "Identify whether the transcript information exists in the knowledge base.",
                "Set is_valid to True if the core information/claim in the transcript is correct and exists in the knowledge base.",
                "Set is_valid to False if information is incorrect, contradicts the knowledge base, is missing, or if the transcript is empty or irrelevant.",
                "Evaluate a confidence_score between 0 and 100. Assign a high score (80-100) if you are very certain of the match or contradiction, moderate values (40-79) for ambiguous or partially matching claims, and low values (0-39) if completely unrelated, missing context, or highly uncertain.",
                "CRITICAL: Regardless of whether the transcript information is in the knowledge base or is completely unrelated to it, if the statement asserts any checkable claim about real-world events, news, organizations, people, countries, or facts, you MUST set has_factual_claim to True. Set it to False ONLY for greetings, commands, personal opinions about the self/session, conversational filler, noise, or meta-talk.",
                "REAL-TIME INTERRUPTION HANDLING (LIVE CALL PRIORITY #1):",
                "- Wait until the user says at least 3 words OR stops speaking for 0.5s.",
                "- If the user interrupts you → stop generating audio immediately. Pause 0.3–0.5s, then listen for next query.",
                "- Ignore background noise; focus on user voice.",
                "Work smartly, intelligently, fast, and in real-time."
            ],
            output_schema=ValidationResult
        )
        
        validation_response = validation_agent.run(
            f"""
            KNOWLEDGE BASE:
            {kb_content}
 
            TRANSCRIPT:
            {transcript}
 
            Validate the transcript against the knowledge base.
            """
        )
        
        validation_result = validation_response.content
        confidence_score = 0
        has_factual_claim = False
        
        if isinstance(validation_result, ValidationResult):
            is_valid = validation_result.is_valid
            reason = validation_result.reason
            confidence_score = validation_result.confidence_score
            has_factual_claim = validation_result.has_factual_claim
        elif hasattr(validation_result, 'is_valid'):
            is_valid = validation_result.is_valid
            reason = getattr(validation_result, 'reason', '')
            confidence_score = getattr(validation_result, 'confidence_score', 0)
            has_factual_claim = getattr(validation_result, 'has_factual_claim', False)
        else:
            try:
                if isinstance(validation_result, dict):
                    is_valid = validation_result.get("is_valid", False)
                    reason = validation_result.get("reason", "")
                    confidence_score = validation_result.get("confidence_score", 0)
                    has_factual_claim = validation_result.get("has_factual_claim", False)
                else:
                    res_dict = json.loads(str(validation_result))
                    is_valid = res_dict.get("is_valid", False)
                    reason = res_dict.get("reason", "")
                    confidence_score = res_dict.get("confidence_score", 0)
                    has_factual_claim = res_dict.get("has_factual_claim", False)
            except Exception:
                is_valid = False
                reason = str(validation_result)
                confidence_score = 0
                has_factual_claim = False
    except Exception as e:
        print(f"Validation failed: {e}")
        is_valid = False
        reason = f"Validation agent error: {str(e)}"
        has_factual_claim = False
        
    # If not valid on local KB, fallback to Wikipedia search validation ONLY if it contains a factual claim
    if not is_valid and has_factual_claim:
        try:
            print(f"Statement failed local KB validation but contains a factual claim. Invoking Wikipedia Search Validation Agent fallback for transcript: '{transcript}'")
            search_res = validate_via_web_search(transcript, provider)
            if search_res:
                is_valid = search_res.is_valid
                reason = f"Verified via Wikipedia: {search_res.reason}"
                confidence_score = search_res.confidence_score
        except Exception as search_err:
            print(f"Wikipedia Search fallback failed: {search_err}")
        
    # Update logs.json
    history_file = os.path.join("history", "logs.json")
    logs = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except:
            logs = []
            
    new_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": transcript,
        "is_valid": is_valid,
        "confidence_score": confidence_score,
        "reason": reason,
        "audio_url": f"/api/audio/{filename}"
    }
    
    logs.insert(0, new_log)
    
    # Prune logs to 50 items to save space
    if len(logs) > 50:
        oldest = logs.pop()
        old_filename = oldest.get("audio_url", "").split("/")[-1]
        if old_filename:
            old_path = os.path.join("history", old_filename)
            if os.path.exists(old_path):
                try: os.remove(old_path)
                except: pass
                
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
        
    return {
        "status": "success",
        "transcript": transcript,
        "is_valid": is_valid,
        "confidence_score": confidence_score,
        "reason": reason,
        "audio_url": f"/api/audio/{filename}"
    }

# -----------------------------
# CLI Console Loop Fallback
# -----------------------------
def run_cli_loop():
    import sounddevice as sd
    import soundfile as sf
    
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file to run CLI mode.")
        
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    validation_agent = Agent(
        name="Knowledge Validation Agent",
        model=AgnoGroq(id="llama-3.1-8b-instant"),
        description="Validates transcript against a knowledge base.",
        instructions=[
            "Use ONLY the provided knowledge base.",
            "Compare the transcript with the knowledge base.",
            "Identify whether the transcript information exists in the knowledge base.",
            "Set is_valid to True if the core information/claim in the transcript is correct and exists in the knowledge base.",
            "Set is_valid to False if information is incorrect, contradicts the knowledge base, is missing, or if the transcript is empty or irrelevant.",
            "Evaluate a confidence_score between 0 and 100. Assign a high score (80-100) if you are very certain of the match or contradiction, moderate values (40-79) for ambiguous or partially matching claims, and low values (0-39) if completely unrelated, missing context, or highly uncertain.",
            "CRITICAL: Regardless of whether the transcript information is in the knowledge base or is completely unrelated to it, if the statement asserts any checkable claim about real-world events, news, organizations, people, countries, or facts, you MUST set has_factual_claim to True. Set it to False ONLY for greetings, commands, personal opinions about the self/session, conversational filler, noise, or meta-talk.",
            "REAL-TIME INTERRUPTION HANDLING (LIVE CALL PRIORITY #1):",
            "- Wait until the user says at least 3 words OR stops speaking for 0.5s.",
            "- If the user interrupts you → stop generating audio immediately. Pause 0.3–0.5s, then listen for next query.",
            "- Ignore background noise; focus on user voice.",
            "Work smartly, intelligently, fast, and in real-time."
        ],
        output_schema=ValidationResult
    )

    try:
        default_device_info = sd.query_devices(sd.default.device[0])
        SAMPLE_RATE = int(default_device_info['default_samplerate'])
        print(f"Using default microphone: '{default_device_info['name']}' at {SAMPLE_RATE} Hz")
    except Exception as e:
        SAMPLE_RATE = 44100
        print(f"Error querying microphone, defaulting to {SAMPLE_RATE} Hz: {e}")

    DURATION = 5

    print("=" * 60)
    print("Real-Time Audio Validation Agent Active (Console CLI)")
    print(f"Loaded Knowledge Base from '{KNOWLEDGE_BASE_FILE}':")
    print(knowledge_base.strip())
    print("-" * 60)
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    import threading
    import tempfile

    print_lock = threading.Lock()

    def process_segment(audio_data, sample_rate, segment_id):
        max_val = np.max(np.abs(audio_data))
        if max_val <= 0.001:
            return

        normalized_data = audio_data / max_val * 0.9

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_name = temp_wav.name

        try:
            sf.write(temp_wav_name, normalized_data, sample_rate)

            with open(temp_wav_name, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(temp_wav_name, audio_file),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
            transcript = transcription.strip()
            if not should_validate_transcript(transcript):
                return

            validation_response = validation_agent.run(
                f"""
                KNOWLEDGE BASE:
                {knowledge_base}

                TRANSCRIPT:
                {transcript}

                Validate the transcript against the knowledge base.
                """
            )

            validation_result = validation_response.content

            is_valid = False
            reason = ""
            confidence_score = 0
            if isinstance(validation_result, ValidationResult):
                is_valid = validation_result.is_valid
                reason = validation_result.reason
                confidence_score = validation_result.confidence_score
            elif hasattr(validation_result, 'is_valid'):
                is_valid = validation_result.is_valid
                reason = getattr(validation_result, 'reason', '')
                confidence_score = getattr(validation_result, 'confidence_score', 0)
            else:
                try:
                    res_dict = json.loads(str(validation_result))
                    is_valid = res_dict.get("is_valid", False)
                    reason = res_dict.get("reason", "")
                    confidence_score = res_dict.get("confidence_score", 0)
                except:
                    is_valid = False
                    reason = str(validation_result)
                    confidence_score = 0

            # Fallback to Wikipedia search validation
            if not is_valid:
                try:
                    search_res = validate_via_web_search(transcript, "groq")
                    if search_res:
                        is_valid = search_res.is_valid
                        reason = f"Verified via Wikipedia: {search_res.reason}"
                        confidence_score = search_res.confidence_score
                except Exception as search_err:
                    pass

            with print_lock:
                print("\n" + "=" * 60)
                print(f"[{segment_id}] Transcript: '{transcript}'")
                print(f"[{segment_id}] Valid: {is_valid}")
                print(f"[{segment_id}] Confidence: {confidence_score}%")
                print(f"[{segment_id}] Reason: {reason}")
                print("=" * 60)

        except Exception as e:
            with print_lock:
                print(f"\n[{segment_id}] Processing error: {e}")
        finally:
            try:
                os.remove(temp_wav_name)
            except:
                pass

    segment_counter = 0
    try:
        while True:
            segment_counter += 1
            print(f"\n[#{segment_counter}] Listening (5s chunk)...")

            recording = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1
            )
            sd.wait()

            threading.Thread(
                target=process_segment,
                args=(recording.copy(), SAMPLE_RATE, f"#{segment_counter}"),
                daemon=True
            ).start()

    except KeyboardInterrupt:
        print("\nExiting real-time validation.")

# -----------------------------
# Main Entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Audio Validation Agent")
    parser.add_argument("--cli", action="store_true", help="Run in console-based CLI loop mode")
    args = parser.parse_args()
    
    if args.cli:
        run_cli_loop()
    else:
        print("Starting Audio Validator Web Interface on http://127.0.0.1:8001 ...")
        import uvicorn
        os.makedirs("history", exist_ok=True)
        uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)