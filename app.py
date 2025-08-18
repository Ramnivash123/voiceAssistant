import pyttsx3
import sounddevice as sd
import numpy as np
import whisper
from docx import Document
import re
import threading
import time
import sys
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ======================
# INITIAL SETUP
# ======================
engine = pyttsx3.init()
model = whisper.load_model("base")

# Global timer
total_time = 10 * 60  # 10 minutes in seconds
remaining_time = total_time
stop_flag = False

# ======================
# TIMER THREAD
# ======================
def countdown_timer():
    global remaining_time, stop_flag
    while remaining_time > 0 and not stop_flag:
        time.sleep(1)
        remaining_time -= 1
    if remaining_time <= 0:
        engine.say("Time is up! Saving your answers.")
        engine.runAndWait()
        stop_flag = True

# ======================
# SPEECH INPUT FUNCTION
# ======================
def get_speech_input(prompt_text):
    global remaining_time, stop_flag
    if stop_flag:   # stop if timer expired
        return "timeout"

    engine.say(prompt_text)
    engine.runAndWait()
    
    print(prompt_text)
    duration = 5  # seconds for recording
    sample_rate = 16000
    
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    
    audio_data = np.squeeze(audio_data)
    result = model.transcribe(audio_data, fp16=False)
    
    # Clean text -> remove punctuation and lowercase
    text = result["text"].strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    
    print(f"You said: {text}")
    
    # Special command: check remaining time
    if text == "time":
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        msg = f"{minutes} minutes {seconds} seconds remaining."
        print(msg)
        engine.say(msg)
        engine.runAndWait()
        return "repeat"  # ask again after telling time
    
    return text

# ======================
# ASK NAME AND SUBJECT
# ======================
user_name = get_speech_input("Please say your name.")
if user_name in ["timeout", ""]:
    engine.say("Could not detect your name. Exiting.")
    engine.runAndWait()
    sys.exit()

subject = get_speech_input("Please say your subject.")
if subject in ["timeout", ""]:
    engine.say("Could not detect the subject. Exiting.")
    engine.runAndWait()
    sys.exit()

if subject != "mathematics":
    engine.say("Only mathematics quiz is available. Exiting.")
    engine.runAndWait()
    sys.exit()

engine.say(f"Welcome {user_name}. Starting the mathematics quiz now.")
engine.runAndWait()

# Start timer in background only after verification
timer_thread = threading.Thread(target=countdown_timer, daemon=True)
timer_thread.start()

# ======================
# LOAD QUESTIONS
# ======================


# ======================
# LOAD QUESTIONS (from image)
# ======================
img = Image.open("sample1.png")

# Extract raw text from image
raw_text = pytesseract.image_to_string(img)

# Split into lines
lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

qa_pairs = []
current_question = []

for text in lines:
    # Detect if line starts with digit + dot (like "1." , "2." etc.)
    if re.match(r"^\d+\.", text):
        if current_question:
            qa_pairs.append(current_question)
        current_question = [text]
    else:
        current_question.append(text)

if current_question:
    qa_pairs.append(current_question)


# ======================
# OUTPUT DOCUMENT
# ======================
new_doc = Document()
new_doc.add_paragraph(f"Name: {user_name}")
new_doc.add_paragraph(f"Subject: {subject}\n")

for q_block in qa_pairs:
    if stop_flag:
        break

    question_text = "\n".join(q_block)
    
    while not stop_flag:  # loop until answered or skipped
        engine.say(question_text)
        engine.runAndWait()
        print("\n" + question_text)
        
        answer = get_speech_input("Please speak your answer for this question.")
        
        if answer == "timeout":
            break  # timer ended
        elif answer == "repeat":
            print("Repeating the question...")
            continue
        elif answer == "skip":
            print("Skipping this question...")
            answer = "Not Answered"
        
        # Save question + answer
        new_doc.add_paragraph(question_text)
        new_doc.add_paragraph(f"Answer: {answer}\n")
        break

# ======================
# SAVE AND EXIT
# ======================
new_doc.save("sample2.docx")
print("All questions and answers have been saved to sample2.docx")
sys.exit()
