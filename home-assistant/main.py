import whisper
import sounddevice as sd
import collections
import webrtcvad
import numpy as np
import os
from pathlib import Path
import json
import uuid
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import ollama
load_dotenv()

# Helper functions
SESSION_ID = str(uuid.uuid4())

if os.getenv("ENV") == "prd":
    BASE_PATH = Path("/home/realskeltz/bob")
else:
    BASE_PATH = Path("/Users/jscheltema/Documents/Personal/Home Assistant")

LOGS_FILE = BASE_PATH / "home-assistant/logs.json"

def save_logs():
    try:
        with open(LOGS_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[SESSION_ID] = {
        "started_at": datetime.now().isoformat(),
        "messages": conversation_history.copy()
    }
    with open(LOGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30
vad = webrtcvad.Vad(2)

# Tools
def get_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%H:%M")

def exit_conversation() -> str:
    """Exit the conversation and say goodbye."""
    return "__EXIT__"

TOOLS = [get_time, exit_conversation]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

conversation_history = [
    {"role": "system", "content": (
        "You are Bob, a concise and helpful AI home assistant. "
        "You have access to the following tools:\n"
        "- get_time(): returns the current time\n"
        "- exit_conversation(): call this when the user wants to end the conversation (e.g. bye, goodbye, quit, exit)\n"
        "Always use tools when appropriate. Never simulate tool results in text."
    )}
]

TOOLS = [get_time, exit_conversation]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

def run(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    messages = conversation_history.copy()
    
    while True:
        response = ollama.chat(
            model="qwen3.5:0.8b",
            messages=messages,
            tools=TOOLS,
            think=False,
            keep_alive=-1,
        )
        
        if response.message.tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.message.tool_calls
                ]
            }
            messages.append(assistant_msg)
            conversation_history.append(assistant_msg)

            for tool_call in response.message.tool_calls:
                print(f"🔧 Tool called: {tool_call.function.name}({tool_call.function.arguments})")
                fn = TOOL_MAP.get(tool_call.function.name)
                result = fn(**tool_call.function.arguments) if fn else "Tool not found"
                print(f"🔧 Tool result: {result}")

                if tool_call.function.name == "exit_conversation":
                    return "__EXIT__"

                tool_msg = {"role": "tool", "content": str(result)}
                messages.append(tool_msg)
                conversation_history.append({
                    "role": "tool",
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "content": str(result)
                })
        else:
            content = response.message.content
            conversation_history.append({"role": "assistant", "content": content})
            return content
        
import platform

def speak(text):
    if platform.system() == "Darwin":
        subprocess.run(["say", "-v", "Daniel", text])
    else:
        subprocess.run(["espeak", text])

def listen():
    ring_buffer = collections.deque(maxlen=SILENCE_THRESHOLD)
    audio_buffer = []
    started = False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        print("Listening...")
        while True:
            frame, _ = stream.read(FRAME_SIZE)
            is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            if is_speech:
                started = True
                ring_buffer.clear()
                audio_buffer.append(frame)
            elif started:
                ring_buffer.append(frame)
                audio_buffer.append(frame)
                if len(ring_buffer) == SILENCE_THRESHOLD:
                    print("Done.")
                    break
    return np.concatenate(audio_buffer).squeeze().astype(np.float32) / 32768.0

class HomeAssistant:
    def __init__(self, mode='text'):
        self.ears = whisper.load_model("tiny")
        self.mode = mode

    def interact(self):
        if self.mode == 'voice':
            recording = listen()
            result = self.ears.transcribe(recording, fp16=False)
            user_input = result["text"].strip()
            print(f"You: {user_input}")
        elif self.mode == 'text':
            user_input = input("You: ")
        
        try:
            response = run(user_input)
        except Exception as e:
            print(f"Error: {e}")
            response = "Sorry, something went wrong."

        if response == "__EXIT__":
            goodbye = "Goodbye!"
            print(f"Bob: {goodbye}")
            if self.mode == 'voice':
                speak(goodbye)
            save_logs()
            raise SystemExit

        print(f"Bob: {response}")
        save_logs()
        if self.mode == 'voice':
            speak(response)

    def run(self):
        while True:
            self.interact()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    bob = HomeAssistant(mode=args.mode)
    bob.run()