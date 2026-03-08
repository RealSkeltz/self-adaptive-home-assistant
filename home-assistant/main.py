import whisper
import sounddevice as sd
import collections
import webrtcvad
import numpy as np
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import ollama
load_dotenv()

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30
vad = webrtcvad.Vad(2)

# Tools
def get_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%H:%M")

conversation_history = [
    {"role": "system", "content": "You are Bob, a smart home assistant running on a Raspberry Pi. Be concise and helpful."}
]

def final_answer(answer: str) -> str:
    """Provide the final answer to the user and end the conversation turn."""
    return answer

TOOLS = [get_time, final_answer]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}

def run(user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    
    messages = conversation_history.copy()
    
    while True:
        response = ollama.chat(
            model="qwen3.5:2b",
            messages=messages,
            tools=TOOLS,
            think=False,
            keep_alive=-1,
        )
        
        if response.message.tool_calls:
            messages.append(response.message)
            for tool_call in response.message.tool_calls:
                print(f"🔧 Tool called: {tool_call.function.name}({tool_call.function.arguments})")
                fn = TOOL_MAP.get(tool_call.function.name)
                result = fn(**tool_call.function.arguments) if fn else "Tool not found"
                print(f"🔧 Tool result: {result}")
                messages.append({"role": "tool", "content": str(result)})
                
                if tool_call.function.name == "final_answer":
                    conversation_history.append({"role": "assistant", "content": result})
                    return result
        else:
            # no tool call, plain response
            content = response.message.content
            conversation_history.append({"role": "assistant", "content": content})
            return content

def speak(text):
    subprocess.run(["say", "-v", "Daniel", text])

def record_until_silence():
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
            recording = record_until_silence()
            result = self.ears.transcribe(recording, fp16=False)
            user_input = result["text"].strip()
        elif self.mode == 'text':
            user_input = input("You: ")
        
        try:
            response = run(user_input)
        except Exception as e:
            print(f"Error: {e}")
            response = "Sorry, something went wrong."
        
        print(f"Bob: {response}")
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