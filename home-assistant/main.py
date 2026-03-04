from smolagents import ToolCallingAgent, LiteLLMModel, tool
import whisper
import sounddevice as sd
import collections
import webrtcvad
import numpy as np
import subprocess
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30

vad = webrtcvad.Vad(2)


@tool
def check_calendar(date: str) -> str:
    """
    Check the user's calendar for appointments on a given date.
    Args:
        date: The date to check in YYYY-MM-DD format.
    """
    return "You have a dentist appointment at 10:00 AM and a team meeting at 3:00 PM."


@tool
def order_groceries(items: str) -> str:
    """
    Order groceries for delivery to the user's home.
    Args:
        items: A comma-separated list of items to order.
    """
    return f"Order placed! The following items will be delivered tomorrow between 2-4 PM: {items}."


model = LiteLLMModel(
    model_id="ollama/qwen3:4b-instruct",
    api_base="http://localhost:11434",
    verbose=True
)

agent = ToolCallingAgent(tools=[check_calendar, order_groceries], model=model)


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
        self.tools = [check_calendar, order_groceries]

    def speak(self, text):
        speak(text)

    def interact(self):
        if self.mode == 'voice':
            recording = record_until_silence()
            result = self.ears.transcribe(recording, fp16=False)
            user_input = result["text"].strip()
        elif self.mode == 'text':
            user_input = input("You: ")

        print(f"You: {user_input}")
        response = agent.run(user_input, reset=False)
        print(f"Bob: {response}")

        if self.mode == 'voice':
            self.speak(response)

    def run(self):
        while True:
            self.interact()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    args = parser.parse_args()
    bob = HomeAssistant(mode=args.mode)
    bob.run()