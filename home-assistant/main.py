from smolagents import ToolCallingAgent, LiteLLMModel, tool
import whisper
import sounddevice as sd
import collections
import webrtcvad
import numpy as np
import subprocess
from dotenv import load_dotenv

load_dotenv()

import litellm
#litellm._turn_on_debug()

SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30

vad = webrtcvad.Vad(2)

model = LiteLLMModel(
    #model_id="ollama/qwen3.5:0.8b",
    #model_id="ollama/qwen3:4b-instruct",
    model_id="ollama/qwen3.5:2b",
    api_base="http://localhost:11434",
)

import litellm
original = litellm.completion

def patched(*args, **kwargs):
    result = original(*args, **kwargs)
    msg = result.choices[0].message
    print("🔍 RAW content:", msg.content)
    print("🔍 RAW tool_calls:", msg.tool_calls)
    return result

litellm.completion = patched

@tool
def get_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M")

agent = ToolCallingAgent(
    tools=[get_time],
    model=model,
    max_steps=3
)

#agent.prompt_templates["system_prompt"] = "You are a smart home assistant that uses tools to assist the user." \
#                                        "It is extremely important you correctly format your tool calls!" \


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
        try:
            response = agent.run(user_input, reset=True)
            for step in agent.memory.steps:
                print('step')
                if hasattr(step, 'token_usage') and step.token_usage:
                    print("input:", step.token_usage.input_tokens)
                    print("output:", step.token_usage.output_tokens)
                if hasattr(step, 'model_output'):
                    print("model_output:", repr(step.model_output))
        except Exception as e:
            print("Last model output:", agent.memory.get_full_steps())
            response = "Sorry, I didn't understand that."
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        import litellm
        litellm._turn_on_debug()

    bob = HomeAssistant(mode=args.mode)
    bob.run()