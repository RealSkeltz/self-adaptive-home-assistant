import asyncio
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient, AssistantMessage, TextBlock
import whisper
import pyttsx3
import time

class HomeAssistant:
    def __init__(self):
        self.voice = pyttsx3.init(driverName='nsss')
        self.ears = whisper.load_model("tiny")
    
        @tool("get_time", "Get the current time", {})
        async def get_time(args):
            from datetime import datetime
            return {
                "content": [{"type": "text", "text": f"Current time: {datetime.now().strftime('%H:%M:%S')}"}]
            }

        server = create_sdk_mcp_server(
            name="home-tools",
            version="1.0.0",
            tools=[get_time]
        )

        self.options = ClaudeAgentOptions(
            system_prompt="You are a helpful home assistant. You have text to speech capabilities, " \
            "so use conversational language and no smileys."\
            "You have the demeanor of an old but sharp British butler like Alfred from Batman.",
            mcp_servers={"home": server},
            allowed_tools=["mcp__home__get_time"]
        )

    async def _interact(self):
        options = ClaudeAgentOptions(
            system_prompt="You are a helpful home assistant. You have text to speech capabilities, so use conversational language and no smileys."\
            "You have the demeanor of an old but sharp British butler like Alfred."
            "You are polite and brief in nature."
        )
        async with ClaudeSDKClient(options=options) as client:
            while True:
                recording = record_until_silence()
                result = self.ears.transcribe(recording, fp16=False)
                user_input = result['text']

                if user_input.lower() in ["exit", "quit"]:
                    break

                await client.query(prompt=user_input)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"Assistant: {block.text}")
                                self.voice.say(block.text)
                                self.voice.runAndWait()
                                while self.voice.isBusy():
                                    time.sleep(0.1)

    def comeAlive(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._interact())
        except RuntimeError:
            asyncio.run(self._interact())


# Helper functions
import sounddevice as sd
import numpy as np
import webrtcvad
import collections

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms, webrtcvad supports 10, 20, or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30  # number of silent frames before stopping

vad = webrtcvad.Vad(2)  # aggressiveness 0-3, higher = more aggressive filtering

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

if __name__ == "__main__":
    bob = HomeAssistant()
    bob.comeAlive()