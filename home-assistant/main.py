import os
import asyncio
import json
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient, AssistantMessage, TextBlock
import whisper
import sounddevice as sd
import numpy as np
import webrtcvad
import collections

LOGS_FILE = Path("/Users/jscheltema/Documents/Personal/Home Assistant/home-assistant/logs.json")
SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_THRESHOLD = 30

vad = webrtcvad.Vad(2)

def speak(text):
    subprocess.run(["say", "-v", "Daniel", text])

def load_logs():
    if LOGS_FILE.exists():
        return json.loads(LOGS_FILE.read_text())
    return {}

def save_logs(logs):
    LOGS_FILE.write_text(json.dumps(logs, indent=2))

def log_message(logs, session_id, role, text):
    logs[session_id]["messages"].append({
        "role": role,
        "text": text,
        "timestamp": datetime.now().isoformat()
    })
    save_logs(logs)

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
    def __init__(self):
        self.ears = whisper.load_model("tiny")
        self.session_id = str(uuid.uuid4())
        self.should_stop = False

        self.logs = load_logs()
        self.logs[self.session_id] = {"started_at": datetime.now().isoformat(), "messages": []}
        save_logs(self.logs)

        @tool("shutdown", "Shut down the home assistant", {})
        async def shutdown(args):
            self.should_stop = True
            return {
                "content": [{"type": "text", "text": "Shutting down."}]
            }

        @tool("get_time", "Get the current time", {})
        async def get_time(args):
            return {
                "content": [{"type": "text", "text": f"Current time: {datetime.now().strftime('%H:%M:%S')}"}]
            }

        server = create_sdk_mcp_server(
            name="home-tools",
            version="1.0.0",
            tools=[get_time, shutdown]
        )

        self.options = ClaudeAgentOptions(
            system_prompt="You are a helpful home assistant. You have text to speech capabilities, "
                          "so use conversational language and no smileys. "
                          "You have the demeanor of an old but sharp British butler like Alfred from Batman. "
                          "You are polite and brief in nature. "
                          "If the user asks you to stop, shut down, or says goodnight, call the shutdown tool.",
            mcp_servers={"home": server},
            allowed_tools=["mcp__home__get_time", "mcp__home__shutdown"]
        )

    async def _interact(self):
        async with ClaudeSDKClient(options=self.options) as client:
            while True:
                recording = record_until_silence()
                result = self.ears.transcribe(recording, fp16=False)
                user_input = result['text'].strip()

                if user_input.lower() in ["exit", "quit"]:
                    break

                print(f"You: {user_input}")
                log_message(self.logs, self.session_id, "user", user_input)

                await client.query(prompt=user_input)

                full_response = ""
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                full_response += block.text

                if full_response:
                    print(f"Assistant: {full_response}")
                    log_message(self.logs, self.session_id, "assistant", full_response)
                    speak(full_response)

                if self.should_stop:
                    break

    def comeAlive(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._interact())
        except RuntimeError:
            asyncio.run(self._interact())


if __name__ == "__main__":
    bob = HomeAssistant()
    bob.comeAlive()