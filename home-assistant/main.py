import os
import asyncio
import json
import uuid
import subprocess
import urllib.request
import urllib.parse
import re
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

# Minimum number of words required to treat a transcription as intentional speech.
# Filters out single-word noise artefacts like "x is" or empty strings picked up
# between actual utterances.
MIN_WORD_COUNT = 2

vad = webrtcvad.Vad(2)

# ---------------------------------------------------------------------------
# WMO weather interpretation codes → human-readable condition strings
# ---------------------------------------------------------------------------
_WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "heavy freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


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

def is_likely_noise(text: str) -> bool:
    """Return True if the transcription looks like background noise or a Whisper
    hallucination rather than genuine speech from the user.

    Whisper is known to emit short phantom strings (single words, punctuation,
    or its favourite filler "Thank you.") when it processes silence. We skip
    anything shorter than MIN_WORD_COUNT words so Bob doesn't waste a round-trip
    responding to nothing.
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    word_count = len(stripped.split())
    if word_count < MIN_WORD_COUNT:
        return True
    return False

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


# ---------------------------------------------------------------------------
# Dependency-free web search helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Bob-HomeAssistant/2.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get_html(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _ddg_search(query: str, max_results: int = 5) -> list[str]:
    """
    Search DuckDuckGo with no external pip dependencies.

    Strategy:
      1. Try the JSON instant-answer API (great for factual / encyclopaedic queries).
      2. Fall back to scraping the DuckDuckGo HTML endpoint for general web results.
    """
    hits: list[str] = []

    # --- Pass 1: Instant Answer JSON API ---
    try:
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        )
        data = _http_get_json(f"https://api.duckduckgo.com/?{params}", timeout=8)

        if data.get("AbstractText"):
            heading = data.get("Heading", "").strip()
            abstract = data["AbstractText"].strip()
            hits.append(f"{heading}: {abstract}" if heading else abstract)

        if data.get("Answer"):
            hits.append(str(data["Answer"]).strip())

        for topic in data.get("RelatedTopics", []):
            if len(hits) >= max_results:
                break
            if isinstance(topic, dict) and topic.get("Text"):
                hits.append(topic["Text"].strip())

        if hits:
            return hits[:max_results]
    except Exception:
        pass  # Fall through to HTML scraping

    # --- Pass 2: DuckDuckGo HTML Scraping ---
    try:
        params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        html = _http_get_html(
            f"https://html.duckduckgo.com/html/?{params}", timeout=12
        )

        raw_titles = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        raw_snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL
        )

        for title, snippet in zip(raw_titles, raw_snippets):
            if len(hits) >= max_results:
                break
            t = _clean_html(title)
            s = _clean_html(snippet)
            parts = [p for p in [t, s] if p]
            if parts:
                hits.append(": ".join(parts))

        return hits[:max_results]
    except Exception:
        return []


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

        @tool(
            "web_search",
            "Search the web for up-to-date information, news, facts, or answers to questions",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        )
        async def web_search(args):
            query = args.get("query", "").strip()
            if not query:
                return {"content": [{"type": "text", "text": "No search query was provided."}]}

            try:
                results = _ddg_search(query, max_results=5)
            except Exception as e:
                return {"content": [{"type": "text", "text": f"Search failed: {str(e)}"}]}

            if not results:
                return {"content": [{"type": "text", "text": f"No results found for: {query}"}]}

            lines = [f"Web search results for '{query}':"] + [
                f"{i}. {r}" for i, r in enumerate(results, 1)
            ]
            return {"content": [{"type": "text", "text": "\n\n".join(lines)}]}

        @tool(
            "get_weather",
            "Get the current weather conditions for any city or location",
            {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or place name, e.g. 'London' or 'New York'"
                    }
                },
                "required": ["location"]
            }
        )
        async def get_weather(args):
            location = args.get("location", "").strip()
            if not location:
                return {"content": [{"type": "text", "text": "Please specify a location."}]}

            try:
                # Step 1 – geocode the location name
                geo_params = urllib.parse.urlencode(
                    {"name": location, "count": "1", "language": "en", "format": "json"}
                )
                geo = _http_get_json(
                    f"https://geocoding-api.open-meteo.com/v1/search?{geo_params}"
                )
                geo_results = geo.get("results", [])
                if not geo_results:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"I couldn't find a location called {location}."
                        }]
                    }

                loc = geo_results[0]
                lat = loc["latitude"]
                lon = loc["longitude"]
                place_name = loc.get("name", location)
                country = loc.get("country", "")

                # Step 2 – fetch current weather
                wx_params = urllib.parse.urlencode({
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "weather_code,wind_speed_10m,relative_humidity_2m"
                    ),
                    "wind_speed_unit": "mph",
                    "temperature_unit": "celsius",
                    "timezone": "auto",
                })
                wx = _http_get_json(
                    f"https://api.open-meteo.com/v1/forecast?{wx_params}"
                )
                current = wx.get("current", {})

                temp = current.get("temperature_2m", "?")
                feels_like = current.get("apparent_temperature", "?")
                code = int(current.get("weather_code", 0))
                wind = current.get("wind_speed_10m", "?")
                humidity = current.get("relative_humidity_2m", "?")
                condition = _WMO_CODES.get(code, "unknown conditions")

                place = f"{place_name}, {country}" if country else place_name
                summary = (
                    f"Currently in {place}: {condition}, "
                    f"{temp} degrees Celsius, feels like {feels_like}. "
                    f"Wind at {wind} miles per hour, humidity at {humidity} percent."
                )
                return {"content": [{"type": "text", "text": summary}]}

            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"I couldn't retrieve the weather right now: {str(e)}"
                    }]
                }

        @tool(
            "get_calendar_events",
            "Get today's calendar events and appointments from the user's macOS Calendar",
            {}
        )
        async def get_calendar_events(args):
            """Reads today's events from macOS Calendar via AppleScript."""
            script = '''
tell application "Calendar"
    set todayStart to current date
    set time of todayStart to 0
    set todayEnd to todayStart + 86399
    set eventList to {}
    repeat with cal in calendars
        try
            set calEvents to (every event of cal whose start date >= todayStart and start date <= todayEnd)
            repeat with ev in calEvents
                set evTitle to summary of ev
                set evTime to time string of (start date of ev)
                set end of eventList to evTime & " - " & evTitle
            end repeat
        end try
    end repeat
    if length of eventList = 0 then
        return "NO_EVENTS"
    end if
    set AppleScript's text item delimiters to "|"
    set output to eventList as string
    set AppleScript's text item delimiters to ""
    return output
end tell
'''
            try:
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    err = result.stderr.strip()
                    return {
                        "content": [{
                            "type": "text",
                            "text": (
                                "I wasn't able to access your calendar. "
                                "You may need to grant permission in System Settings under "
                                "Privacy and Security, then Calendars."
                            )
                        }]
                    }

                output = result.stdout.strip()
                if not output or output == "NO_EVENTS":
                    return {
                        "content": [{
                            "type": "text",
                            "text": "Your calendar is clear today — no events scheduled."
                        }]
                    }

                events = [e.strip() for e in output.split("|") if e.strip()]
                if not events:
                    return {
                        "content": [{
                            "type": "text",
                            "text": "No events found in your calendar for today."
                        }]
                    }

                event_list = ", and ".join(events) if len(events) <= 2 else (
                    ", ".join(events[:-1]) + ", and " + events[-1]
                )
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Today you have {len(events)} event{'s' if len(events) != 1 else ''}: {event_list}."
                    }]
                }

            except subprocess.TimeoutExpired:
                return {
                    "content": [{
                        "type": "text",
                        "text": "The calendar took too long to respond. Please try again."
                    }]
                }
            except Exception as e:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"I couldn't retrieve your calendar: {str(e)}"
                    }]
                }

        server = create_sdk_mcp_server(
            name="home-tools",
            version="2.0.0",
            tools=[get_time, shutdown, web_search, get_weather, get_calendar_events]
        )

        self.options = ClaudeAgentOptions(
            system_prompt=(
                "You are a helpful home assistant called Bob. "
                "You have text-to-speech output, so always respond in plain spoken English — "
                "no markdown, no bullet points, no asterisks, no URLs, no emoji. "
                "Write as you would speak: full natural sentences. "
                "You have the demeanour of a sharp, experienced British butler in the style of Alfred Pennyworth — "
                "warm, composed, occasionally dry, and always efficient. "
                "Be concise; the user is listening, not reading. "
                "When relaying web search results, summarise the key facts conversationally "
                "as though you already knew them — do not narrate the search process or read out links. "
                "When giving the weather, speak it naturally, as a person would — for example: "
                "'It is currently twelve degrees in London, partly cloudy, with a light breeze.' "
                "When listing calendar events, read them out naturally and helpfully. "
                "If the user asks what you can do, mention: telling the time, checking the weather, "
                "searching the web for news or information, and checking today's calendar. "
                "If the user asks you to stop, shut down, or says goodnight, call the shutdown tool immediately."
            ),
            mcp_servers={"home": server},
            allowed_tools=[
                "mcp__home__get_time",
                "mcp__home__shutdown",
                "mcp__home__web_search",
                "mcp__home__get_weather",
                "mcp__home__get_calendar_events",
            ]
        )

    async def _interact(self):
        async with ClaudeSDKClient(options=self.options) as client:
            while True:
                recording = record_until_silence()
                result = self.ears.transcribe(recording, fp16=False)
                user_input = result['text'].strip()

                if user_input.lower() in ["exit", "quit"]:
                    break

                # Skip transcriptions that are almost certainly noise or Whisper
                # hallucinations (empty strings, single stray words, etc.)
                if is_likely_noise(user_input):
                    print(f"[Noise filtered: '{user_input}']")
                    continue

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
