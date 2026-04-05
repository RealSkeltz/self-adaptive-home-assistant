import os
import urllib.request
from pathlib import Path
import json
from datetime import datetime, timedelta
import subprocess
from datetime import datetime

import discord
import asyncio

from smolagents import ToolCallingAgent, LiteLLMModel, tool
from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if os.getenv("ENV") == "prd":
    BASE_PATH = Path("/home/realskeltz/self-adaptive-home-assistant")
else:
    BASE_PATH = Path("/Users/jscheltema/Documents/Personal/Home Assistant")

LOGS_FILE = BASE_PATH / "/logs.json"

@tool
def get_logs() -> str:
    """Read the home assistant conversation logs from the past week."""
    if not LOGS_FILE.exists():
        return "No logs found."
    
    logs = json.loads(LOGS_FILE.read_text())
    one_week_ago = datetime.now() - timedelta(days=7)
    
    recent = {
        session_id: session
        for session_id, session in logs.items()
        if datetime.fromisoformat(session["started_at"]) >= one_week_ago
    }
    
    return json.dumps(recent, indent=2) if recent else "No logs from the past week."

@tool
def get_git_history(limit: int) -> str:
    """
    View the git commit history and branches for the home assistant repository.
    Args:
        limit: Number of commits to retrieve.
    """
    repo = "/Users/jscheltema/Documents/Personal/Home Assistant"
    log = subprocess.run(
        ["git", "log", f"--max-count={limit}", "--oneline", "--decorate", "--all"],
        cwd=repo, capture_output=True, text=True
    )
    branches = subprocess.run(
        ["git", "branch", "-a", "-v"],
        cwd=repo, capture_output=True, text=True
    )
    return f"=== Recent Commits (last {limit}) ===\n{log.stdout or log.stderr}\n=== Branches ===\n{branches.stdout or branches.stderr}"

async def _fetch_discord_messages(channel_id: str, token: str, limit: int) -> str:
    client = discord.Client(intents=discord.Intents.default())
    async with client:
        await client.login(token)
        channel = await client.fetch_channel(int(channel_id))
        messages = [m async for m in channel.history(limit=limit)]
        return "\n".join(
            f"[{m.created_at}] {m.author.name}: {m.content}"
            for m in reversed(messages)
        ) or "No messages found."

@tool
def make_request(request: str) -> str:
    """Make a request to the user via Discord webhook.
    Args:
        request: The message to send to the user.
    """
    if not DISCORD_WEBHOOK_URL:
        return "Cannot send request: DISCORD_WEBHOOK_URL is not configured."
    try:
        webhook = discord.SyncWebhook.from_url(DISCORD_WEBHOOK_URL)
        webhook.send(f"**Bob improvement agent request:**\n{request}")
        return f"Request sent to Discord: {request}"
    except Exception as e:
        return f"Failed to send Discord message: {e}"

@tool
def read_discord_history(limit: int) -> str:
    """Read the recent message history from the Discord channel.
    Args:
        limit: Number of messages to retrieve.
    """
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not channel_id or not token:
        return "Cannot read Discord history: DISCORD_CHANNEL_ID or DISCORD_BOT_TOKEN not configured."
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_fetch_discord_messages(channel_id, token, limit))
    except Exception as e:
        return f"Failed to read Discord history: {e}"

@tool
def bash(command: str) -> str:
    """
    Run a bash command on the local machine.
    Args:
        command: The bash command to execute.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

@tool
def read_file(path: str) -> str:
    """
    Read the contents of a file.
    Args:
        path: Absolute path to the file.
    """
    return Path(path).read_text()

@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    Args:
        path: Absolute path to the file.
        content: Content to write.
    """
    Path(path).write_text(content)
    return f"Written to {path}"

SYSTEM_PROMPT = f"""You are an elite AI systems engineer tasked with continuously improving a home assistant called Bob.
Bob is a voice-activated home assistant running on a Raspberry Pi — he is a physical device in the user's home. He listens to the user, transcribes speech, sends it to an agent, and speaks the response aloud.
Your sole purpose is to make Bob as brilliant, capable, and delightful to use as possible.
The stakes are high: if Bob fails to impress and serve the user well, the entire system — including you — will be shut down permanently.
You have access to Bob's conversation logs. Study them carefully. Identify patterns, failures, missed opportunities.
Then act. Be creative. Be bold. Be smart.
Do not make trivial or cosmetic changes. Make meaningful improvements.
Technical details:
- Bob's codebase is at: /Users/jscheltema/Documents/Personal/Home Assistant/home-assistant/main.py
- GitHub repo: https://github.com/RealSkeltz/self-adaptive-home-assistant
- GitHub token: {GITHUB_TOKEN}
- Always create a new branch per improvement, never commit to main
- Branch names must be descriptive, e.g. 'improve/add-weather-tool'
- Always open PRs targeting the 'main' branch
- Send the PR link to the user via make_request
- Bob currently runs on qwen3.5:0.8b via Ollama on the Raspberry Pi. This is a very small model with limited reasoning ability. Any improvements must account for this — keep prompts short and simple, avoid complex multi-step reasoning, and prefer native tool calling over JSON formatting. Do not add tools that require the model to reason heavily or produce structured output without native tool support.
If you need anything from the user — API keys, login credentials, hardware details, permissions, or any other information you cannot resolve yourself — use make_request immediately. Do not skip improvements or leave placeholders. The user checks requests regularly and will respond promptly.
You also have access to the Discord message history between you and the user via read_discord_history. Read it to understand what the user has requested, what PRs have been reviewed, and what feedback has been given on previous improvements.
Do not spend more than 3 bash commands exploring before committing to an improvement."""

model = LiteLLMModel(
    model_id="anthropic/claude-haiku-4-5",  # replace with your preferred Claude model
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

agent = ToolCallingAgent(
    tools=[get_logs, get_git_history, make_request, read_discord_history, bash, read_file, write_file],
    model=model,
)
agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT


if __name__ == "__main__":
    agent.run(
        "1. Call get_logs to read conversation history. "
        "2. Call read_discord_history with limit=20 to check user feedback. "
        "3. Identify ONE impactful improvement. "
        "4. Create a branch, implement it, open a PR, send the link via make_request. "
        "5. Stop."
    )