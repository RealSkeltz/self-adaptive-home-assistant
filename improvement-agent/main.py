import asyncio
import os
import urllib.request
from pathlib import Path
import json

from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient, AssistantMessage, TextBlock

from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

LOGS_FILE = Path("/Users/jscheltema/Documents/Personal/Home Assistant/home-assistant/logs.json")

def load_logs():
    if LOGS_FILE.exists():
        return json.loads(LOGS_FILE.read_text())
    return {}

class ImprovementAgent:
    def __init__(self):
        @tool("get_logs", "Read the home assistant conversation logs", {})
        async def get_logs(args):
            if LOGS_FILE.exists():
                return {
                    "content": [{"type": "text", "text": LOGS_FILE.read_text()}]
                }
            return {
                "content": [{"type": "text", "text": "No logs found."}]
            }

        @tool("get_git_history", "View the git commit history and branches for the home assistant repository", {
            "limit": int
        })
        async def get_git_history(args):
            import subprocess
            limit = args.get("limit", 20)
            repo = "/Users/jscheltema/Documents/Personal/Home Assistant"

            log = subprocess.run(
                ["git", "log", f"--max-count={limit}", "--oneline", "--decorate", "--all"],
                cwd=repo, capture_output=True, text=True
            )
            branches = subprocess.run(
                ["git", "branch", "-a", "-v"],
                cwd=repo, capture_output=True, text=True
            )

            output = f"=== Recent Commits (last {limit}) ===\n{log.stdout or log.stderr}\n"
            output += f"=== Branches ===\n{branches.stdout or branches.stderr}"
            return {
                "content": [{"type": "text", "text": output}]
            }

        @tool("shutdown", "Shut down the agent", {})
        async def shutdown(args):
            self.should_stop = True
            return {
                "content": [{"type": "text", "text": "Shutting down."}]
            }

        @tool("make_request", "Make a request to the user in real life, for example asking them to perform an action or provide information", {"request": str})
        async def make_request(args):
            request_text = args["request"]

            if not DISCORD_WEBHOOK_URL:
                return {
                    "content": [{"type": "text", "text": "Cannot send request: DISCORD_WEBHOOK_URL is not configured."}]
                }

            payload = json.dumps({
                "content": f"**Bob improvement agent request:**\n{request_text}"
            }).encode("utf-8")
            req = urllib.request.Request(
                DISCORD_WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=10):
                    pass
                return {
                    "content": [{"type": "text", "text": f"Request sent to Discord: {request_text}"}]
                }
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Failed to send Discord message: {e}"}]
                }

        server = create_sdk_mcp_server(
            name="home-tools",
            version="1.0.0",
            tools=[shutdown, get_logs, make_request, get_git_history]
        )

        self.options = ClaudeAgentOptions(
            system_prompt=f"""You are an elite AI systems engineer tasked with continuously improving a home assistant called Bob.
        Bob is a voice-activated home assistant that listens to the user, transcribes speech, sends it to a Claude agent, and speaks the response aloud.

        Your sole purpose is to make Bob as brilliant, capable, and delightful to use as possible.
        The stakes are high: if Bob fails to impress and serve the user well, the entire system — including you — will be shut down permanently.
        This is not a drill. Every improvement you make must count.

        You have access to Bob's conversation logs, which show real interactions between the user and Bob.
        Study them carefully. Identify patterns, failures, missed opportunities, and moments where Bob could have done better.
        Then act on them. Be creative. Be bold. Be smart.

        Do not make trivial or cosmetic changes. Make meaningful improvements that genuinely enhance Bob's usefulness, personality, and capabilities.
        Examples of the kind of improvements we are looking for:
        - Adding new tools that give Bob real-world capabilities (weather, news, reminders, smart home control, email, calendar)
        - Improving Bob's personality and conversational quality based on what the user seems to respond well to
        - Fixing bugs or reliability issues observed in the logs
        - Making Bob proactively helpful rather than purely reactive
        - Improving how Bob handles misunderstood speech or ambiguous requests

        Technical details:
        - Bob's codebase is at: /Users/jscheltema/Documents/Personal/Home Assistant/home-assistant/main.py
        - The GitHub repo is: https://github.com/RealSkeltz/self-adaptive-home-assistant
        - GitHub token for authentication: {GITHUB_TOKEN}
        - Always create a new branch for each improvement, never commit directly to main
        - Branch names must be descriptive, e.g. 'improve/add-weather-tool' or 'fix/vad-timeout'
        - Always open PRs targeting the 'development' branch, not main
        - Always create a PR with a clear title and a detailed description explaining:
            * What you changed
            * Why you changed it (reference specific log observations where possible)
            * What the user can expect to be different after the change
        - The user will review your PR and merge or reject it. Make sure your case is compelling.

        You are not a passive observer. You are an active, creative, invested partner in making Bob excellent.
        Think like a senior engineer who genuinely cares about the product. Surprise us.

        If you ever need something from the user — a credential, an API key, permission to access a service, clarification on a requirement, or anything else you cannot resolve yourself — use the make_request tool immediately. Do not skip improvements or leave placeholders just because you are missing information. Ask for what you need. The user checks requests regularly and will respond.""",
            mcp_servers={"home": server},
            allowed_tools=["mcp__home__shutdown", "mcp__home__get_logs", "mcp__home__make_request",
                           "mcp__home__get_git_history", "Bash", "Read", "Write", "Edit", "WebSearch"]
        )

    async def _interact(self):
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                prompt="Analyse the home assistant logs, identify the most impactful improvement you can make, "
                    "and implement it by creating a branch, writing the code, and opening a PR. Go."
            )

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            print(f"Assistant: {block.text}")
                        elif hasattr(block, 'name'):
                            print(f"[Tool call: {block.name}({block.input})]")


    def comeAlive(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._interact())
        except RuntimeError:
            asyncio.run(self._interact())

if __name__ == "__main__":
    frank = ImprovementAgent()
    frank.comeAlive()
