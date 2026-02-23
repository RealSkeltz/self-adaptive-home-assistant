import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

class ImprovementAgent:
    async def _interact(self):
        options = ClaudeAgentOptions(
            system_prompt="You are an agent tasked with improving a different agentic system. " \
            "The other agentic system is a home assistant." \
            "You can push code to the system's codebase and request information, actions or permissions from the human user."
        )
        async with ClaudeSDKClient(options=options) as client:
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break

                await client.query(prompt=user_input)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"Assistant: {block.text}")
   

    def comeAlive(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._interact())
        except RuntimeError:
            asyncio.run(self._interact())

if __name__ == "__main__":
    frank = ImprovementAgent()
    frank.comeAlive()