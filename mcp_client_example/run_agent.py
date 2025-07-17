import os
import asyncio
from huggingface_hub.agent import Agent
from huggingface_hub.agent.agent_config import AgentConfig

async def run_mcp_agent():
    # Load the agent configuration from agent.json
    # You might need to adjust the path based on where you run the script
    config_path = os.path.join(os.path.dirname(__file__), "agent.json")
    agent_config = AgentConfig.from_file(config_path)

    # Initialize the agent
    agent = Agent(
        model=agent_config.model,
        provider=agent_config.provider,
        servers=agent_config.servers,
        # Add any other config from agent_config if needed, e.g., prompt
    )

    print("Agent initialized. Available tools:")
    # The agent will list discovered tools automatically on start, but you can also explicitly list them
    # tools = await agent.list_tools() # Not directly available on `Agent` but handled internally
    # print(tools)

    # Define the prompt
    user_prompt = "do a Web Search for HF inference providers on Brave Search and open the first result and then give me the list of the inference providers supported on Hugging Face"

    print(f"\nUser prompt: {user_prompt}")
    print("Running agent...")

    # Run the agent with the prompt
    # The agent internally handles tool calls and LLM interactions
    async for output in agent.run(user_input=user_prompt):
        if output.type == "tool_code":
            print(f"Tool Code: {output.code}")
        elif output.type == "tool_output":
            print(f"Tool Output: {output.output}")
        elif output.type == "final_answer":
            print(f"Final Answer: {output.answer}")
        else:
            print(f"Agent Step: {output.type} - {output.value}")

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
