import json
import os
import urllib.parse

import json5

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

def init_agent_service(model_name='qwen2-1.5b-instruct', system_message="You are a helpful assistant."):
    llm_cfg = {'model': model_name, 'model_server': 'dashscope', 'api_key': os.getenv("DASHSCOPE_API_KEY"),
               # (Optional) LLM hyperparameters for generation:
            'generate_cfg': {
                'top_p': 0.5,
                'temperature': 0.5,
            }
    }
    tools = [
        'code_interpreter',
    ]  # code_interpreter is a built-in tool in Qwen-Agent
    bot = Assistant(
        llm=llm_cfg,
        name='AI Math Solver',
        description='AI Math Solver',
        system_message=system_message,
        function_list=tools,
    )

    return bot

class QwenAgent:
    def __init__(self, model_name='qwen2-1.5b-instruct', system_message="You are a helpful assistant.", user_prompt_format="{problem}"):
        self.bot = init_agent_service(model_name, system_message)
        self.user_prompt_format = user_prompt_format

    def answer(self, problem):
        messages = [{'role': 'user', 'content': self.user_prompt_format.format(problem=problem)}]
        responses = []
        for response in self.bot.run(messages=messages):
            responses.append(response)
        return responses
    
    def test(self, problem):
        messages = [{'role': 'user', 'content': self.user_prompt_format.format(problem=problem)}]
        for response in self.bot.run(messages=messages):
            print('bot response:', response)


if __name__ == '__main__':
    problem = r"Abigail spent 60% of her money on food, and 25% of the remainder on her phone bill. After spending $20 on entertainment, she is left with $40. How much money did Abigail have initially?"
    qwen_agent = QwenAgent()
    qwen_agent.answer(problem)
