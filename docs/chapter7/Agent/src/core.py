from openai import OpenAI
import json
from typing import List, Dict, Any
from src.utils import function_to_json
from src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia, get_current_temperature

import pprint

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List=[], verbose : bool = True):
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


    

