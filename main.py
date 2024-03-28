# coding: utf-8
from config import *
from tools import *
from template import *

import requests
import simplejson as json
from typing import Annotated, TypedDict

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, create_openai_functions_agent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


class ChatMessage(object):
    def __init__(self, message, is_ai=False, is_human=False):
        self.message = message
        self.is_ai = is_ai
        self.is_human = is_human


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_action: str
    cur_state: str


class ChatHistory(object):
    def __init__(self, user):
        self.user = "syj"
        self.chat_map = {
            "syj": [
                ChatMessage("你好，我是塔罗牌大师，我可以帮你解答问题。", is_ai=True),
                # ChatMessage("你好，我想测一下最近和女友的情感发展。", is_human=True)
            ],
        }

    def get_chat_history(self):
        return self.chat_map[self.user]

    def add_chat_message(self, message, is_ai=False, is_human=False):
        self.chat_map[self.user].append(ChatMessage(message, is_ai=is_ai, is_human=is_human))


class TaLuoAgent(object):
    def __init__(self, user):
        self.open_ai_key = 'agent_thorndike'
        self.llm = None
        self.memory_key = 'chat_history'
        self.memory = None
        self.callbacks = []
        self.tools = [fake_tools]
        self.chat_history = ChatHistory(user)
        self.format_agent = None
        self.predict_agent = None
        self.supervisor_agent = None
        self.graph = None

    def init_context(self):
        self.llm = ChatOpenAI(temperature=0, model='gpt-4', verbose=True, openai_api_base=MAIGPT_BASE_URL,
                              openai_api_key=self.open_ai_key, callbacks=self.callbacks)
        self.memory = AgentTokenBufferMemory(memory_key=self.memory_key, llm=self.llm, max_token_limit=4000)
        messages = self.chat_history.get_chat_history()
        for _message in messages:
            if _message.is_ai:
                self.memory.chat_memory.add_ai_message(_message.message)
            elif _message.is_human:
                self.memory.chat_memory.add_user_message(_message.message)
        self.create_format_agent()
        self.create_predict_agent()
        self.create_supervisor_agent()
        self.build_graph()

    def create_format_agent(self):
        system_message = SystemMessage(content=FormationTemplate,
                                       additional_kwargs={"format_instructions": format_instructions,
                                                          "formation_infos": formation_infos})
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)]
        )
        _agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.format_agent = AgentExecutor(agent=_agent, tools=self.tools, memory=self.memory, verbose=True,
                                          return_intermediate_steps=True, callbacks=self.callbacks)

    def create_predict_agent(self):
        system_message = SystemMessage(content=PredictionTemplate,
                                       additional_kwargs={"format_instructions": predict_instructions})
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)]
        )
        _agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.predict_agent = AgentExecutor(agent=_agent, tools=self.tools, memory=self.memory, verbose=True,
                                           return_intermediate_steps=True, callbacks=self.callbacks)

    def create_supervisor_agent(self):
        system_message = SystemMessage(content=SupervisorTemplate)
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)],
        )
        print("prompt: ", prompt)
        print("messages", self.memory)
        _agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.supervisor_agent = AgentExecutor(agent=_agent, tools=self.tools, memory=self.memory, verbose=True,
                                              return_intermediate_steps=True, callbacks=self.callbacks)

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("supervisor", self.call_supervisor)
        workflow.add_node("introduction", self.introduction)
        workflow.add_node("format", self.format)
        workflow.add_node("predict", self.predict)
        workflow.add_node("response", self.response)

        workflow.add_edge("introduction", "supervisor")
        workflow.add_edge("format", "supervisor")
        workflow.add_edge("predict", "supervisor")
        workflow.add_edge("response", "supervisor")

        function_map = {
            AgentStepState.INTRODUCTION: "introduction",
            AgentStepState.FORMATION: "format",
            AgentStepState.PREDICTION: "predict",
            AgentStepState.RESPONSE: "response",
            AgentStepState.END: END
        }
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges("supervisor", lambda x: x['next_action'], function_map)
        self.graph = workflow.compile()

    def call_supervisor(self, state):
        if state['next_action'] in [AgentStepState.RESPONSE, AgentStepState.END]:
            return {'next_action': state['next_action']}
        response = self.supervisor_agent.invoke({"input": state['messages'][-1].content})
        return {'next_action': response['output'], 'cur_state': response['output']}

    def introduction(self, state):
        print("enter introduction")
        contents = [
            "你好~我是专业的的塔罗牌测算师Cindy，可以帮您解答进行爱情、人际关系、工作（学业）等方面塔罗牌预测分析？",
        ]
        messages = [AIMessage(content=content) for content in contents]
        return {"messages": messages, 'next_action': AgentStepState.RESPONSE}

    def response(self, state):
        print("enter response")
        recommend_messages = []
        for _message in state['messages']:
            if _message.type == 'ai':
                recommend_messages.append(_message.content)
                self.chat_history.add_chat_message(_message.content, is_ai=True)  # todo 删掉这个，改成数据库
        if recommend_messages:
            ai_recommend_data = {'messages': recommend_messages}
            ai_content = json.dumps(ai_recommend_data)
            print(ai_content)
        return {'next_action': AgentStepState.END}

    def format(self, state):
        print("enter format")
        response = self.format_agent.invoke({"input": state['messages'][-1].content})
        messages = []
        json_res = json.loads(response['output'])
        for res in json_res:
            messages.append(AIMessage(content=f'''
            推荐牌阵：{res['formation']}
            推荐原因：{res['reason']}
            '''))
        else:
            contents = [
                "咱们推荐选择以下牌阵进行塔罗牌测算："
                "1. 三牌阵：过去、现在、未来"
                "2. 十字牌阵：过去、现在、未来、环境、障碍、希望、结果、内心、外在、未来"
                "3. 爱情牌阵：你、对方、你的态度、对方的态度、你的环境、对方的环境、你的希望、对方的希望、结果",
            ]
            messages = [AIMessage(content=content) for content in contents]
        return {"messages": messages, 'next_action': '发送消息'}

    def predict(self, state):
        print("enter predict")
        response = self.predict_agent.invoke({"input": state['messages'][-1].content})
        messages = []
        json_res = json.loads(response['output'])
        for res in json_res:
            messages.append(AIMessage(content=f'''
            塔罗牌解读：{res['predict']}
            '''))
        return {"messages": messages, 'next_action': '发送消息'}

    def receive_message(self, text):
        self.init_context()
        inputs = {"messages": [HumanMessage(content=text)]}
        self.chat_history.add_chat_message(text, is_human=True)  # todo 删掉这个，改成数据库
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print("agent graph output from node %s: %s " % (key, value))


def do_xxx():
    key = "Bearer userprofile-resume-formatter"
    headers = {"Authorization": key}
    data = {
        "messages": [{
            "content": "\n    你是一个塔罗牌预测程序的任务管理者, 你的任务是根据系统和用户的对话，决定下一步的动作。\n    你可以选择的动作有:\n        自我介绍: 你可以向用户介绍自己\n        选择牌阵: 如果用户没有指定预测的牌阵，你可以向用户推荐牌阵\n        塔罗预测: 你可以向根据用户的问题和所选的牌阵进行，进行塔罗牌预测\n        回复消息: 你可以向用户发送消息\n    如果无法决策，你可以选择结束对话。\n",
            "role": "system"
        }, {
            "content": "你好，我是塔罗牌大师，我可以帮你解答问题。",
            "role": "assistant"
        }, {
            "content": "你好，我想测一下最近和女友的情感发展",
            "role": "user"
        }],
        "model": "gpt-4",
        "temperature": 0.2
    }
    response = requests.post("https://maigpt.in.taou.com/rpc/platforms/go_pbs/maigpt/proxy/v1/chat/completions",
                             json=data, headers=headers, timeout=240)
    print(response.json())


if __name__ == '__main__':
    taluo = TaLuoAgent(user='syj')
    taluo.receive_message('你好，我想测一下最近和女友的情感发展')
    # do_xxx()
