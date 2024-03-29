# coding: utf-8
from config import *
from template import *
from tools import *

import requests
import simplejson as json
from typing import Annotated, TypedDict

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, create_openai_functions_agent, load_tools
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
        self.open_ai_key = 'sk-'
        self.llm = None
        self.memory_key = 'chat_history'
        self.memory = None
        self.callbacks = []
        self.tools = []
        self.chat_history = ChatHistory(user)
        self.format_agent = None
        self.predict_agent = None
        self.supervisor_agent = None
        self.graph = None

    def init_context(self):
        self.llm = ChatOpenAI(temperature=0.5, model='gpt-4', verbose=True, openai_api_base=MAIGPT_BASE_URL,
                              openai_api_key=self.open_ai_key, callbacks=self.callbacks)
        self.tools = [search_tool]
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
                                          return_intermediate_steps=True, callbacks=self.callbacks,
                                          output_parser=format_output_parser)

    def create_predict_agent(self):
        system_message = SystemMessage(content=PredictionTemplate,
                                       additional_kwargs={"format_instructions": predict_instructions})
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)]
        )
        _agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.predict_agent = AgentExecutor(agent=_agent, tools=self.tools, memory=self.memory, verbose=True,
                                           return_intermediate_steps=True, callbacks=self.callbacks,
                                           output_parser=predict_output_parser)

    def create_supervisor_agent(self):
        system_message = SystemMessage(content=SupervisorTemplate)
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=self.memory_key)],
        )
        _agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=prompt)
        self.supervisor_agent = AgentExecutor(agent=_agent, tools=self.tools, memory=self.memory, verbose=True,
                                              return_intermediate_steps=True, callbacks=self.callbacks,
                                              output_parser=supervisor_output_parser)

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
        print("enter call_supervisor")
        if state['next_action'] in [AgentStepState.RESPONSE, AgentStepState.END]:
            return {'next_action': state['next_action']}
        response = self.supervisor_agent.invoke({"input": state['messages'][-1].content})
        json_response = supervisor_output_parser.parse(response['output'])
        action = json_response['step']
        return {'next_action': action, 'cur_state': action}

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
        if recommend_messages:
            ai_recommend_data = {'messages': recommend_messages}
            ai_content = json.dumps(ai_recommend_data)
            print(ai_content)
        return {'next_action': AgentStepState.END}

    def format(self, state):
        print("enter format")
        response = self.format_agent.invoke({"input": state['messages'][-1].content})
        messages = []
        json_res = format_output_parser.parse(response['output'])
        for res in json_res["results"]:
            messages.append(AIMessage(content=f'''
            推荐牌阵：{res['formation']}
            推荐原因：{res['reason']}
            '''))
        return {"messages": messages, 'next_action': AgentStepState.RESPONSE}

    def predict(self, state):
        print("enter predict")
        response = self.predict_agent.invoke({"input": state['messages'][-1].content})
        messages = []
        json_res = predict_output_parser.parse(response['output'])
        messages.append(AIMessage(content=f'''
        塔罗牌解读：{json_res['predict']}
        '''))
        return {"messages": messages, 'next_action': AgentStepState.RESPONSE}

    def receive_message(self, text):
        self.init_context()
        inputs = {"messages": [HumanMessage(content=text)]}
        self.chat_history.add_chat_message(text, is_human=True)  # todo 删掉这个，改成数据库
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print("agent graph output from node %s: %s " % (key, value))


def do_xxx():
    key = "Bearer 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"
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
    response = requests.post("https://api.openai.com/v1/chat/completions",
                             json=data, headers=headers, timeout=240)
    print(response.json())


if __name__ == '__main__':
    taluo = TaLuoAgent(user='syj')
    taluo.chat_history.add_chat_message("你好，我想测一下最近和女友的情感发展", is_human=True)
    # taluo.receive_message('你好，我想测一下最近和女友的情感发展')
    res_messages = [AIMessage(
        content='\n            推荐牌阵：恋人金字塔\n            推荐原因：恋人金字塔牌阵专门用于解答恋爱走向问题，它可以帮助你理解你和女友各自的期望，目前的关系状态，以及未来可能的发展，这对于你的问题非常合适。\n            '),
        AIMessage(
            content='\n            推荐牌阵：吉普赛十字\n            推荐原因：吉普赛十字牌阵可以帮助你了解你和女友的想法，存在的问题，目前的环境，以及关系的发展结果，这对于你想要了解的情感发展非常有帮助。\n            '),
        AIMessage(
            content='\n            推荐牌阵：圣三角\n            推荐原因：圣三角牌阵适用于任何包含逻辑链的占卜主题，它可以帮助你理解你和女友关系发展的原因，现状，以及可能的结果，这对于你的问题也有一定的参考价值。\n            ')]
    for res_message in res_messages:
        taluo.chat_history.add_chat_message(res_message.content, is_ai=True)
    taluo.receive_message('我选择圣三角牌阵\n 第一张牌是愚者正位\n 第二张牌是恶魔逆位\n 第三张牌是星币六逆位\n')
    # do_xxx()
