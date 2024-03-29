# coding: utf-8
import os

os.environ['SERPER_API_KEY'] = "773ddb9f6eaa4923c165598bb918e2999edd9ab9150a088d83360187d5142392"

MAIGPT_BASE_URL = 'https://api.openai.com/v1'
EMBEDDINGS_URL = 'https://api.openai.com/v1/embeddings'
EMBEDDINGS_MODEL = 'text-embedding-ada-002'


class AgentStepState(object):
    INTRODUCTION = '自我介绍'
    FORMATION = '选择牌阵'
    PREDICTION = '塔罗预测'
    RESPONSE = '回复消息'
    END = '结束对话'
