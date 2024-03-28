# coding: utf-8

MAIGPT_BASE_URL = 'https://maigpt.in.taou.com/rpc/platforms/go_pbs/maigpt/proxy/v1'
EMBEDDINGS_URL = 'https://maigpt.in.taou.com/rpc/platforms/go_pbs/maigpt/proxy/v1/embeddings'
EMBEDDINGS_MODEL = 'text-embedding-ada-002'


class AgentStepState(object):
    INTRODUCTION = '自我介绍'
    FORMATION = '选择牌阵'
    PREDICTION = '塔罗预测'
    RESPONSE = '回复消息'
    END = '结束对话'
