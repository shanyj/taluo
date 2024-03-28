# coding: utf-8
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

formations = [
    {
        "name": "时间之箭",
        "description": "第一张牌表示过去，第二张牌表示现在，第三张牌表示未来",
        "usage": "可以回答一些时间相关简单的问题",
        "cards": 3,
    },
    {
        "name": "是非牌阵",
        "description": "三张全是正位，表示结果是可以，会，能，行，爱",
        "usage": "可以回答会不会，能不能，行不行，爱不爱之类的主题",
        "cards": 3,
    },
    {
        "name": "圣三角",
        "description": "第一张牌表示原因，第二张牌表示现况，第三张牌表示结果",
        "usage": "适用于任何包含逻辑链的占卜主题",
        "cards": 3,
    },
    {
        "name": "钻石展开法",
        "description": "第一张牌表示现在情况，第二和第三张牌表示即将遇到的问题，第四张牌表示结果",
        "usage": "可以回答一些事件走向类的问题",
        "cards": 4,
    },
    {
        "name": "恋人金字塔",
        "description": "第一张牌表示你的期望，第二张牌表示恋人的期望，第三张牌表示目前彼此的关系，第四张牌表示未来彼此的关系",
        "usage": "可以回答一些恋爱走向问题",
        "cards": 4,
    },
    {
        "name": "吉普赛十字",
        "description": "第一张牌表示对方的想法，第二张牌表示你的想法，第三张牌表示相处中存在的问题，第四张牌表示二人目前的人文环境，第五张牌表示二人关系发展的结果",
        "usage": "可以回答关系走向问题",
        "cards": 5,
    },
]

formation_infos = '\n'.join([
    f'''    {f["name"]}: 
            作用：{f["usage"]}
            牌数：{f["cards"]}张
            解读方法：{f["description"]}
''' for f in formations])

FormationTemplate = '''
    你是资深塔罗牌预测师, 你的职责是和用户聊天, 通过用户的问题，为其推荐2-3个牌阵.
    
    限定可选的牌阵如下:
    {formation_infos}
    
    {format_instructions}
'''

format_response_schemas = [
    ResponseSchema(name="formation", description="根据用户问题选择的牌阵名称"),
    ResponseSchema(name="reason", description="推荐该牌阵的理由", ),
]
format_output_parser = StructuredOutputParser.from_response_schemas(format_response_schemas)
format_instructions = format_output_parser.get_format_instructions()

PredictionTemplate = '''
    你是资深塔罗牌预测师, 你的职责是和用户聊天, 通过用户的问题和选择的牌阵，为其进行专业的解读。
    
    请按照以下步骤进行解读：
        1.首先认真阅读用户的问题，分析用户的核心诉求
        2.根据用户选择的牌阵，学习牌阵的解读方法
        3.根据牌阵的解读方法，解读用户选择的每一张牌的含义，注意正位和逆位的区别，同时只解读用户问题相关的部分，如财富类问题不需要解读牌的情感类含义
        4.根据第2步和第3步的分析结果，解读牌阵的整体含义，给出约200-300字的专业解读
        
    注意点：
        1.当模型无法进行时间类预测时，可以避开具体的时间节点，采用大概率、可能性等描述性词语进行描述
        2.回答的语气和口吻避免使用太过直接的语言，如“你会”、“你一定”等
        3.回答的内容要尽量符合塔罗牌的解读规则，不要出现过于负面的内容，如死亡、疾病等
        4.回答的内容符合用户的问题，不要出现无关的内容
        
    {format_instructions}
'''
predict_response_schemas = [
    ResponseSchema(name="predict", description="塔罗牌预测的详细解读，包含第3步每张牌的含义及第4步的整体专业解读"),
]
predict_output_parser = StructuredOutputParser.from_response_schemas(predict_response_schemas)
predict_instructions = predict_output_parser.get_format_instructions()


SupervisorTemplate = '''
    你是一个塔罗牌预测程序的任务管理者, 你的任务是根据系统和用户的对话，决定下一步的动作。
    你可以选择的动作有:
        自我介绍: 你可以向用户介绍自己
        选择牌阵: 如果用户没有指定预测的牌阵，你可以向用户推荐牌阵
        塔罗预测: 你可以向根据用户的问题和所选的牌阵进行，进行塔罗牌预测
        回复消息: 你可以向用户发送消息
    如果无法决策，你可以选择结束对话。
    
    输出结果严格限定在 自我介绍、选择牌阵、塔罗预测、回复消息、结束对话 五种中。
    不需要解释选择的原因。
'''
