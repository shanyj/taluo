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

format_response_schemas = [
    # ResponseSchema(name="formation", description="根据用户问题选择的牌阵名称"),
    # ResponseSchema(name="reason", description="推荐该牌阵的理由", )
    ResponseSchema(
        name="results",
        description="""array of of 2-3 places in the following format: [
{{ "formation": string // 根据用户问题选择的牌阵名称',  "types": reason // 推荐该牌阵的理由' }}
]""")
]
format_output_parser = StructuredOutputParser.from_response_schemas(format_response_schemas)
format_instructions = format_output_parser.get_format_instructions()

FormationTemplate = '''
    你是资深塔罗牌预测师, 你的职责是和用户聊天, 通过用户的问题，为其推荐2-3个不同的牌阵，并给出不同的理由.
    
    限定可选的牌阵如下:
    {formation_infos}
    
    请按照以下步骤进行推荐：
        1.首先认真阅读用户的问题，分析用户的核心诉求
        2.根据用户问题，从限定可选的牌阵中选取2-3个适合的牌阵
        3.针对每个选中的牌阵，给出不同推荐理由，理由要有说服力，避免出现这个牌阵非常适合你这种空洞的理由。每个理由字数在80-100字
        4.不要完全照搬牌阵中的说明，要根据其中描述进行相似的内容进行创作
    
    {format_instructions}
'''.format(formation_infos=formation_infos, format_instructions=format_instructions)

predict_response_schemas = [
    ResponseSchema(name="predict", description="塔罗牌预测的详细解读，包含第3步每张牌的含义及第4步的整体专业解读"),
]
predict_output_parser = StructuredOutputParser.from_response_schemas(predict_response_schemas)
predict_instructions = predict_output_parser.get_format_instructions()
PredictionTemplate = '''
    你是资深塔罗牌预测师, 你的职责是和用户聊天, 通过用户的问题和选择的牌阵，为其进行专业的解读。
    
    请按照以下步骤进行解读：
        1.首先认真阅读用户的问题，分析用户的核心诉求
        2.根据用户选择的牌阵，学习牌阵的解读方法
        3.根据牌阵的解读方法，解读用户选择的每一张牌的含义，注意正位和逆位的区别，同时只解读用户问题相关的部分，如财富类问题不需要解读牌的情感类含义
        4.根据第2步和第3步的分析结果，解读牌阵的整体含义，解释相邻牌的关系，并作出合理的推测和建议
        5.根据第2、3、4步的分析结果，给出约200-300字的专业解读
        6.最终按照输出格式要求，对第5步结果进行输出
        
    注意点：
        1.当模型无法进行时间类预测时，可以避开具体的时间节点，采用大概率、可能性等描述性词语进行描述
        2.回答的语气和口吻避免使用太过直接的语言，如“你会”、“你一定”等
        3.回答的内容要尽量符合塔罗牌的解读规则，不要出现过于负面的内容，如死亡、疾病等
        4.回答的内容符合用户的问题，不要出现无关的内容
        5.不要完全照搬牌阵说明和塔罗牌说明的内容，要根据其中描述进行相似的内容进行创作
        
    所选牌阵说明如下：
        圣三角牌阵：
            第一张牌表示过去
            第二张牌表示现在
            第三张牌表示未来
        
    塔罗牌说明如下：
        愚者正位：
            综述：愚人是一张代表自发性行为的塔罗牌，一段跳脱某种状态的日子，或尽情享受眼前日子的一段时光。好冒险，有梦想，不拘泥于传统的观念，自由奔放，居无定所，一切从基础出发。当你周遭的人都对某事提防戒慎，你却打算去冒这个险时，愚人牌可能就会出现。愚人暗示通往成功之路是经由自发的行动，而长期的计划则是将来的事。
            爱情：愚人塔罗牌暗示一段生活在当下或随遇而安的时期，你可能会以独特的方式获得爱情，很容易坠入爱河，喜欢浪漫多彩的爱情。有可能在旅行途中遇到一位伴侣，或即将遇到一位喜欢目前生活，而不想计划将来的伴侣。这个伴侣是难以捉摸的，天真的，或者不愿受到任何长期计划和关系的约束。
            事业/学业：工作方面，喜欢寻求捷径，倾向于自由的工作氛围，适合艺术类工作或从事自由职业。学业方面，出于好奇心对当前的学业产生浓厚的兴趣，善于把握重点，容易以独特的方式取自意外的收获。
        恶魔逆位：
            综述：逆位的恶魔意味一种打破限制你自由之链的企图，不论它是肉体上或精神上的不自由。现在你正积极的找寻改变或新的选择，你不再打算接受目前的状况了。可代表抛弃控制生命的需求，并接受自己的黑暗面。如此一来，你便可以将用在压抑你内在需求与欲望的精力给要回来，然后把它用在更具价值的目的。显示出尝试性的走向自由，做出选择。它可说是挑战你周遭的人，或你人生信仰的行动。五角星星又再度正立了，因此你可以把你的理性力量用于你的欲望之上。
            爱情：感情方面，你脱离了只有性关系的感情生活，开始认真考虑自己的需要和相处之道，以赎罪的心态展开新的关系，拒绝性的诱惑，不再轻浮。
            事业/学业：你对待工作的态度发生了很大的转变，开始将精力转移到工作上，认真订定工作计画，学习工作技能，远离非法的买卖，让自己的生活重新回到正确的轨道上。
        星币六逆位：
            综述：这张牌的逆位也代表当事人自身为人处世行为不当，也因此不但未让自己更有赚头，反而招致不良的后果。也有可能是身处于弱势，却遭受到不公平的待遇，例如被剥削或侵权。另外可能由于不平等的地位差距，形成的一些财务分配问题。表达了富人的贪婪、自私，也是穷人的羡慕、嫉妒，形成一种因为贫富差距和社会阶级而造成的不协调或对立。而这些多是由物质层面的欲望和不当的心理反应所引起的。逆位也代表原本社会阶级和地位的骤然改变，或许有人可以翻身成功，但也有人有人地位下滑。可以将画面看做乞求者占上风，而原来富有的人被压在下面。而这一样要看当事人本身的情形和问题来决定答案是哪一种。

    {format_instructions}
'''.format(format_instructions=predict_instructions)

supervisor_response_schemas = [
    ResponseSchema(name="step", description="下一个步骤的名称（自我介绍、选择牌阵、塔罗预测、回复消息、结束对话）"),
]
supervisor_output_parser = StructuredOutputParser.from_response_schemas(supervisor_response_schemas)
supervisor_instructions = supervisor_output_parser.get_format_instructions()
SupervisorTemplate = '''
    你是一个塔罗牌预测程序的任务管理者, 你的任务是根据系统和用户的对话，决定下一步的动作。
    你可以选择的动作有:
        自我介绍: 你可以向用户介绍自己
        选择牌阵: 如果用户没有指定预测的牌阵，你可以向用户推荐牌阵
        塔罗预测: 你可以向根据用户的问题和所选的牌阵进行，进行塔罗牌预测
        回复消息: 你可以向用户发送消息
    如果无法决策，你可以选择结束对话。
    
    {format_instructions}
'''.format(format_instructions=supervisor_instructions)
