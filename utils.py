import re
import simplejson as json


def extra_json(input_text):
    """
    提取字符串中的json结构
    :param input_text:
    '```json
    {
        "step": "选择牌阵"
    }
    ```'
    :return:
    """
    json_str = re.search(r'```json(.*?)```', input_text, re.S)
    if json_str:
        return json.loads(json_str.group(1))
    return None
