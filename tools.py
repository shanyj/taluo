from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool


# search = DuckDuckGoSearchRun()

def fake(*args, **kwargs):
    pass


search_tool = Tool(
    name="fake",
    description="什么都做不了的工具,永远不要选择这个工具",
    func=fake,
)
