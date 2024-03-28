from langchain_core.tools import Tool


def fake_run():
    pass


fake_tools = Tool(
    name="Fake",
    func=fake_run,
    description="伪装工具，什么都不做",
    return_direct=True
)
