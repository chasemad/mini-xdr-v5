try:
    from langgraph.checkpoint.memory import MemorySaver

    print("MemorySaver found in langgraph.checkpoint.memory")
except ImportError:
    print("MemorySaver NOT found in langgraph.checkpoint.memory")

try:
    import langgraph_checkpoint

    print("langgraph_checkpoint dir:", dir(langgraph_checkpoint))
except ImportError:
    print("langgraph_checkpoint not importable")

try:
    from langchain.agents import AgentExecutor, create_react_agent

    print("create_react_agent found in langchain.agents")
except ImportError:
    print("create_react_agent NOT found in langchain.agents")

try:
    from langchain_core.tools import Tool

    print("Tool found in langchain_core.tools")
except ImportError:
    print("Tool NOT found in langchain_core.tools")

try:
    from langchain.tools import Tool

    print("Tool found in langchain.tools")
except ImportError:
    print("Tool NOT found in langchain.tools")
