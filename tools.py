from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime
from langchain_core.tools import tool # Modern decorator

# 1. Custom Tool using the @tool decorator
@tool
def save_tool(data: str, filename: str = "research_output.txt") -> str:
    """Saves structured research data to a local text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

# 2. Search Tool
search = DuckDuckGoSearchRun()
# Wrap it so you can customize the name/description for the agent
@tool
def search_tool(query: str) -> str:
    """Search the web for real-time information and current events."""
    return search.run(query)

# 3. Wikipedia Tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper,
    name="wikipedia",
    description="Look up facts and summaries on Wikipedia."
)
