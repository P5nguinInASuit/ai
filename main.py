from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent  # Modern version
import tools as tools_

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Gemini 1.5/2.0 is the current standard; 2.5 is not released yet
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

instructions = f"""
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools. 
Wrap the output in this format and provide no other text:
{parser.get_format_instructions()}"""

tools = [tools_.search_tool, tools_.wiki_tool, tools_.save_tool]

# create_agent in v0.3 returns a compiled graph/executor
agent_executor = create_agent(
    model=llm,  # Use 'model' instead of 'llm'
    system_prompt=instructions,  # Use 'system_prompt' instead of 'prompt'
    tools=tools
)

query = input("What can I help you with? ")

# Invoke directly
raw_response = agent_executor.invoke({"messages": [("human", query)]})


try:
    # 1. Get the content from the last message
    final_content = raw_response["messages"][-1].content
    
    # 2. If it's a list (like in your error), extract the text part
    if isinstance(final_content, list):
        # Find the dictionary that has the 'text' key
        final_text = next((item['text'] for item in final_content if 'text' in item), "")
    else:
        final_text = final_content

    # 3. Clean up potential markdown code blocks (```json ... ```)
    final_text = final_text.replace("```json", "").replace("```", "").strip()

    # 4. Parse the cleaned string
    structured_response = parser.parse(final_text)
    
    print("\n--- Research Results ---")
    print(f"Topic: {structured_response.topic}")
    print(f"Summary: {structured_response.summary}")
    print(f"Sources: {', '.join(structured_response.sources)}")
    print(f"Tools Used: {', '.join(structured_response.tools_used)}")

except Exception as e:
    print(f"Parsing error: {e}")
    print("Raw text attempt:", final_text if 'final_text' in locals() else "No text found")
