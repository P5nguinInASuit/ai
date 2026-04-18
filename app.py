import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from main import llm, tools, instructions, parser # Importing your existing setup

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# --- 1. MEMORY SETUP ---
# We use cache_resource so the MemorySaver object stays alive in the background
@st.cache_resource
def get_memory():
    return MemorySaver()

# Initialize memory and the agent
memory = get_memory()
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=instructions,
    checkpointer=memory
)

# --- 2. SESSION MANAGEMENT ---
# Generate a unique thread ID for this specific user session if it doesn't exist
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(hash(st.session_state)) 

# --- 3. SIDEBAR & CLEAR BUTTON ---
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat History"):
        # We clear the history by changing the thread_id
        # This makes the agent think it's a brand new conversation
        st.session_state.thread_id = str(hash(st.session_state.thread_id + "new"))
        st.rerun()
    st.info(f"Current Thread: {st.session_state.thread_id}")

st.title("🔍 AI Research Assistant")

# --- 4. CHAT INTERFACE ---
query = st.chat_input("What would you like to research?")

if query:
    # Set the config with our session's thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Thinking..."):
        # Run agent with memory config
        raw_response = agent_executor.invoke(
            {"messages": [("human", query)]}, 
            config=config
        )

    # --- 5. DISPLAY ALL MESSAGES ---
    # The agent state now contains the full history
    for msg in raw_response["messages"]:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        
        elif msg.type == "ai" and msg.content:
            with st.chat_message("assistant"):
                content = msg.content
                # Fix for the 'list' content error we saw earlier
                if isinstance(content, list):
                    content = next((item['text'] for item in content if 'text' in item), "")
                
                # If it looks like JSON, try to parse it nicely
                if "{" in content and "}" in content:
                    try:
                        # Clean markdown wrap
                        clean_json = content.replace("```json", "").replace("```", "").strip()
                        res = parser.parse(clean_json)
                        st.subheader(res.topic)
                        st.write(res.summary)
                        st.caption(f"Sources: {', '.join(res.sources)}")
                    except:
                        st.write(content)
                else:
                    st.write(content)
