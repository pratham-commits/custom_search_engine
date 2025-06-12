import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

##arxiv
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

##wiki
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

##duck duck go search
search=DuckDuckGoSearchRun(name="search")

st.title("Langchain - with search")

"""
In n this example , we are using 'StreamlitCallbackHandlder' to displat the thoughts
and the actions of the AI agent
"""

## sidebar for settings
st.sidebar.title("Settings")
api_key=os.getenv("GROQ_API_KEY")


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hey I am a chatbot who can search the web. How can I help you?"}
        
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

prompt = st.chat_input(placeholder="What is machine learning?")
if  prompt and api_key:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(groq_api_key=api_key,model_name="Llama-3.3-70B-Versatile",streaming=True)
    tools=[search,wiki,arxiv]
    
    search_agent=initialize_agent(
                                  tools,
                                  llm,
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                  handle_parsing_errors=True
                                  )
    
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
        
        
