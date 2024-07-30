import streamlit as st
from langchain_core.prompts.prompt import PromptTemplate
from multi_doc_class import MultiDocAgent
from langchain_openai import ChatOpenAI

#Vector search output
@st.cache_resource
def get_agent():
    multiDocAgent=MultiDocAgent()
    agent=multiDocAgent.get_agent("data")
    return agent

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5,top_p=0.5,max_tokens=4096,presence_penalty=1.0)
agent = get_agent()
def invoke_agent(query,agent):
    print("QUERY:"+query)
    context=""
    response = agent.query(query)
    for node in response.source_nodes:
        context+=node.get_text()
    return response,context

st.title("A Streamlit powered Multi-document Chatbot Demo using MistralAI, OpenAI and Agentic RAG with Llamaindex")

st.header("Type exit to discontinue a conversation and start a new one")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "ConvCtr" not in st.session_state:
    st.session_state["ConvCtr"] = 0

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if query := st.chat_input():
    response=""
    if (query=="exit"):
        st.session_state["chat_history"]=[]
        #st.chat_message("assistant").write("Goodbye!")
        st.session_state["ConvCtr"]=0
    else:
        st.session_state["ConvCtr"] += 1

    if (st.session_state["ConvCtr"]==1):
        #do vector search
        response,context = invoke_agent(query,agent)
        print("RESPONSE from Llama index:" + str(response))

        chat_history = st.session_state["chat_history"]
        prompt_template=PromptTemplate.from_template(""" \n
        Context : {context} \n
        Current conversation : {chat_history} \n
        Human : {input} \n
        AI :         
        """)
        prompt = prompt_template.format(chat_history=chat_history,input=query,context=context)
        st.session_state["chat_history"].append((prompt,str(response)))

    elif(st.session_state["ConvCtr"]>1):
        chat_history = st.session_state["chat_history"]

        prompt_template = PromptTemplate.from_template(""" \n
                       You are a helpful assistant to answer scientific questions based on the context provided
                       Do not use external information, answer only based on the Current conversation \n
                       Current conversation : {chat_history} \n
                       Human : {input} \n
                       AI :         
                       """)
        prompt = prompt_template.format(chat_history=chat_history, input=query)
        response=llm.predict(text=prompt)
        print("RESPONSE from OPENAI:"+response)
        st.session_state["chat_history"].append((prompt, response))

    if (st.session_state["ConvCtr"] >= 1):
        st.chat_message("user").write(query)
        msg = str(response)
        st.chat_message("assistant").write(msg)
