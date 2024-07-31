import streamlit as st
from langchain_core.prompts.prompt import PromptTemplate
from multi_doc_class import MultiDocAgent
from langchain_openai import ChatOpenAI
from llama_index.llms.mistralai import MistralAI

#Vector search output
@st.cache_resource
def get_agent():
    multiDocAgent=MultiDocAgent()
    agent=multiDocAgent.get_agent("data")
    return agent

#llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5,top_p=0.5,max_tokens=4096,presence_penalty=1.0)
llm = MistralAI(model="mistral-large-latest")
agent = get_agent()
def invoke_agent(query,agent):
    print("QUERY:"+query)
    context=""
    response = agent.query(query)
    for node in response.source_nodes:
        context+=node.get_text()
    return response,context

st.title("A Streamlit powered Multi-document Chatbot with Agentic RAG with Llamaindex supported by LLM")

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
        st.session_state["ConvCtr"]=0
    else:
        st.session_state["ConvCtr"] += 1

    if (st.session_state["ConvCtr"]==1):
        #invoke agentic RAG
        response,context = invoke_agent(query,agent)
        print("RESPONSE from Llama index Agent:" + str(response))

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
        #Get context from chat history
        chat_history = st.session_state["chat_history"]

        prompt_template = PromptTemplate.from_template(""" \n
                       You are a helpful assistant to answer scientific questions based on the context provided
                       Do not use external information, answer only based on the Current conversation \n
                       Current conversation : {chat_history} \n
                       Human : {input} \n
                       AI :         
                       """)
        prompt = prompt_template.format(chat_history=chat_history, input=query)
        response=llm.complete(prompt)
        print("RESPONSE from LLM:"+str(response))
        st.session_state["chat_history"].append((prompt, str(response)))

    if (st.session_state["ConvCtr"] >= 1):
        st.chat_message("user").write(query)
        msg = str(response)
        st.chat_message("assistant").write(msg)
