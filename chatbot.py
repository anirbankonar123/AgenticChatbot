import streamlit as st
from langchain_core.prompts.prompt import PromptTemplate
from multi_doc_agent import MultiDocAgent
from langchain_openai import ChatOpenAI
from llama_index.llms.mistralai import MistralAI
from image_extractor import ImageExtractor


@st.cache_resource
def get_agent():
    multiDocAgent=MultiDocAgent()
    agent=multiDocAgent.get_agent("data")
    return agent

@st.cache_resource
def get_imageExtractor():
    imageExt = ImageExtractor()
    return imageExt

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5,top_p=0.5,max_tokens=4096,presence_penalty=1.0)
#llm = MistralAI(model="mistral-large-latest") #the LLM can be switched to MistralAI, if using MistralAI, the cmd is llm.complete
agent = get_agent()
img_ext = get_imageExtractor()

def get_best_answer(query,msg,msg1,msg2):
    prompt_template = PromptTemplate.from_template(""" \n
                                       You are a helpful assistant to answer scientific questions based on the context provided
                                       Do not use external information, select the best answer based on the query and the three responses provided 
                                       Only give the response value \n
                                       query : {query} \n
                                       response1 : {msg} \n
                                       response2 : {msg1} \n
                                       response3 : {msg2} \n
                                       AI :         
                                       """)
    prompt = prompt_template.format(query=query, msg=msg, msg1=msg1, msg2=msg2)
    response = llm.predict(prompt)
    return response

def get_summarized_context(chat_history):
    print("ENTERING SUMMARIZATION CALL")
    prompt_template = PromptTemplate.from_template(""" \n
                           You are a helpful assistant to answer scientific questions based on the context provided
                           Summarize the current conversation within 50000 tokens, and return it \n
                           Current conversation : {chat_history} \n
                           AI :         
                           """)
    prompt = prompt_template.format(chat_history=chat_history)
    response = llm.predict(prompt)
    return response
def invoke_agent(query,agent):
    print("QUERY:"+query)
    context=""
    response = agent.query(query)
    for node in response.source_nodes:
        context+=node.get_text()
    with open("output.txt", "a") as f:
        for node in response.source_nodes:
            f.writelines(str(node.metadata))
    f.close()
    return response,context

st.title("A Streamlit powered Chatbot with Agentic RAG and Image information extraction")
label = "Type exit to discontinue a conversation and start a new one"

s = f"<p style='font-size:14px;'>{label}</p>"
st.markdown(s, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "ConvCtr" not in st.session_state:
    st.session_state["ConvCtr"] = 0

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "query" not in st.session_state:
    st.session_state["query"] = ""

if query := st.chat_input():
    response=""
    if (query=="exit"):
        st.session_state["chat_history"]=[]
        st.session_state["ConvCtr"]=0
        st.session_state["query"] = ""
    else:
        st.session_state["ConvCtr"] += 1
        st.session_state["query"]=query

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
        try:
            response=llm.predict(prompt)
        except Exception:
            chat_history = get_summarized_context(str(chat_history))
            prompt_template = PromptTemplate.from_template(""" \n
                                   You are a helpful assistant to answer scientific questions based on the context provided
                                   Do not use external information, answer only based on the Current conversation \n
                                   Current conversation : {chat_history} \n
                                   Human : {input} \n
                                   AI :         
                                   """)
            prompt = prompt_template.format(chat_history=chat_history, input=query)
            response = llm.predict(prompt)
        print("RESPONSE from LLM:"+str(response))
        st.session_state["chat_history"].append((prompt, str(response)))

    if (st.session_state["ConvCtr"] >= 1):
        st.chat_message("user").write(query)
        msg = str(response)
        st.chat_message("assistant").write(msg)

if st.button("Search Image"):

    if (st.session_state["ConvCtr"]==1):
        query = st.session_state["query"]
    else:
        query = str(st.session_state["chat_history"]) + """ \n query:""" + st.session_state["query"]

    msg = img_ext.get_response(query,"data_output")
    msg1 = img_ext.get_response(query,"data_output1")
    msg2 = img_ext.get_response(query, "data_output2")
    response = get_best_answer(query, msg, msg1, msg2)
    print(response)
    if (response.lower()=='response1'):
        msg = msg
    elif (response.lower()=='response2'):
        msg = msg1
    elif (response.lower()=='response3'):
        msg = msg2
    st.chat_message("assistant").write(st.session_state["query"])
    st.chat_message("user").write(msg)
    st.session_state["chat_history"].append((query, str(msg)))




