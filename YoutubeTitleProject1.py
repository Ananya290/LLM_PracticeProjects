import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory




os.environ['Google_API_KEY']="AaaIzastSyBB2_xvoJ5F7t-DHHV-s1ryJRC-W8fy3ho"


#app framework

st.title(" Vaishnavi's Bot  for  youtube Script title")

#prompt_template
title_template=PromptTemplate(
    input_variables=['topic'],
    template='Write me Youtube video title about {topic}')

script_template=PromptTemplate(
    input_variables=['title'],
    template='Write me Youtube video script based on this TITLE {title}'

)


prompt = st.text_input("Write  your doubt here")

#LLM
llm = GoogleGenerativeAI(model="models/text-bison-001",
                         google_api_key='AaaIzaSsstyBB2_xvoJ5F7t-DHHV-s1tuhryJRC-W8fy3ho',
                         temperature=0.9)
#Memory
memory=ConversationBufferMemory(input_key='topic', memory_key='chat_history')


#chain
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script',memory=memory)

sequential_chain=SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
                                 output_variables=['title', 'script'], verbose=True)
#Show stuff to the screen if there is a prompt
if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)
