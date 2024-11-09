# Common imports
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import json
import lolviz
import requests
# Import the key CrewAI classes
import streamlit as st
import tiktoken
from helper_functions.utility import check_password  

#set up for langchain
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


#setup for nativeRag
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader



# embedding model that we will use for the session
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

# llm to be used in RAG pipeplines in this notebook
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, seed=42)



if load_dotenv('.env'):
   # for local development
   OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
   OPENAI_KEY = st.secrets['OPENAI_API_KEY']

client = OpenAI(api_key=OPENAI_KEY)


#function for getting embeddings
def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]

#function for getting response

# This is the "Updated" helper function for calling LLM
def get_completion(prompt, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=256, n=1, json_output=False):
    if json_output == True:
      output_json_structure = {"type": "json_object"}
    else:
      output_json_structure = None

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create( #originally was openai.chat.completions
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        response_format=output_json_structure,
    )
    return response.choices[0].message.content


#get completion by messages

# This a "modified" helper function that we will discuss in this session
# Note that this function directly take in "messages" as the parameter.
def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content

# This function is for calculating the tokens given the "message"
# ⚠️ This is simplified implementation that is good enough for a rough estimation

def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))

def count_tokens_from_message_rough(messages):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    value = ' '.join([x.get('content') for x in messages])
    return len(encoding.encode(value))


#create the prompt chain.
#load in the CV here
cv_content = "i am a data scientist"

job_posting_url= "https://www.mycareersfuture.gov.sg/job/information-technology/data-scientist-cpo-03dba75ab1fec49a3aac63d2c676949a?source=MCF&event=Search"

personal_profiler=f""" play the role of a personal profiler, your goal is to Conduct comprehensive research 
on job applicants to help them stand out in the job market.


The candidates resume is here.{cv_content}
    back story: As a Job Researcher, your expertise in navigating and extracting critical information 
    from job postings is unparalleled. Your skills help identify the necessary qualifications and skills
    sought by employers, forming the foundation for effective application tailoring.
    
    your task is to Compile a detailed personal and professional profile based on the current CV.

    your are expected to output a comprehensive profile document that includes skills, project experiences, contributions, interests, and communication style.
    """




resume_strategist=f'''
play the role of a Resume Strategist, your goal is to Discover the best ways to make a resume stand out in the job market.

your goal is to Discover the best ways to make a resume stand out in the job market.

backstory: With a strategic mind and an eye for detail, you excel at refining resumes to highlight the most relevant 
    skills and experiences, ensuring they resonate perfectly with the job's requirements.

your task: Using the profile and job requirements obtained from previous tasks, tailor the resume to 
    highlight the most relevant areas. Employ tools to adjust and enhance the resume content. Make sure 
    this is the best resume ever but don't make up any information. Update every section, including the initial summary, 
    work experience, skills, and education. All to better reflect the candidate's abilities and how it matches the job posting.

you are expected to output an updated resume that effectively highlights the candidate's qualifications and experiences relevant to the job.

'''

from helper_functions.utility import check_password  



# Check if the password is correct.  
if not check_password():  
    st.stop()

#Streamlit section
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Understanding the Closure of CPF Special Account"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Understanding the Closure of CPF Special Account")

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    
    st.toast(f"User Input Submitted - {user_prompt}")
    
    st.divider()


resume=st.file_uploader('upload resume here')

joblink=st.text_input('input job posting URL', placeholder="https://www.mycareersfuture.gov.sg/job/information-technology/data-scientist-cpo-03dba75ab1fec49a3aac63d2c676949a?source=MCF&event=Search" )

# job_description=''
# if not joblink:
#     st.write(f"Please enter job URL")
# else:
#     response = requests.get(joblink)
#     if response.status_code == 200:
#             job_description = response.text 
#             st.write(job_description) 
#     else:
#             st.write(f"Please enter job URL")




job_researcher=f"""
play the role of a Job Researcher, your goal is to perform thorough analysis on job postings to assist job applicants.

backstory: Equipped with analytical prowess, you dissect and synthesize information from diverse sources to craft 
    comprehensive personal and professional profiles, laying the groundwork for personalized resume enhancements...

    your task is to analyze the job posting URL provided (`{joblink}`) to extract key skills, experiences, 
    and qualifications required. Use the tools to gather content and identify and categorize the requirements.
you are expected to output a structured list of job requirements, including necessary skills, qualifications, and experiences.


"""


messages = [{"role": "user", "content": job_researcher}]
summary=get_completion_by_messages(messages)

st.write(summary)