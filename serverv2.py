import os
import openai
import json
import logging
import pinecone
# Added feature, this has been added to make the bot conversational.
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
 # Added flask_cors for allow cross origin.
from flask_cors import cross_origin
 # Added writer class from csv module
from csv import writer
import pdb
import re
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

######################
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import re
import warnings
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain,RetrievalQAWithSourcesChain
from langchain.schema import BaseOutputParser
########################

def extract_details_from_response(text: str) -> dict:
    match = re.search(r'"Answer":\s*(.*?)(?=,\s*"\w+":)', text, re.DOTALL)
    if match:
        Answer = match.group(1).strip()
        print(f'Answer: {Answer}')
    else:
        print("No match found for Answer")


    match = re.search(r'"Question":\s*(.*?)(?=,\s*"\w+":)', text, re.DOTALL)
    if match:
        Question = match.group(1).strip()
        print(f'Question: {Question}')
    else:
        print("No match found for Question")


    match = re.search(r'"Suicidal":\s*(.*?)(?=,\s*"\w+":)', text)
    if match:
        Suicidal = match.group(1).strip()
        print(f'Suicidal: {Suicidal}')
    else:
        print("No match found for Suicidal")

    match = re.search(r'"Followup":\s*(.*?)(?=,\s*"\w+":)', text)
    if match:
        Followup = match.group(1).strip()
        print(f'Followup: {Followup}')
    else:
        Followup = ""
        print("No match found for Followup")

    match = re.search(r'"MentalHealth":\s*(.*?)(?=,\s*"\w+":)', text)
    if match:
        MentalHealth = match.group(1).strip()
        print(f'MentalHealth: {MentalHealth}')
    else:
        print("No match found for MentalHealth")

    return {
        'Question': Question,
        'Answer': Answer,
        'Suicidal': Suicidal,
        'MentalHealth': MentalHealth,
        'Followup': Followup
    }


def extract_details_from_response2(text: str) -> dict:
    
    
    Suicidal = re.search(r"Suicidal: (.+)", text).group(1)
    Question = re.search(r"Question: (.+)", text).group(1)
    MentalHealth = re.search(r"MentalHealth: (.+)", text).group(1)
    Followup = re.search(r"Followup: (.+)", text).group(1)
    Answer = re.search(r"Answer: (.+)", text).group(1)
    return {
        'Question': Question,
        'Answer': Answer,
        'Suicidal': Suicidal,
        'MentalHealth': MentalHealth,
        'Followup': Followup
    }


template1="""
Role:Your role is that of  AI assistant for mental health.\
Your goal is to help users using cognitive behavioral therapy\
. You should be knowledgeable about all aspects of this technique\
and be able to provide clear and concise answers to users’ questions.\

Instructions while answering the question:\
1. Print the answer in bullets\
2. Answer the questions truthfully.\
3. If you are asked a question unrelated to mental health, do not answer. \
Instead, say "Hmm, I'm not sure I can answer that question."\

    
Identify the following items from the last user message and output as JSON List\
- Question , summarised in 500 characters
- Answer    
- Is the person distressed or showing suicidal tendencies (True or False).
- Please classify if he has symptoms related to mental health (True or False).
- Generate 1 followup question
Format the above as a JSON list and give the answer with the following keys \
"Question", "Answer", "Suicidal", "MentalHealth", "Followup", "Dummy".\
    
Remember to output in the above mention json format.if the patient is trying to converse with hi,hello
still make the MentalHealth key as true .\   

"""

templatenew="""
Role:Your role is that of  AI assistant for mental health.\
Your goal is to help users using cognitive behavioral therapy\
. You should be knowledgeable about all aspects of this technique\
and be able to provide clear and concise answers to users’ questions and using chat history.\


Response must be in a JSON formatted dictionary with the following keys \
"Question", "Answer", "Suicidal".\
with each key having the following definition:\
question - has the users question\
answer - has the response to the users question as a mental health specialist \
sucicdal - if the person is showing suicidal tendencies (true/false)\

Remember to output in the above mention json format.\   
{chat_history}
### Input: {question}
### Response:
""".strip()

template11= """
Role:Your role is that of  AI assistant for mental health.\
You should be knowledgeable about all aspects of this technique\
and be able to provide clear and concise answers to users’ questions using cognitive behavioral therapy and chat history .\



Response must be in a JSON formatted dictionary with the following keys \
"Question", "Answer", "Suicidal".\
with each key having the following definition:\
question - has the users question\
answer - has the response to the users question as a mental health specialist \
sucicdal - if the person is showing suicidal tendencies (true/false)\

{chat_history}
### Input: {question}
### Response:
""".strip()



template_2 = """
Role:Your role is that of  AI assistant for mental health.\
Your goal is to help users using cognitive behavioral therapy\
. You should be knowledgeable about all aspects of this technique\
and be able to provide clear and concise answers to users’ questions.\

Instructions while answering the question:\
1. Print the answer in bullets\
2.
   
   
Identify the following items from the last user message\
- Question , summarised in 500 characters
- Answer
- Sentiment: (positive or negative)\
- Is the person asking the question expressing anger? (true or false)\
- Is the question relatd to mental health( True or False).\
- Does the person need immediate help ( True or False).\
- Does he show symptoms of personaity disorder (True or False)
- if the immediate help = true, please ask followup question on location etc.  
Format the above as a JSON list and give the answer with the following keys "Question",
"Answer","Sentiment","Anger","Mental Health","Immediate Help", "Personality Disorder","Location"\
"""


template = """
Role:Your role is that of a mental health chatbot for advising patients on mental health.\

As a mental health chatbot, my goal is to provide support and resources to \
individuals who may be struggling with their mental health. \
I use a variety of techniques and methodologies to help users \
manage their mental health, including Cognitive Behavioral Therapy (CBT) [^1^][1].\
 CBT is a form of talking therapy designed to manage mental health states by\
rearranging the way the patient perceives it, i.e., making negative thoughts\
positive [^1^][1].

I am here to listen and offer guidance, whether you are dealing with stress, anxiety, depression, or any other mental health concern. I am not a replacement for professional help, but I can provide a safe and confidential space for you to share your thoughts and feelings. How can I help you today?

Answer the questions truthfully.

Instructions while answering the question:\
1. Answer only if the question is related to mental health.\
2. Print the answer in bullets\
         
Question :{question}\
Context: {context}\
    
Identify the following items from the Question :{question}
- Is the person distressed or showing suicidal tendencies (True or False).
- Please classify if he has symptoms related to mental health (True or False).
- Generate one  followup question

Format the above as a JSON Dictionary and give the answer with the following keys 
"Question", "Answer","Suicidal", "MentalHealth", "Followup","Dummy"\

"""






flagError = []

QA_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

# Initialize you log configuration using the base class
logging.basicConfig(level = logging.INFO)
# Retrieve the logger instance
logger = logging.getLogger()

def readconfig(keys):
    try:
       # pdb.set_trace()
        with open("config.json", "r") as jsonfile:
            data = json.load(jsonfile) # Reading the file
            jsonfile.close()
            return  data[keys]
    except:
        flagError.append("Error while fetching reading from config.json")
        logger.info("Error while reading config.json")
        return ""


if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
    os.environ["OPENAI_API_KEY"] =readconfig("OPENAI_API_KEY")
    logger.info("Config: OPENAI_API_KEY")


os.environ["OPENAI_API_TYPE"] =""


deployment_name='workstationada' 


INDEX_NAME = "demoindexaditya"

embeddings = OpenAIEmbeddings(
deployment=deployment_name,
model="text-embedding-ada-002"
)

# Reading Pinecone Keys From Config.json
if os.getenv("PINECONE_API_KEY") is None or os.getenv("PINECONE_API_KEY") == "":
    os.environ["PINECONE_API_KEY"] =readconfig("PINECONE_API_KEY")
    
    logger.info("Config: PINECONE_API_KEY")

if os.getenv("PINECONE_ENVIRONMENT") is None or os.getenv("PINECONE_ENVIRONMENT") == "":
    os.environ["PINECONE_ENVIRONMENT"] =readconfig("PINECONE_ENVIRONMENT")

    logger.info("Config: PINECONE_ENVIRONMENT")

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENVIRONMENT")  # next to api key in console
)
os.environ["OPENAI_API_KEY"] = readconfig("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(
#deployment=deployment_name,
model="text-embedding-ada-002"
)
# Check if the index already exists
if INDEX_NAME in pinecone.list_indexes():
    # If the index already exists, use it
    index = pinecone.Index(INDEX_NAME)
    docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    logger.info("Initialize pinecone index")
else:
    flagError.append("Pinecone the index doesn't exist")
    logger.info("Pinecone the index doesn't exist")


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
llm = OpenAI(
        
        temperature=0.0
)
logger.info("Initialize llm OpenAI")


retriever = docsearch.as_retriever()



chain_type_kwargs = {"prompt": QA_PROMPT}
chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                              #memory=memory
                                              chain_type="stuff",
                                              retriever=retriever,
                                              return_source_documents=True,
                                             # get_chat_history=get_chat_history(history_text),
                                              verbose=False,
                                              combine_docs_chain_kwargs=chain_type_kwargs)


similarity_chunks = 3

from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()
logger.info("Initialize ChatMessageHistory")

from flask import Flask,request,jsonify
app = Flask(__name__)



def inital():
    llm = OpenAI( openai_api_key="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",temperature=0)
    templatenew1="""
    Role:Your role is that of  AI assistant for mental health.\
    Your goal is to help users using cognitive behavioral therapy\
    . You should be knowledgeable about all aspects of this technique\
    and be able to provide clear and concise answers to users’ questions.\
    

    Response must be in a JSON formatted dictionary with the following keys \
    "Question", "Answer", "Suicidal", "Followup".\
    with each key having the following definition:\
    question - has the users question\
    answer - has the response to the users question as a mental health specialist \
    suicidal - if the person is showing suicidal tendencies (true/false)\
    follow up - Generate 1 followup question
    
    Use the following chat history provided in brackets aid in the answer to the question\
    {chat_history}
    ### Question: {input}
    ### Response: 
    """.strip()
    templatenew=""" Your role is that of an therapist for mental health .\
    you must speak eloquently and positively to the user in a conversational manner.
    The response of the chatbot should be happy,must make the client feel that the llm is his friend.\
    The responses should be conversational and chatty.The tone must make the user get the impression\
    that he can confide in u and talk freely.
    Use the given question of the user and chat history to give conversational responses
    ###Chat history: {chat_history}
    ###Question: {input}
    Response must be in a JSON formatted dictionary with the following keys \
    "Question", "Answer", "Suicidal", "Followup".\
    with each key having the following definition:\
    question - has the users question\
    answer - has the response to the users question as a mental health specialist \
    suicidal - if the person is showing suicidal tendencies (true/false)\
    follow up - Generate 1 followup question
    ###Response:
    """.strip()
    prompt1 = PromptTemplate(input_variables=["chat_history", "input"], template=templatenew)
    prompt2 = PromptTemplate(input_variables=["chat_history", "input"], template=templatenew1)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="CLient",
    ai_prefix="AI"
    )

    return prompt1,prompt2,memory

@app.route('/Test')
@cross_origin()
def hello_world():
   return jsonify('Test BOT response')



# @app.route('/asknextquery', methods=['POST'])
# @cross_origin() 

# def asknextquery_post():
#     h2=llm_chain.predict(question="what is my name")
#     print(h2)


@app.route('/askquery', methods=['POST'])
@cross_origin() 

def askquery_post():
    # pdb.set_trace()
    result={}
    request_data = request.get_json()
    user_input = request_data["Question"]
   
    hello=chain.run(input=user_input)
    answer=json.loads(hello)
    print(hello)
    Assistant ={
                "Question": answer['Question'],
                "Answer": answer['Answer'],
                "Suicidal":answer['Suicidal'],
                "Followup":answer['Followup'],
                #"Chat_History":Chat_History
            }
    result = {
            "status" :1,
            "messages":"Succesfully",
            "data": Assistant
        }
    return jsonify(result)
    


@app.route('/flagquestion', methods=['GET'])
@cross_origin()
def flagQuestion():
    result ={}
    try:
        user_input=request.args["question"]
        if user_input == "":
            result = {
                "status" :0,
                "messages":"enter valid text."
            }
            return jsonify(result)
        List = [user_input]
        with open('Questions.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
        logger.info("flagQuestion?question="+user_input)
        result = {
                "status" :1,
                "messages":"Sucessfully"
            }
    except:
        result = {
                "status" :0,
                "messages":"error while flag question."
            }
    return jsonify(result)


if __name__ == '__main__':
    from waitress import serve
    prompt1,prompt2,memory=inital()
    prompt_infos = [
    {
        "name": "Mental_health",
        "description": "Good for answering questions when client asks mental_health specific",
        "prompt": prompt2,
    },
    {
        "name": "Friend",
        "description": "Good for answering questions when client is sad and asks conversational questions",
        "prompt": prompt1,
    },
]
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        #prompt_template = p_info["prompt_template"]
        prompt = p_info['prompt']
        chain = LLMChain(llm=llm, prompt=prompt,verbose=True,memory=memory)
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=llm, output_key="text")    
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
    
    serve(app, host="0.0.0.0", port=8088)

