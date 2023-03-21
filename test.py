import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os

from llama_index import LLMPredictor, QuestionAnswerPrompt, GPTPineconeIndex, SimpleDirectoryReader

from langchain.agents import ConversationalAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

import pinecone


os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
pinecone_api_key = 'PINECONE_API_KEY'

user_message = "Recommend some songs for the first dance."


pinecone.init(api_key=pinecone_api_key, environment="us-central1-gcp")
pinecone.create_index("llama-test", dimension=1536, metric="euclidean", pod_type="p1")
index = pinecone.Index("llama-test")

# # How Index was created
documents = SimpleDirectoryReader('data').load_data()
INDEX = GPTPineconeIndex(documents, pinecone_index=index, chunk_size_limit=512)

## How I load index after first run
# INDEX = GPTPineconeIndex('', pinecone_index=index, chunk_size_limit=512)

## Define custom Question/Answer prompt for GPT 3.5 Turob (ChatGPT)
QA_TURBO_TEMPLATE_MSG = [
    SystemMessagePromptTemplate.from_template(
        """
        Context information is below. \n
        ---------------------\n
        {context_str}
        \n---------------------\n
        Given this information, please answer the question and spell check everything especially the tradition or custom names in context and replace them with correct one.
        """
    ),
    HumanMessagePromptTemplate.from_template("What is the average age difference between bride and groom?"),
    AIMessagePromptTemplate.from_template("Approximately two years. According to the 2016 Real Weddings Study, the average age of to-be-weds in the US was 29 for the bride and 31 for the groom. This indicates that couples are taking their time entering life stages, such as moving in with their partner premarriage or finishing up a master's degree before getting married. This is also reflected in the birth rates, which increased in women aged 30 to 34, 35 to 39 and 40 to 44, and decreased slightly in the age groups 20 to 24 and 25 to 29."),
    HumanMessagePromptTemplate.from_template("{query_str} \n")
]

QA_TURBO_TEMPLATE_LC = ChatPromptTemplate.from_messages(QA_TURBO_TEMPLATE_MSG)
QA_TURBO_TEMPLATE = QuestionAnswerPrompt.from_langchain_prompt(QA_TURBO_TEMPLATE_LC)

QA_PROMPT = QA_TURBO_TEMPLATE

## Set number of output tokens
NUM_OUTPUT = 600
## Define model and it's parameters
LLM_PREDICTOR = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=NUM_OUTPUT))
## Set how many matches are found
SIMILARITY_TOP_K = 3

print(INDEX.prompt_helper.chunk_size_limit)
response = INDEX.query(user_message, llm_predictor=LLM_PREDICTOR, text_qa_template=QA_PROMPT, similarity_top_k=SIMILARITY_TOP_K, response_mode="compact", verbose=True)
print(len(response.source_nodes[0].source_text.split(" ")))

print(response)