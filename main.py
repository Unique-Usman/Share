from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import re
from pprint import pprint
from typing import List
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableSequence


DB_FAISS_PATH = 'models/vectorstore/db_faiss/'
DB_FAISS_PATH_PERSONNEL = 'models/vectorstore_personnel/db_faiss/'
DB_FAISS_PATH_OTHERS = "models/vectorestore_others/db_faiss/"
os.environ["TAVILY_API_KEY"] = "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"
local_llm = "llama3.1"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
db_personnel = FAISS.load_local(DB_FAISS_PATH_PERSONNEL, embeddings, allow_dangerous_deserialization=True)
db_others = FAISS.load_local(DB_FAISS_PATH_OTHERS, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
retriever_personnel = db_personnel.as_retriever(search_kwargs={'k': 1})
retriever_others = db_others.as_retriever(search_kwargs={'k': 1})
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Retriever Grader
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

llm = ChatOllama(model=local_llm, temperature=0)
# rag_chain  /
prompt = PromptTemplate(
    template = """You are SuperVanni, the Plaksha University assistant dedicated Question and Answering.
    Ensure to Add all the links such as website and email from the gotten context.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Question: {question} 
    Documents: {documents} 
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(model="llama3", verbose=True, temperature=0.5)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()


# Hallucination Grader  

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

# Answer Grader 

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an  others it involve other questions
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Give 'no' if the answer contains I don't know. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

llm = ChatOllama(model="llama3", temperature=0)
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing 
    user question to a vectorstore or others. Return vectorstore when question is about a course, professor, founder or a name. 
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use others. Give a binary choice 'others' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm | JsonOutputParser()

# SQL CHAIN ====================================================================================

llm = Ollama(model = "llama3.1")

# Load the database
def load_db():
    # Connect to the MySQL database
    db = SQLDatabase.from_uri("mysql+pymysql://root:M0h%40l%21%232024@localhost/faculty_members")
    return db

# Create a query generation chain
def chain_create():
    # API Key (consider moving to environment variables or a secure vault)
    api_key = os.getenv("GROQ_API_KEY", "")
    llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=api_key)
    
    # Define the prompt template with schema description and instructions
    prompt_template = PromptTemplate.from_template(
        """The database contains the following tables and columns:
        1. professors:
           - id (INT, PRIMARY KEY)
           - name (VARCHAR(100))
           - email (VARCHAR(255))
           - webpage (VARCHAR(255))
           The professors table contains the names, emails, and webpages of the professors along with their respective id's.

        2. expertise:
           - id (INT, PRIMARY KEY)
           - name (VARCHAR(100))
           The expertise table contains the area of expertise of the professors referenced from the professors table along with the expertise id's.

        3. professor_expertise:
           - professor_id (INT, FOREIGN KEY to professors.id)
           - expertise_id (INT, FOREIGN KEY to expertise.id)
           The professor_expertise table links the professors to their respective areas of expertise.

           **Instructions**:
        - To answer questions about professors and their expertise, first refer to the `professors` and `expertise` tables to find the names and IDs.
        - Use the `professor_expertise` table to match IDs and determine the relationship between professors and their expertise.
        - Generate SQL queries to retrieve names of professors along with their areas of expertise based on these relationships.
        - Ensure that the query matches keywords partially. Give the answer even if the keyword is in lowercase or uppercase.
        - Handle variations in keyword matching (e.g., different forms or related terms should be considered the same keyword).
        - Group the results so that each professor is listed only once with all their relevant expertise.
        - Include the professor's name, email, and webpage only oquery_gen"nce. followed by a list of their expertise combined.
        - Return the professor names along with their areas of expertise in the results.
        - Also Handle variations if the words are not completely matching, even if one word matches the answer then return the results, and 
        even if something is written in acronym form  still return the correct results.
        - Present the final answer in a well-structured and engaging manner, avoided repetitive listings. Provide the answer
        in a way that conveys the asked information and helps the user understand the context and avoid any hallucinated or fake data.
        - Be able to handle variations in questions asked, even if the professors name is in lower case, or even if the last name of the professor is not there, 
        still provide the answer according to the first name.
        
        User Question: {question}
        SQL Query: """
    )
    
    # Create the LLM chain for generating SQL queries
    sql_chain = prompt_template | llm 
    return sql_chain

# Process the user's question and retrieve the answer
def sql_infer(db, llm_chain, user_question):
    try:
        # Generate SQL query using the LLM
        response = llm_chain.invoke({"question": user_question})
        response_text = response.content.strip()
        
        # Extract the SQL query from the response
        sql_query = re.search(r"```sql\n(.*?)\n```", response_text, re.DOTALL)
        if sql_query:
            sql_query = sql_query.group(1).strip()
        else:
            raise ValueError("SQL query not found in the response.")
        
        # Debug: Print the generated SQL query
        print(f"Generated SQL Query: {sql_query}")
        
        # Check if sql_query is a valid string
        if not isinstance(sql_query, str) or not sql_query:
            raise ValueError("Generated SQL query is invalid or empty.")
        
        # Execute the SQL query
        result = db.run(sql_query)
        print(result)
        return result
    except Exception as e:
        return f"Error: in generating or executing sql_infer()ing SQL query: {e}"

# Example Usage
# Define a conditional edge to decide whether to continue or end the workflow for sql
def route_sql(state):
    """
    Route question to back to retrieve_sql or to grade_documents.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    messages = state["documents"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if last_message.page_content.startswith("Error:"):
        return "retrieve_sql"
    else:
        return "generate"
# Web Search 

# State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    
### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval

    documents = retriever_personnel.invoke(question)
    # Load the database
    return {"documents": documents, "question": question}

def retrieve_other(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE OTHERS---")
    question = state["question"]
    # Retrieval
    documents = retriever_others.invoke(question)
    # Load the database
    return {"documents": documents, "question": question}


def retrieve_sql(state):
    """
    Retrieve documents from mysql_database

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents from sql
    """
    print("---RETRIEVE SQL---")
    question = state["question"]
    documents = state["documents"]

    # Load the database
    db = load_db()
    
    # Create the SQL LLM chain
    llm_chain = chain_create()

    db = load_db()

    # Create the LLM chain
    llm_chain = chain_create()
    
    sql_docs = sql_infer(db, llm_chain, question)
    if sql_docs is not None:
        doc = Document(page_content=sql_docs, metadata={"source": "sql"})
        print(doc)
        documents.append(doc)

    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


### Conditional edge
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "others":
        print("---ROUTE QUESTION TO OTHER SEARCH---")
        return "retrieve_other"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("retrieve_sql", retrieve_sql)  # retrieve sql
workflow.add_node("retrieve_other", retrieve_other)  # retrieve sql
workflow.add_node("generate", generate)  # generatae

# Build the graph  

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve_other": "retrieve_other",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "retrieve_sql")
workflow.add_edge("retrieve_other", "generate")
workflow.add_conditional_edges(
    "retrieve_sql",
    route_sql,
    {
        "retrieve_sql": "retrieve_sql",
        "generate": "generate",
    }
)
workflow.add_edge("generate", END,)
def create_app():
    app = workflow.compile()
    return app
