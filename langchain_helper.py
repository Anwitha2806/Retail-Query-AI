import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from examples import examples

load_dotenv()

def find_sql_query(response):
    """
    Extracts the SQL query from the given response string.
    
    The function looks for the SQL query that starts with the keyword 'SELECT'
    and extracts it from the response. It also performs some cleaning on the
    response to ensure proper extraction.

    Args:
        response (str): The response string containing the SQL query.

    Returns:
        sql_query (str): The extracted SQL query, or None if not found.
    """
    
    # Step 1: Clean the response by replacing newlines with spaces and removing code block indicators
    cleaned_response = response.replace("\n", " ").replace("```", "").strip()
    
    # Step 2: Find the starting index of the SQL query by locating the keyword 'SELECT'
    select_start = cleaned_response.find("SELECT")
    
    # Step 3: Check if 'SELECT' was found in the response
    if select_start == -1:
        # If 'SELECT' is not found, print an error message and return None
        print("Error: Could not find 'SELECT' in the response")
        return None
    
    # Step 4: Extract the SQL query starting from the found 'SELECT' index to the end of the cleaned response
    sql_query = cleaned_response[select_start:].strip()
    
    # Step 5: Print the extracted SQL query for debugging or verification
    print("Extracted SQL Query:", sql_query)
    
    # Step 6: Return the extracted SQL query
    return sql_query


   

def connect_db():
    """
    Connects to the MySQL database using provided credentials and returns the connection object.

    Returns:
        db (SQLDatabase): A SQLDatabase object for interacting with the connected MySQL database.
    """
    # Database credentials and connection parameters
    db_user = "root"           # The username for the MySQL database
    db_password = "123456"      # The password for the MySQL database
    db_host = "localhost"       # The hostname or IP address where the database is hosted
    db_name = "estore_tshirts"  # The name of the specific database to connect to

    # Create the SQLDatabase object by constructing the MySQL connection URI
    # The URI format is 'mysql+pymysql://username:password@host/database_name'
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3  # Optional: Set the number of rows to display in table info
    )

    # Return the SQLDatabase object, which will be used to run SQL queries
    return db

def create_llm_model():
    """
    Creates and configures the LLM model using Groq API, Hugging Face embeddings, and a semantic similarity-based
    example selector for few-shot prompting. Returns a chain that can generate SQL queries based on user input.
    
    Returns:
        chain (SQLQueryChain): A Langchain SQL query chain object that integrates LLM and database connection.
    """
    
    # Step 1: Get Groq API key from environment variables
    groq_api_key = os.environ["groq_api_key"]  # API key for authentication with the Groq API

    # Step 2: Initialize the LLM (Large Language Model) using the ChatGroq class
    # The 'llama-3.1-70b-versatile' model is chosen for its versatility in different tasks
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    
    # Step 3: Initialize Hugging Face sentence embeddings model for transforming textual inputs into vectors
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Step 4: Convert example texts into vectorized format for storing in the vector database (Chroma)
    # 'examples' is a list of training examples, each containing a set of values
    to_vectorize = [" ".join(example.values()) for example in examples]  # Flatten the example data into text
    
    # Step 5: Initialize a Chroma vector store for semantic similarity search
    # This is used to store the vectorized examples, with metadata for each example
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples, persist_directory="./chroma_store")

    # Step 6: Create a Semantic Similarity Example Selector
    # This selector finds the most similar examples (top k=2) from the vector store based on the query input
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,  # Use the Chroma vector store
        k=2,  # Number of closest examples to retrieve based on similarity
    )

    # Step 7: Define a template for how each example will be structured
    # This template includes the Question, SQLQuery, SQLResult, and Answer, which are key components in our chain
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],  # Variables for the prompt
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",  # Template structure
    )

    # Step 8: Create a Few-Shot Prompt Template
    # It uses the example selector to select the best-fitting examples from the vector store and applies them in the chain
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,  # The semantic similarity-based example selector
        example_prompt=example_prompt,  # The structure of each example
        prefix=_mysql_prompt,  # Custom prefix (e.g., prompt to guide the LLM in SQL query generation)
        suffix=PROMPT_SUFFIX,  # Custom suffix (e.g., prompt completion variables)
        input_variables=["input", "table_info", "top_k"],  # Input variables for the prefix and suffix prompts
    )

    # Step 9: Connect to the MySQL database
    db = connect_db()  # Calls the 'connect_db' function to get a SQLDatabase object

    # Step 10: Create the SQL query chain
    # The chain combines the LLM, database connection, and the few-shot prompt for generating SQL queries
    chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)

    # Step 11: Return the chain object for querying the database using natural language inputs
    return chain
    

def convert_number(result):
    """
    Extracts and cleans the numerical data from the result of a SQL query.
    
    The result is typically in the form of a tuple inside a string (e.g., "(123.45,)", or "('123.45',)")
    This function removes any unwanted characters (like parentheses or single quotes) and extracts the 
    number part as a string.
    
    Args:
        result (str): The result from the SQL query, usually formatted as a string containing a tuple.
    
    Returns:
        number_string (str): The cleaned numerical value as a string.
    """
    # Step 1: Remove the outer brackets and parentheses from the result string
    # This extracts the content between "('" and "')", which removes the tuple's parentheses and quotes
    cleaned_result = result[result.find("('") + 2:result.find("')")]
    # Step 2: Split the cleaned result by comma (in case there are multiple values) and take the first part
    # This step handles cases where the result contains multiple values, but only the first value is needed
    number_string = cleaned_result.strip("()").split(",")[0]
    # Step 3: Return the cleaned number as a string
    return number_string



