#Retail Query AI

Overview

Retail Query AI is a Streamlit application designed to allow users to query a MySQL database using natural language. The application utilizes a large language model (LLM) to interpret user queries, convert them into SQL, and return the results from the database.

Features
Natural Language Processing: Users can enter queries in simple language, which are then converted into SQL queries.
Database Connection: Connects to a MySQL database using pymysql.
Query Execution: Executes the generated SQL queries and returns the results.
Streamlit Interface: Provides a user-friendly web interface for interaction.
Requirements
To run this project, you need to have the following installed:

Python 3.x
Streamlit
LangChain
PyMySQL
Hugging Face Transformers
Chroma
Llama 3.1 instance from Groq
Any additional dependencies specified in requirements.txt
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/retail-query-ai.git
cd retail-query-ai
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your database:

Ensure you have a MySQL database named estore_tshirts.
Update the database connection parameters in connect_db() function if necessary.
Set your environment variables:

Set your Groq API key:
bash
Copy code
export groq_api_key='your_groq_api_key'  # On Windows use `set`
Usage
Run the Streamlit app:

bash
Copy code
streamlit run main.py
Open your web browser and navigate to http://localhost:8501.

Enter your query in the input field and press Enter. The app will generate the corresponding SQL query and display the results.

Code Structure
main.py: The main application file containing the Streamlit interface and logic for handling user input.
langchain_helper.py: Helper functions for connecting to the database, creating the LLM model, finding SQL queries, and converting results.
requirements.txt: Lists the required packages for the project.
Functions
connect_db(): Connects to the MySQL database using specified credentials.
create_llm_model(): Creates and returns a language model instance for processing queries.
find_sql_query(response): Extracts the SQL query from the response generated by the LLM.
convert_number(result): Cleans and formats the query result before display.
Contributing
If you'd like to contribute to this project, feel free to submit a pull request. Any improvements or bug fixes are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Streamlit
LangChain
PyMySQL
Hugging Face Transformers
