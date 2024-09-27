import streamlit as st
from langchain_helper import create_llm_model, connect_db, find_sql_query, convert_number

# Set up the app title and description
st.set_page_config(page_title="Retail Query AI", layout="wide")

# Add a title and description for the app
st.title("Retail Query AI: Query your database in simple language")
st.write("""
### Easily query your database using natural language. 
Simply enter your question, and we'll handle the rest, generating the SQL and returning the result!
""")

# Add an input field for the user query
question = st.text_input("Enter your query in simple language")

# Provide feedback if the user has not entered a query yet
if not question:
    st.info("Please enter a question to get started.")
else:
    # When the user submits a question
    if st.button("Submit Query"):
        # Add a spinner while waiting for the AI to process the query
        with st.spinner("Processing your query..."):
            try:
                # Invoke the LLM model to generate SQL
                chain = create_llm_model()
                response = chain.invoke({"question": question})
                
                # Extract the SQL query from the response
                sql_query = find_sql_query(response)
                st.write(f"Generated SQL Query:\n```{sql_query}```")
                
                # Connect to the database and execute the query
                db = connect_db()
                result = db.run(sql_query)
                
                # Display results
                if result:
                    st.success("Query executed successfully!")
                    st.write("### Query Results:")
                    st.write(convert_number(result))
                else:
                    st.warning("No results found for your query.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add some styling to the Streamlit app using CSS
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)
