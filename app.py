# Install necessary libraries
# !pip install -q streamlit langchain pandas openai

import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import AzureChatOpenAI
import json

# Function to initialize the Azure OpenAI model
def initialize_model():
    try:
        # Reading Azure OpenAI endpoint and API key
        with open('resources/config-azure.json') as f:
            azure_configs = f.read()
        azure_creds = json.loads(azure_configs)

        model_name = azure_creds['MODEL_NAME']

        # Initialize the Azure OpenAI model
        llm = AzureChatOpenAI(
            azure_endpoint=azure_creds['AZURE_OPENAI_ENDPOINT'],
            api_key=azure_creds['AZURE_OPENAI_KEY'],
            api_version=azure_creds['API_VERSION'],
            deployment_name=model_name,
            temperature=0,
            openai_api_version=azure_creds['API_VERSION'],
        )
        return llm
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure 'config-azure.json' is in the 'resources' directory.")
        raise
    except json.JSONDecodeError:
        st.error("Error decoding the configuration file. Please check its structure.")
        raise
    except KeyError as e:
        st.error(f"Missing key in configuration file: {e}")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred while initializing the model: {e}")
        raise

# Function to create a DataFrame agent
def create_agent(df, llm):
    try:
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, 
                            allow_dangerous_code=True,
                            max_iterations=100,  # Increase max iterations as needed
        max_execution_time=60,
        handle_parsing_errors=True)
        return agent
    except Exception as e:
        st.error(f"An error occurred while creating the DataFrame agent: {e}")
        raise

# Streamlit application
def main():
    st.title("Data Insights Generator")
    st.write("Upload your dataset (CSV/Excel), and ask questions based on the data!")

    # File upload section
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return

            st.write("**Dataset Preview:**")
            st.dataframe(df.head())  # Display the first few rows of the dataset

            try:
                # Initialize the Azure OpenAI model
                llm = initialize_model()

                # Create a DataFrame agent
                agent = create_agent(df, llm)

                # Question prompt
                question = st.text_input("Ask a question about the dataset:")
                if st.button("Get Answer"):
                    if question.strip():
                        try:
                            answer = agent.run(question)
                            st.write("**Answer:**")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"An error occurred while generating the answer: {e}")
                    else:
                        st.warning("Please enter a valid question.")
            except Exception as e:
                st.error(f"An error occurred during model initialization or agent creation: {e}")
        except pd.errors.ParserError:
            st.error("Error reading the file. Please check the file format and try again.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload a dataset to proceed.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
