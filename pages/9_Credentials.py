import requests

import streamlit as st

st.set_page_config(page_title='Credentials', page_icon='data/alkane_logo.png', layout="wide")


FASTAPI_ENDPOINT = "http://localhost:4557"

def main():
    st.title("Credentials")

    api_keys_set = requests.get(f"{FASTAPI_ENDPOINT}/are_api_keys_set").json()['message']
    
    if api_keys_set:
        st.success("Your API keys are already set.")
    
    st.markdown("\n")
    st.markdown("To update your credentials, ")
    st.markdown("Enter your OPENAI and PINECONE API Keys here:")

    openai_api_key = st.text_input("OPENAI API Key", type="password")
    pinecone_api_key = st.text_input("PINECONE API Key", type="password")
    if st.button("Save Credentials"):
        try:
            with st.spinner('Setting OpenAI API key ..'):
                openai_set_post = requests.post(f"{FASTAPI_ENDPOINT}/update_openai_api_key", params={"api_key": openai_api_key})
                openai_set = openai_set_post.json()['message']
            st.success(openai_set)
        except Exception as e:
            st.error(f"Error setting OpenAI API key: {str(e)}")
            st.error(f"{openai_set_post.status_code} - {openai_set_post.text}")

        try:
            with st.spinner('Setting Pinecone API key ..'):
                pinecone_set_post = requests.post(f"{FASTAPI_ENDPOINT}/update_pinecone_api_key", params={"api_key": pinecone_api_key})
                pinecone_set = pinecone_set_post.json()['message']
            st.success(pinecone_set)
        except Exception as e:
            st.error(f"Error setting Pinecone API key: {str(e)}")
            st.error(f"{pinecone_set_post.status_code} - {pinecone_set_post.text}")
            st.stop()

        try:
            api_keys_set_post = requests.get(f"{FASTAPI_ENDPOINT}/are_api_keys_set")
            api_keys_set = api_keys_set_post.json()['message']
            st.success("API keys set successfully.")
        except Exception as e:
            st.error(f"Error checking API keys: {str(e)}")
            st.error(f"{api_keys_set_post.status_code} - {api_keys_set_post.text}")
    st.stop()

if __name__ == "__main__":
    main()