import streamlit as st
from streamlit_chat import message
from langchain import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain import LLMChain
import openai
import pinecone

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
PINECONE_NAMESPACE = st.secrets["PINECONE_NAMESPACE"]
INITIAL_PROMPT = st.secrets["INITIAL_PROMPT"]
EMBEDDING_MODEL = 'text-embedding-ada-002'
index_name = PINECONE_INDEX_NAME

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV  # may be different, check at app.pinecone.io
)
# connect to index
index = pinecone.Index(index_name)


@st.cache_resource
def LLM_chain_response():
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="Answer the question based on the context below. If you cannot answer based on the "
                 "context, say I don't know. Use Markdown and text formatting to format your answer. "
                 "\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    )

    llm = OpenAI(
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        model_name="text-davinci-003",
        max_tokens=256
    )

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=256)
    )
    return chatgpt_chain


# Define the retrieve function
# @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3), retry_error_callback=retry_if_not_exception_type(TypeError))
def retrieve(query):
    # retrieve from Pinecone
    res = openai.Embedding.create(input=[query], model=EMBEDDING_MODEL)
    xq = res['data'][0]['embedding']
    # sq = splade(query)

    # get relevant contexts
    # pinecone_res = index.query(xq, top_k=4, include_metadata=True, sparse_vector=sq)
    pinecone_res = index.query(xq, top_k=4, include_metadata=True, namespace=PINECONE_NAMESPACE)
    contexts = [x['metadata']['chunk_text'] for x in pinecone_res['matches']]
    # contexts = [x['metadata']['chunk_text'] for x in pinecone_res['matches'] if x['score'] > 0.8]
    # urls = [x['metadata']['url'] for x in pinecone_res['matches']]
    # print([score['score'] for score in pinecone_res['matches']])

    # temporary hard code but this should be a metadata attribute
    #source = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000008670/8f968664-2bd7-4993-ac87-b731040eb43f.pdf"

    # pinecone_contexts = (
    #     "\n\n---\n\n".join(contexts) + f"\n\n source: {source}"
    # )
    pinecone_contexts = (
        "\n\n---\n\n".join(contexts)
    )
    #print(pinecone_contexts)
    return pinecone_contexts


# From here down is all the StreamLit UI.
image = open(f"./app/citi.png", "rb").read()

st.image(image)

st.write("### Pinecone - ChatGPT Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", f"{INITIAL_PROMPT}", key="input")
    return input_text


# Main function for the Streamlit app
def main():
    chatgpt_chain = LLM_chain_response()
    user_input = get_text()
    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            pinecone_contexts = retrieve(query)
            output = chatgpt_chain.predict(input=query + '\nContext: ' + pinecone_contexts)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")


if __name__ == "__main__":
    main()
