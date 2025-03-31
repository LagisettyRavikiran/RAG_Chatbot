import streamlit as st
import fitz
from PIL import Image
import os
import io
import faiss
import numpy as np
import torch
import random
import yaml
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_core.documents import Document
import uuid
load_dotenv()
# Load config.yml
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

prompt_template = config.get("prompt_template", {})

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
os.makedirs("chatbot_pdf", exist_ok=True)
def load_lf_env_vars(config,environment):
    os.environ["LANGFUSE_PUBLIC_KEY"] = config["langfuse"][environment]["public_key"]
    os.environ["LANGFUSE_SECRET_KEY"] = config["langfuse"][environment]["secret_key"]
    os.environ["LANGFUSE_HOST"] = config["langfuse"][environment]["host"]

#Function to give userfeed bace in Langfuse
def user_feedback(config,trace_if_val, score_val, feedback_val, environment):
    print("The Langfuse Trace ID is : " +str(trace_if_val))

    # check: in case, user skips feedback option on UI
    if not feedback_val:
        feedback_val = "No feedback provided by user"

    langfuse_client = Langfuse(
        public_key = config["langfuse"][environment]["public_key"],
        secret_key = config["langfuse"][environment]["secret_key"],
        host = config["langfuse"][environment]["host"],
    )
    # score method
    langfuse_client.score(
        name = "User Feedback",
        trace_id = trace_if_val,
        value = score_val,
        comment = feedback_val
    )
       
    print("User feedback and score submitted to Langfuse server.\n")

# Function to extract text and images from the PDF
def load_pdf_with_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_data.append(page.get_text("text"))
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            img_path = f"./chatbot_pdf/page-{page_num+1}-img-{img_index+1}.png"
            img.save(img_path)
            images.append(img_path)
    return text_data, images

def generate_image_embeddings(images):
    embeddings = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(normalized.cpu().numpy())
    return np.vstack(embeddings)

class ParentChildTextSplitter:
    def __init__(self, parent_chunk_size=2048, parent_overlap=204, child_chunk_size=512, child_overlap=12):
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_overlap)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_overlap)

    def create_parent_child_chunks(self, text_list):
        parent_chunks = self.parent_splitter.create_documents(text_list)
        all_child_chunks = []
        for parent_chunk in parent_chunks:
            child_texts = self.child_splitter.split_text(parent_chunk.page_content)
            child_chunks = [Document(page_content=text) for text in child_texts]
            all_child_chunks.extend(child_chunks)
        return all_child_chunks
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    environment = "development"  # Define the environment variable
)

@st.cache_resource
def gen_user_id():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = random.randint(100,999)
        return st.session_state.user_id


# Function to log the user query
def log_user_query(query):
    trace = langfuse.trace(
        name="User Query",
        input={"query": query},
        user_id="rag-chatbot-"+str(gen_user_id()), # Using dynamic user_id
    )
    return trace

# Function to log the LLM response
def log_llm_response(trace, response):
    trace.update(
        output={"response": response},
        status="completed"
    )

def main_app_2():
    with open('./css/chatbot.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title("📄 PDF INTERACTION QA BOT")

    # Sidebar: File uploader
    uploaded_files = st.sidebar.file_uploader("📂 Upload your PDF file(s)", type=["pdf"], accept_multiple_files=True)
    print("Pdf Uploaded is done and processing")
    if uploaded_files:
        all_texts = []
        all_images = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join("chatbot_pdf", uploaded_file.name)
            if file_path is None:
                os.mkdir("chatbot_pdf")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            texts, images = load_pdf_with_images(file_path)
            all_texts.extend(texts)
            all_images.extend(images)

        text_splitter = ParentChildTextSplitter(
            parent_chunk_size=2048,
            parent_overlap=204,
            child_chunk_size=512,
            child_overlap=12
        )
        text_chunks = text_splitter.create_parent_child_chunks(all_texts)
        text_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Model loaded successfully!")
        text_db = FAISS.from_documents(text_chunks, text_embeddings)
        image_embeddings = generate_image_embeddings(all_images)
        dimension = image_embeddings.shape[1]
        image_index = faiss.IndexFlatL2(dimension)
        image_index.add(image_embeddings)

        os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(model_name='llama-3.1-8b-instant')
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False, output_key="answer")
        retriever = text_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever,
            return_source_documents=True,
            output_key="answer"
        )

        st.success("✅ PDFs processed successfully!")

        tab1, tab2 = st.tabs(["💬 Ask Questions", "📸 Explore Images"])

        with tab1:
            query = st.text_input("Ask a question about the PDFs or images:")
            submit_button = st.button("Submit Query")

            if submit_button and query:
                with st.spinner("Searching for answers..."):
                    query_inputs = processor(text=[query], return_tensors="pt")
                    with torch.no_grad():
                        query_features = model.get_text_features(**query_inputs)
                        query_normalized = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
                    query_array = query_normalized.cpu().numpy()
                    k = 3
                    distances, indices = image_index.search(query_array, k)
                    formatted_query = prompt_template["user_query"].format(question=query)
                    trace = log_user_query(formatted_query)
                    print("Generated User id = ",gen_user_id())
                    global current_trace_id
                    current_trace_id = trace.id
                    result = qa({"question": formatted_query})
                    source_documents = result.get("source_documents", [])
                    references = [
                        {
                            "chunk": doc.page_content[:300],  # First 200 characters of the chunk
                            "source": doc.metadata.get("source", "Unknown"),  # Extract filename or source
                            "page": doc.metadata.get("page", "N/A"),  # Page number if available
                        }
                        for doc in source_documents
                    ]
                    answer = "I don't know." if "i don't have real-time information" in result["answer"].lower() else result["answer"]
                    formatted_answer = prompt_template["output_format"].format(answer=answer)
                    log_llm_response(trace, formatted_answer)
                    print(trace)
                    st.write("💡Answer:", formatted_answer)
                    trace.update(
                        output={"response": formatted_answer},
                        metadata={"references": references},  # Store references in metadata
                        status="completed"
                    )
                    if references:
                        with st.expander("📖 References (Click to Expand)"):
                            for ref in references:
                                st.write(ref)
                    st.info("Check the '📸 Explore Images' tab to view related images!")
                    print("User Query and LLM Response is done")
                    print("---------------- USER FEEDBACK ----------------")
                    environment = "development" # Define the environment variable

                    def upvote_success_msg():
                    # This function calls submit feedback/score function and display notification on UI for upvote
                        score = 1
                        user_feedback(config, current_trace_id, score, st.session_state['uv_user_input'], environment)
                        st.toast('Feedback submitted!', icon="✅")
 
                    def downvote_sucess_msg():
                        # This function calls submit feedback/score function and display notification on UI for downvote
                        score = -1
                        user_feedback(config, current_trace_id, score, st.session_state['dv_user_input'], environment)
                        st.toast('Feedback submitted!', icon="✅")
 
                    # create st.columns for feedback UI (configured for browser zoom: 100%)
                    col1,col2,col3,col4 = st.columns([0.6,0.6,0.6,10])
    
                    # User score and feedback UI elements
                    if "score_user_feedback" not in st.session_state:
                        # Upvote elements
                        with col1:
                            with st.popover(":thumbsup:"):
                                with st.form(key='uv_form',clear_on_submit=False):
                                    st.markdown("Upvote Feedback")
                                    # Here we assign a key so that it can be uniquely identified and displayed
                                    user_input = st.text_area("Provide feedback for upvote and click on submit:", key = "uv_user_input")      
                                    submitted=st.form_submit_button("Submit", on_click = upvote_success_msg, type = "primary")
                        # Downvote elements
                        with col2:
                            with st.popover(":thumbsdown:"):
                                with st.form(key='dv_form', clear_on_submit=False):
                                    st.markdown("Downvote Feedback")
                                    user_input = st.text_area("Provide feedback for downvote and click on submit", placeholder = "optional", key = 'dv_user_input')
                                    submitted=st.form_submit_button("Submit", on_click = downvote_sucess_msg, type = "primary")
        with tab2:
            st.subheader("📸 Extracted Images")
            for img_path in all_images:
                st.image(img_path, use_container_width=True)

        st.sidebar.subheader("📜 Extracted Text")
        for i, text in enumerate(all_texts):
            with st.sidebar.expander(f"Page {i + 1}"):
                st.write(text)
main_app_2()
