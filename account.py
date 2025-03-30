import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from chatbot import main_app_2  # Assuming main_app_2 is implemented in pdf_reader.py
import re
import json
# Initialize Firebase
if not firebase_admin._apps:
    firebase_config = json.loads(st.secrets["firebase"])  # Load from Streamlit secrets
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

st.set_page_config(page_title="AI Assistant & PDF Q&A Bot", layout="wide",page_icon="🤖")
with open('./css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'useremail' not in st.session_state:
    st.session_state.useremail = ''
if 'signedout' not in st.session_state:
    st.session_state.signedout = False
if 'signout' not in st.session_state:
    st.session_state.signout = False
if 'selected_app' not in st.session_state:
    st.session_state.selected_app = 'chatbot'
# Login function
def login(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.username = user.uid  # Using UID as the username
        st.session_state.useremail = user.email
        st.session_state.signedout = True
        st.session_state.signout = True
        st.session_state.page = 'dashboard'
        st.success("Login successful")
        # st.query_params(page='dashboard')  # Update URL
    except Exception as e:
        st.warning(f'Login failed: {str(e)}')

# Logout function
def logout():
    st.session_state.signedout = False
    st.session_state.signout = False
    st.session_state.username = ''
    st.session_state.useremail = ''
    st.session_state.page = 'login'
    # st.query_params(page='login')  # Update URL

# Signup function
def signup():
    st.subheader("Create an account")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type='password')
    username = st.text_input("Enter your Name")
    
    if st.button('Create my account'):
        try:
            user = auth.create_user(email=email, password=password, uid=username)
            st.success('Account created successfully')
            st.session_state.page = 'login'  # Redirect to login page
            # st.query_params(page='login')
            st.rerun()
        except firebase_admin.auth.EmailAlreadyExistsError:
            st.error('Account creation failed: Email already exists')
        except Exception as e:
            st.error(f'Account creation failed: {str(e)}')
    
    if st.button("Already have an account? Login"):
        st.session_state.page = 'login'
        # st.query_params(page='login')
        st.rerun()

# Login page
def login_page():
    st.subheader("Login")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        login(email, password)
    
    if st.button("Don't have an account? Create one"):
        st.session_state.page = 'signup'
        # st.query_params(page='signup')
        st.rerun()

# Dashboard with sidebar
def dashboard():
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    option = st.sidebar.selectbox("Select an option", ["PDF Q&A ReaderBot"])
    
    if option == "PDF Q&A ReaderBot":
        st.session_state.selected_app = 'chatbot'
    
    if st.sidebar.button("Signout"):
        logout()
        st.rerun()
    
    if st.session_state.selected_app == 'chatbot':
        main_app_2()
if st.session_state.page == 'login':
    st.title("Welcome to AI Assistant Resume & PDF Q&A Bot")
    login_page()
elif st.session_state.page == 'signup':
    st.title("Welcome to AI Assistant Resume & PDF Q&A Bot")
    signup()
elif st.session_state.page == 'dashboard':
    dashboard()
