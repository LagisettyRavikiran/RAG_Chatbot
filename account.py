import streamlit as st
import sqlite3
from chatbot import main_app_2  # Assuming main_app_2 is implemented in pdf_reader.py
import re
import hashlib
st.set_page_config(page_title="AI Assistant & PDF Q&A Bot", layout="wide", page_icon="🤖")
with open('./css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Database setup
def create_users_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                  (username, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    finally:
        conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE email = ? AND password = ?", 
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def validate_password(password):
    if (len(password) > 6 or
        not re.search(r'[A-Z]', password) or
        not re.search(r'\d', password) or
        not re.search(r'[@$!%*?&]', password)):
        return False
    return True

# Initialize database
create_users_table()
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
    user = authenticate_user(email, password)
    if user:
        st.session_state.username = user[0]  # Username from DB
        st.session_state.useremail = email
        st.session_state.signedout = True
        st.session_state.signout = True
        st.session_state.page = 'dashboard'
        st.success("Login successful")
    else:
        st.warning("Invalid email or password")

# Logout function
def logout():
    st.session_state.signedout = False
    st.session_state.signout = False
    st.session_state.username = ''
    st.session_state.useremail = ''
    st.session_state.page = 'login'

# Signup function
def signup():
    st.subheader("Create an account")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type='password')
    username = st.text_input("Enter your Name")
    
    if st.button('Create my account'):
        if not validate_password(password):
            st.warning("Password must be at most 6 characters long and include at least one capital letter, one special character, and one digit.")
        elif add_user(username, email, password):
            st.success('Account created successfully')
            st.session_state.page = 'login'
            st.rerun()
        else:
            st.error('Account creation failed: Email already exists')
    
    if st.button("Already have an account? Login"):
        st.session_state.page = 'login'
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
