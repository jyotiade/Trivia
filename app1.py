import streamlit as st
import sqlite3
import pandas as pd
import time
import hashlib
import uuid
from datetime import datetime
import json
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import mysql.connector  # Added for MySQL database on EC2

# Set page configuration
st.set_page_config(
    page_title="Trivia Quiz Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'quiz_started' not in st.session_state:
    st.session_state['quiz_started'] = False
if 'quiz_completed' not in st.session_state:
    st.session_state['quiz_completed'] = False
if 'quiz_timer' not in st.session_state:
    st.session_state['quiz_timer'] = 300  # 5 minutes in seconds
if 'quiz_questions' not in st.session_state:
    st.session_state['quiz_questions'] = []
if 'user_answers' not in st.session_state:
    st.session_state['user_answers'] = {}
if 'current_quiz_id' not in st.session_state:
    st.session_state['current_quiz_id'] = None
if 'selected_semester' not in st.session_state:
    st.session_state['selected_semester'] = None
if 'selected_subject' not in st.session_state:
    st.session_state['selected_subject'] = None
if 'quiz_start_time' not in st.session_state:
    st.session_state['quiz_start_time'] = None

# Database configuration for EC2 MySQL
DB_CONFIG = {
    'host': 'ec2-13-203-161-41.ap-south-1.compute.amazonaws.com',  # Replace with your EC2 public DNS
    'user': 'admin',          # Replace with your MySQL username
    'password': 'admin123',  # Replace with your MySQL password
    'database': 'academic_quiz'  # Replace with your database name
}

# Helper function to get database connection
def get_db_connection():
    try:
        # Try to connect to MySQL database on EC2
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn, "mysql"
    except Exception as e:
        # Fallback to SQLite if EC2 connection fails
        print(f"Error connecting to MySQL on EC2: {e}")
        print("Falling back to SQLite database")
        conn = sqlite3.connect('academic_quiz.db')
        return conn, "sqlite"

# Database setup
def init_db():
    conn, db_type = get_db_connection()
    
    try:
        cursor = conn.cursor()
        
        # Create users table
        if db_type == "mysql":
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL
            )
            ''')
            
            # Create semesters table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS semesters (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL
            )
            ''')
            
            # Create subjects table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                semester_id INT,
                FOREIGN KEY (semester_id) REFERENCES semesters (id) ON DELETE CASCADE
            )
            ''')
            
            # Create quizzes table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quizzes (
                id VARCHAR(36) PRIMARY KEY,
                subject_id INT,
                created_at TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE
            )
            ''')
            
            # Create questions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                quiz_id VARCHAR(36),
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer VARCHAR(1) NOT NULL,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            )
            ''')
            
            # Create results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(36),
                quiz_id VARCHAR(36),
                score INT,
                total_questions INT,
                completed_at TIMESTAMP,
                time_taken INT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            )
            ''')
        else:
            # SQLite version
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            )
            ''')
            
            # Create semesters table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS semesters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            ''')
            
            # Create subjects table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                semester_id INTEGER,
                FOREIGN KEY (semester_id) REFERENCES semesters (id) ON DELETE CASCADE
            )
            ''')
            
            # Create quizzes table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quizzes (
                id TEXT PRIMARY KEY,
                subject_id INTEGER,
                created_at TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects (id) ON DELETE CASCADE
            )
            ''')
            
            # Create questions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id TEXT,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            )
            ''')
            
            # Create results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                quiz_id TEXT,
                score INTEGER,
                total_questions INTEGER,
                completed_at TIMESTAMP,
                time_taken INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (id) ON DELETE CASCADE
            )
            ''')
        
        # Insert initial admin user if not exists - works for both MySQL and SQLite
        if db_type == "mysql":
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            admin_exists = cursor.fetchone()
        else:
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            admin_exists = cursor.fetchone()
            
        if not admin_exists:
            admin_id = str(uuid.uuid4())
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()
            if db_type == "mysql":
                cursor.execute("INSERT INTO users VALUES (%s, %s, %s, %s)", 
                          (admin_id, 'admin', password_hash, 'admin'))
            else:
                cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                          (admin_id, 'admin', password_hash, 'admin'))
        
        conn.commit()
    finally:
        conn.close()

# Initialize database
init_db()

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("SELECT id, password_hash, role FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
        else:
            cursor.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
        
        if user and user[1] == hash_password(password):
            return {"id": user[0], "username": username, "role": user[2]}
        return None
    finally:
        conn.close()

def register_user(username, password, role="user"):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        
        if db_type == "mysql":
            try:
                cursor.execute("INSERT INTO users VALUES (%s, %s, %s, %s)", 
                          (user_id, username, password_hash, role))
                conn.commit()
                return {"id": user_id, "username": username, "role": role}
            except mysql.connector.IntegrityError:
                return None
        else:
            try:
                cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                          (user_id, username, password_hash, role))
                conn.commit()
                return {"id": user_id, "username": username, "role": role}
            except sqlite3.IntegrityError:
                return None
    finally:
        conn.close()

def get_semesters():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM semesters ORDER BY name")
        semesters = cursor.fetchall()
        
        if not semesters:
            return pd.DataFrame(columns=['id', 'name'])
        
        return pd.DataFrame(semesters, columns=['id', 'name'])
    finally:
        conn.close()

def get_subjects_by_semester(semester_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Make sure semester_id is an integer
        semester_id = int(semester_id)
        
        # Debugging - Print the SQL query parameters
        print(f"Fetching subjects for semester_id: {semester_id}")
        
        # Execute the query with parameter binding
        if db_type == "mysql":
            cursor.execute("SELECT id, name, semester_id FROM subjects WHERE semester_id = %s ORDER BY name", (semester_id,))
        else:
            cursor.execute("SELECT id, name, semester_id FROM subjects WHERE semester_id = ? ORDER BY name", (semester_id,))
            
        subjects = cursor.fetchall()
        
        # Debugging - Print raw query results
        print(f"Raw query results: {subjects}")
        
        if not subjects:
            print("No subjects found for this semester_id")
            return pd.DataFrame(columns=['id', 'name', 'semester_id'])
        
        result_df = pd.DataFrame(subjects, columns=['id', 'name', 'semester_id'])
        print(f"Returning DataFrame: {result_df}")
        return result_df
        
    except Exception as e:
        print(f"Error in get_subjects_by_semester: {e}")
        return pd.DataFrame(columns=['id', 'name', 'semester_id'])
    finally:
        conn.close()

def get_all_subjects():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("""
                SELECT subjects.id, subjects.name, semesters.name as semester 
                FROM subjects 
                JOIN semesters ON subjects.semester_id = semesters.id 
                ORDER BY semesters.name, subjects.name
            """)
        else:
            cursor.execute("""
                SELECT subjects.id, subjects.name, semesters.name as semester 
                FROM subjects 
                JOIN semesters ON subjects.semester_id = semesters.id 
                ORDER BY semesters.name, subjects.name
            """)
            
        subjects = cursor.fetchall()
        
        if not subjects:
            return pd.DataFrame(columns=['id', 'name', 'semester'])
        
        return pd.DataFrame(subjects, columns=['id', 'name', 'semester'])
    finally:
        conn.close()

def get_all_users():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, username, role FROM users ORDER BY username")
        users = cursor.fetchall()
        
        if not users:
            return pd.DataFrame(columns=['id', 'username', 'role'])
        
        return pd.DataFrame(users, columns=['id', 'username', 'role'])
    finally:
        conn.close()

def get_all_results():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                results.id,
                users.username,
                subjects.name,
                semesters.name,
                results.score,
                results.total_questions,
                results.completed_at,
                results.time_taken
            FROM results
            JOIN users ON results.user_id = users.id
            JOIN quizzes ON results.quiz_id = quizzes.id
            JOIN subjects ON quizzes.subject_id = subjects.id
            JOIN semesters ON subjects.semester_id = semesters.id
            ORDER BY results.completed_at DESC
        """)
        results = cursor.fetchall()
        
        if not results:
            return pd.DataFrame(columns=['id', 'username', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
        
        return pd.DataFrame(results, columns=['id', 'username', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
    
    except Exception as e:
        print(f"Error in get_all_results: {e}")
        return pd.DataFrame(columns=['id', 'username', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
    finally:
        conn.close()

def get_user_results(user_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("""
                SELECT 
                    results.id,
                    subjects.name,
                    semesters.name,
                    results.score,
                    results.total_questions,
                    results.completed_at,
                    results.time_taken
                FROM results
                JOIN quizzes ON results.quiz_id = quizzes.id
                JOIN subjects ON quizzes.subject_id = subjects.id
                JOIN semesters ON subjects.semester_id = semesters.id
                WHERE results.user_id = %s
                ORDER BY results.completed_at DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT 
                    results.id,
                    subjects.name,
                    semesters.name,
                    results.score,
                    results.total_questions,
                    results.completed_at,
                    results.time_taken
                FROM results
                JOIN quizzes ON results.quiz_id = quizzes.id
                JOIN subjects ON quizzes.subject_id = subjects.id
                JOIN semesters ON subjects.semester_id = semesters.id
                WHERE results.user_id = ?
                ORDER BY results.completed_at DESC
            """, (user_id,))
            
        results = cursor.fetchall()
        
        if not results:
            return pd.DataFrame(columns=['id', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
        
        return pd.DataFrame(results, columns=['id', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
    
    except Exception as e:
        print(f"Error in get_user_results: {e}")
        return pd.DataFrame(columns=['id', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken'])
    finally:
        conn.close()

def generate_quiz(subject_id, subject_name):
    # Initialize Groq LLM
    groq_api_key = "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"
    if not groq_api_key:
        st.error("GROQ API Key not found. Please set the GROQ_API_KEY environment variable.")
        return None
    
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )
    
    # Create a quiz generation prompt
    system_template = """You are an expert academic quiz creator. Your task is to create a challenging but fair multiple-choice quiz on the subject of {subject}.
    
    Create exactly 10 multiple choice questions, each with FOUR options (labeled A, B, C, D) where exactly one option is correct.
    
    Your questions should be academically rigorous, clear, and directly related to the subject. Make sure all questions are unique and cover different aspects of the subject.
    
    Format your response as a JSON array of question objects, with each object having the fields:
    - question_text: The question being asked
    - option_a: First answer choice
    - option_b: Second answer choice
    - option_c: Third answer choice
    - option_d: Fourth answer choice
    - correct_answer: The letter of the correct answer (A, B, C, or D)
    
    Make sure the output is valid JSON. Do not include any text before or after the JSON.
    """
    
    human_template = "Generate 10 academically rigorous multiple-choice questions on {subject}."
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    try:
        # Generate questions using LangChain and Groq
        with st.spinner("Generating quiz questions..."):
            response = chain.run(subject=subject_name)
            
            # Parse the JSON response
            try:
                quiz_data = json.loads(response)
                
                # Validate the data structure
                if not isinstance(quiz_data, list) or len(quiz_data) != 10:
                    st.error("Invalid question format received from AI model.")
                    return None
                
                # Store quiz in database
                quiz_id = str(uuid.uuid4())
                conn, db_type = get_db_connection()
                cursor = conn.cursor()
                
                try:
                    if db_type == "mysql":
                        cursor.execute(
                            "INSERT INTO quizzes (id, subject_id, created_at) VALUES (%s, %s, %s)",
                            (quiz_id, subject_id, datetime.now())
                        )
                        
                        for question in quiz_data:
                            cursor.execute(
                                """
                                INSERT INTO questions 
                                (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_answer) 
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    quiz_id, 
                                    question['question_text'], 
                                    question['option_a'], 
                                    question['option_b'], 
                                    question['option_c'], 
                                    question['option_d'], 
                                    question['correct_answer']
                                )
                            )
                    else:
                        cursor.execute(
                            "INSERT INTO quizzes (id, subject_id, created_at) VALUES (?, ?, ?)",
                            (quiz_id, subject_id, datetime.now())
                        )
                        
                        for question in quiz_data:
                            cursor.execute(
                                """
                                INSERT INTO questions 
                                (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_answer) 
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    quiz_id, 
                                    question['question_text'], 
                                    question['option_a'], 
                                    question['option_b'], 
                                    question['option_c'], 
                                    question['option_d'], 
                                    question['correct_answer']
                                )
                            )
                    
                    conn.commit()
                    return {"quiz_id": quiz_id, "questions": quiz_data}
                    
                finally:
                    conn.close()
                    
            except json.JSONDecodeError:
                st.error("Failed to parse AI response as JSON.")
                return None
    
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def get_quiz_questions(quiz_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("""
                SELECT 
                    id,
                    question_text,
                    option_a,
                    option_b,
                    option_c,
                    option_d,
                    correct_answer
                FROM questions 
                WHERE quiz_id = %s
            """, (quiz_id,))
        else:
            cursor.execute("""
                SELECT 
                    id,
                    question_text,
                    option_a,
                    option_b,
                    option_c,
                    option_d,
                    correct_answer
                FROM questions 
                WHERE quiz_id = ?
            """, (quiz_id,))
            
        questions = cursor.fetchall()
        
        if not questions:
            return []
        
        # Convert to list of dictionaries
        question_list = []
        for q in questions:
            question_list.append({
                'id': q[0],
                'question_text': q[1],
                'option_a': q[2],
                'option_b': q[3],
                'option_c': q[4],
                'option_d': q[5],
                'correct_answer': q[6]
            })
        
        return question_list
    finally:
        conn.close()

def save_quiz_result(user_id, quiz_id, score, total_questions, time_taken):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute(
                """
                INSERT INTO results 
                (user_id, quiz_id, score, total_questions, completed_at, time_taken) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, quiz_id, score, total_questions, datetime.now(), time_taken)
            )
        else:
            cursor.execute(
                """
                INSERT INTO results 
                (user_id, quiz_id, score, total_questions, completed_at, time_taken) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, quiz_id, score, total_questions, datetime.now(), time_taken)
            )
        
        conn.commit()
    finally:
        conn.close()

def update_user_role(user_id, new_role):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("UPDATE users SET role = %s WHERE id = %s", (new_role, user_id))
        else:
            cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
            
        conn.commit()
    finally:
        conn.close()

def delete_user(user_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        else:
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
        conn.commit()
    finally:
        conn.close()

def add_semester(name):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            try:
                cursor.execute("INSERT INTO semesters (name) VALUES (%s)", (name,))
                conn.commit()
                return True
            except mysql.connector.IntegrityError:
                return False
        else:
            try:
                cursor.execute("INSERT INTO semesters (name) VALUES (?)", (name,))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    finally:
        conn.close()

def add_subject(name, semester_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("INSERT INTO subjects (name, semester_id) VALUES (%s, %s)", (name, semester_id))
        else:
            cursor.execute("INSERT INTO subjects (name, semester_id) VALUES (?, ?)", (name, semester_id))
            
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding subject: {e}")
        return False
    finally:
        conn.close()

def delete_subject(subject_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("DELETE FROM subjects WHERE id = %s", (subject_id,))
        else:
            cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
            
        conn.commit()
    finally:
        conn.close()

def delete_semester(semester_id):
    conn, db_type = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if db_type == "mysql":
            cursor.execute("DELETE FROM semesters WHERE id = %s", (semester_id,))
        else:
            cursor.execute("DELETE FROM semesters WHERE id = ?", (semester_id,))
            
        conn.commit()
    finally:
        conn.close()

# UI Components
def display_login_register():
    st.title("🧠 Academic Quiz Generator")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="btn_login"):
            user = verify_user(username, password)
            if user:
                st.session_state['user_id'] = user["id"]
                st.session_state['username'] = user["username"]
                st.session_state['role'] = user["role"]
                st.success(f"Successfully logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register", key="btn_register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not new_username or not new_password:
                st.error("Username and password are required")
            else:
                user = register_user(new_username, new_password)
                if user:
                    st.success("Registration successful! Please login.")
                    st.session_state['user_id'] = user["id"]
                    st.session_state['username'] = user["username"]
                    st.session_state['role'] = user["role"]
                    st.rerun()
                else:
                    st.error("Username already exists. Please choose a different username.")

def display_user_home():
    st.title(f"🧠 Welcome to Academic Quiz Generator, {st.session_state['username']}!")
    
    # Reset quiz state if returning to home
    if st.session_state['quiz_completed'] and st.button("Return to Home", key="btn_return_home"):
        st.session_state['quiz_started'] = False
        st.session_state['quiz_completed'] = False
        st.session_state['quiz_questions'] = []
        st.session_state['user_answers'] = {}
        st.session_state['current_quiz_id'] = None
        st.session_state['selected_semester'] = None
        st.session_state['selected_subject'] = None
        st.rerun()
    
    if not st.session_state['quiz_started'] and not st.session_state['quiz_completed']:
        st.subheader("Select a Subject")
        
        # Get semesters - Always fetch fresh data from database
        semesters = get_semesters()
        
        if semesters.empty:
            st.info("No semesters available. Please ask an administrator to add semesters and subjects.")
            
            # Display user's past results
            display_user_results()
            return
        
        semester_options = semesters['name'].tolist()
        
        # Default to first semester if none selected
        selected_semester_index = 0
        if st.session_state['selected_semester'] is not None:
            try:
                selected_semester_index = semester_options.index(st.session_state['selected_semester'])
            except ValueError:
                selected_semester_index = 0
                # Reset selected subject when semester not found
                st.session_state['selected_subject'] = None
        
        selected_semester_name = st.selectbox(
            "Select Semester", 
            options=semester_options,
            index=selected_semester_index,
            key="select_semester"
        )
        
        # Store selected semester in session state
        st.session_state['selected_semester'] = selected_semester_name
        
        # Get semester ID
        selected_semester_id = semesters.loc[semesters['name'] == selected_semester_name, 'id'].iloc[0]
        
        # Get subjects for selected semester
        subjects = get_subjects_by_semester(selected_semester_id)
        
        if subjects.empty:
            st.info(f"No subjects available for {selected_semester_name}. Please ask an administrator to add subjects.")
            
            # Display user's past results
            display_user_results()
            return
        
        # Create subject options
        subject_options = subjects['name'].tolist()
        selected_subject_index = 0
        
        if st.session_state['selected_subject'] is not None and st.session_state['selected_subject'] in subject_options:
            selected_subject_index = subject_options.index(st.session_state['selected_subject'])
        
        selected_subject_name = st.selectbox(
            "Select Subject", 
            options=subject_options,
            index=selected_subject_index,
            key="select_subject"
        )
        
        # Store selected subject in session state
        st.session_state['selected_subject'] = selected_subject_name
        
        # Get subject ID
        selected_subject_id = subjects.loc[subjects['name'] == selected_subject_name, 'id'].iloc[0]
        
        # Add hover effect with CSS
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Start quiz button
        if st.button("Start Quiz", key="btn_start_quiz"):
            quiz_data = generate_quiz(selected_subject_id, selected_subject_name)
            
            if quiz_data:
                st.session_state['quiz_started'] = True
                st.session_state['quiz_questions'] = quiz_data["questions"]
                st.session_state['current_quiz_id'] = quiz_data["quiz_id"]
                st.session_state['quiz_start_time'] = time.time()
                st.rerun()
            else:
                st.error("Failed to generate quiz. Please try again.")
        
        # Display user's past results
        display_user_results()
    
    elif st.session_state['quiz_started'] and not st.session_state['quiz_completed']:
        display_quiz()
    
    elif st.session_state['quiz_completed']:
        display_quiz_results()

def display_quiz():
    st.title(f"Quiz: {st.session_state['selected_subject']}")
    
    # Calculate remaining time
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    remaining_time = max(0, st.session_state['quiz_timer'] - elapsed_time)
    
    # Display timer
    timer_col, _ = st.columns([1, 3])
    with timer_col:
        st.metric("Time Remaining", f"{remaining_time // 60}:{remaining_time % 60:02d}")
    
    # Check if time's up
    if remaining_time <= 0:
        st.session_state['quiz_completed'] = True
        st.rerun()
    
    # Create a form for the quiz
    with st.form(key="quiz_form"):
        for i, question in enumerate(st.session_state['quiz_questions']):
            st.subheader(f"Question {i+1}")
            st.write(question["question_text"])
            
            # Store the user's answer
            option_key = f"question_{i}"
            options = [
                ("A", question["option_a"]),
                ("B", question["option_b"]),
                ("C", question["option_c"]),
                ("D", question["option_d"])
            ]
            
            selected_option = st.radio(
                "Select your answer:",
                options,
                format_func=lambda x: f"{x[0]}. {x[1]}",
                key=option_key
            )
            
            # Store just the option letter (A, B, C, D)
            st.session_state['user_answers'][i] = selected_option[0]
            
            st.markdown("---")
        
        # Add some CSS to make the submit button more attractive with hover effect
        st.markdown("""
        <style>
        div.stButton > button[kind="primaryFormSubmit"] {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        div.stButton > button[kind="primaryFormSubmit"]:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 20px 0 rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Submit button
        if st.form_submit_button("Submit Quiz"):
            time_taken = int(time.time() - st.session_state['quiz_start_time'])
            st.session_state['quiz_completed'] = True
            st.rerun()

def display_quiz_results():
    st.title("Quiz Results")
    
    # Calculate score
    score = 0
    total_questions = len(st.session_state['quiz_questions'])
    
    for i, question in enumerate(st.session_state['quiz_questions']):
        if i in st.session_state['user_answers']:
            if st.session_state['user_answers'][i] == question['correct_answer']:
                score += 1
    
    # Calculate time taken
    time_taken = int(time.time() - st.session_state['quiz_start_time'])
    
    # Save result to database
    save_quiz_result(
        st.session_state['user_id'],
        st.session_state['current_quiz_id'],
        score,
        total_questions,
        time_taken
    )
    
    # Display score
    st.subheader(f"Score: {score}/{total_questions}")
    
    # Display time taken
    minutes, seconds = divmod(time_taken, 60)
    st.subheader(f"Time taken: {minutes}m {seconds}s")
    
    # Display percentage
    percentage = (score / total_questions) * 100
    st.subheader(f"Percentage: {percentage:.1f}%")
    
    # Display pass/fail
    if percentage >= 60:
        st.success("PASSED! 🎉")
    else:
        st.error("FAILED. Try again.")
    
    # Review answers
    st.subheader("Review Answers")
    
    for i, question in enumerate(st.session_state['quiz_questions']):
        st.write(f"**Question {i+1}**: {question['question_text']}")
        
        options = [
            ("A", question["option_a"]),
            ("B", question["option_b"]),
            ("C", question["option_c"]),
            ("D", question["option_d"])
        ]
        
        for opt, text in options:
            if opt == question['correct_answer']:
                st.success(f"{opt}. {text} ✓")
            elif i in st.session_state['user_answers'] and opt == st.session_state['user_answers'][i]:
                st.error(f"{opt}. {text} ✗")
            else:
                st.write(f"{opt}. {text}")
        
        st.markdown("---")
    
    # Add some CSS to make the button more attractive with hover effect
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Return to home button
    st.button("Return to Home", key="btn_return_home")

def display_user_results():
    st.subheader("Your Quiz History")
    
    results_df = get_user_results(st.session_state['user_id'])
    
    if results_df.empty:
        st.info("You haven't taken any quizzes yet.")
        return
    
    # Format the dataframe
    results_df['Score'] = results_df['score'].astype(str) + '/' + results_df['total_questions'].astype(str)
    results_df['Percentage'] = (results_df['score'] / results_df['total_questions'] * 100).round(1).astype(str) + '%'
    
    # Format time taken
    results_df['Time'] = results_df['time_taken'].apply(lambda x: f"{x // 60}m {x % 60}s")
    
    # Display results
    st.dataframe(
        results_df[['subject', 'semester', 'Score', 'Percentage', 'Time', 'completed_at']].rename(columns={
            'subject': 'Subject',
            'semester': 'Semester',
            'completed_at': 'Date'
        })
    )

def display_admin_panel():
    st.title("Admin Panel")
    
    tabs = st.tabs(["Users", "Semesters & Subjects", "Quiz Results", "Database Setup"])
    
    with tabs[0]:
        st.subheader("Manage Users")
        
        users_df = get_all_users()
        
        if users_df.empty:
            st.info("No users found.")
            return
        
        # Don't show the current user
        users_df = users_df[users_df['id'] != st.session_state['user_id']]
        
        # Display users with action buttons
        for index, row in users_df.iterrows():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"**{row['username']}**")
            
            with col2:
                st.write(f"Role: {row['role']}")
            
            with col3:
                if st.button("Make Admin", key=f"make_admin_{row['id']}"):
                    update_user_role(row['id'], "admin")
                    st.success(f"Updated {row['username']} to admin")
                    st.rerun()
            
            with col4:
                if st.button("Delete User", key=f"delete_user_{row['id']}"):
                    delete_user(row['id'])
                    st.success(f"Deleted user {row['username']}")
                    st.rerun()
            
            st.markdown("---")
    
    with tabs[1]:
        st.subheader("Manage Semesters & Subjects")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Add Semester")
            semester_name = st.text_input("Semester Name", key="new_semester_name")
            
            if st.button("Add Semester", key="btn_add_semester"):
                if semester_name:
                    success = add_semester(semester_name)
                    if success:
                        st.success(f"Added semester: {semester_name}")
                        st.rerun()
                    else:
                        st.error(f"Semester '{semester_name}' already exists.")
                else:
                    st.error("Please enter a semester name")
        
        with col2:
            st.write("### Add Subject")
            
            # Get semesters for dropdown
            semesters_df = get_semesters()
            if not semesters_df.empty:
                semester_options = semesters_df['name'].tolist()
                selected_semester = st.selectbox("Select Semester", options=semester_options, key="semester_for_subject")
                semester_id = semesters_df.loc[semesters_df['name'] == selected_semester, 'id'].iloc[0]
                
                subject_name = st.text_input("Subject Name", key="new_subject_name")
                
                if st.button("Add Subject", key="btn_add_subject"):
                    if subject_name:
                        success = add_subject(subject_name, semester_id)
                        if success:
                            st.success(f"Added subject: {subject_name}")
                            st.rerun()
                        else:
                            st.error("Failed to add subject")
                    else:
                        st.error("Please enter a subject name")
            else:
                st.info("Add a semester first")
        
        st.markdown("---")
        
        # View and delete semesters
        st.write("### Manage Semesters")
        semesters_df = get_semesters()
        
        if not semesters_df.empty:
            for index, row in semesters_df.iterrows():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{row['name']}**")
                
                with col2:
                    if st.button("Delete", key=f"delete_semester_{row['id']}"):
                        delete_semester(row['id'])
                        st.success(f"Deleted semester: {row['name']}")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No semesters found")
        
        # View and delete subjects
        st.write("### Manage Subjects")
        subjects_df = get_all_subjects()
        
        if not subjects_df.empty:
            for index, row in subjects_df.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{row['name']}**")
                
                with col2:
                    st.write(f"Semester: {row['semester']}")
                
                with col3:
                    if st.button("Delete", key=f"delete_subject_{row['id']}"):
                        delete_subject(row['id'])
                        st.success(f"Deleted subject: {row['name']}")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("No subjects found")
    
    with tabs[2]:
        st.subheader("Quiz Results")
        
        results_df = get_all_results()
        
        if results_df.empty:
            st.info("No quiz results yet.")
            return
        
        # Format the dataframe
        results_df['Score'] = results_df['score'].astype(str) + '/' + results_df['total_questions'].astype(str)
        results_df['Percentage'] = (results_df['score'] / results_df['total_questions'] * 100).round(1).astype(str) + '%'
        
        # Format time taken
        results_df['Time'] = results_df['time_taken'].apply(lambda x: f"{x // 60}m {x % 60}s")
        
        # Display results
        st.dataframe(
            results_df[['username', 'subject', 'semester', 'Score', 'Percentage', 'Time', 'completed_at']].rename(columns={
                'username': 'User',
                'subject': 'Subject',
                'semester': 'Semester',
                'completed_at': 'Date'
            })
        )
    
    with tabs[3]:
        st.subheader("EC2 Database Setup")
        
        st.markdown("""
        ### Configure MySQL Database on EC2
        
        Follow these steps to set up the MySQL database on your EC2 instance:
        
        1. **Launch an EC2 instance**:
           - Use Amazon Linux 2023 or Ubuntu Server
           - Ensure port 3306 is open in the security group
        
        2. **SSH into your EC2 instance**:
           ```
           ssh -i your-key.pem ec2-user@your-ec2-public-dns.compute.amazonaws.com
           ```
        
        3. **Install MySQL**:
           
           For Amazon Linux:
           ```
           sudo yum update -y
           sudo yum install -y mysql-server
           sudo systemctl start mysqld
           sudo systemctl enable mysqld
           ```
           
           For Ubuntu:
           ```
           sudo apt update
           sudo apt install -y mysql-server
           sudo systemctl start mysql
           sudo systemctl enable mysql
           ```
        
        4. **Secure MySQL installation**:
           ```
           sudo mysql_secure_installation
           ```
        
        5. **Create database and user**:
           ```
           sudo mysql -u root -p
           ```
           
           Then in MySQL shell:
           ```sql
           CREATE DATABASE academic_quiz;
           CREATE USER 'db_user'@'%' IDENTIFIED BY 'db_password';
           GRANT ALL PRIVILEGES ON academic_quiz.* TO 'db_user'@'%';
           FLUSH PRIVILEGES;
           EXIT;
           ```
        
        6. **Configure MySQL to accept remote connections**:
           
           Edit the MySQL configuration file:
           ```
           sudo nano /etc/my.cnf   # For Amazon Linux
           # OR
           sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf   # For Ubuntu
           ```
           
           Find the line with `bind-address` and change it to:
           ```
           bind-address = 0.0.0.0
           ```
           
           Restart MySQL:
           ```
           sudo systemctl restart mysqld   # For Amazon Linux
           # OR
           sudo systemctl restart mysql    # For Ubuntu
           ```
        
        7. **Update the application configuration**:
           
           Update the DB_CONFIG in this application with your EC2 details:
           ```python
           DB_CONFIG = {
               'host': 'your-ec2-instance-public-dns.compute.amazonaws.com',
               'user': 'db_user',
               'password': 'db_password',
               'database': 'academic_quiz'
           }
           ```
        """)
        
        # Form to update database configuration
        st.write("### Update Database Configuration")
        
        with st.form("db_config_form"):
            ec2_host = st.text_input("EC2 Host/Public DNS", value=DB_CONFIG.get('host', ''))
            db_user = st.text_input("Database Username", value=DB_CONFIG.get('user', ''))
            db_password = st.text_input("Database Password", type="password", value=DB_CONFIG.get('password', ''))
            db_name = st.text_input("Database Name", value=DB_CONFIG.get('database', ''))
            
            if st.form_submit_button("Test Connection"):
                try:
                    test_config = {
                        'host': ec2_host,
                        'user': db_user,
                        'password': db_password,
                        'database': db_name
                    }
                    conn = mysql.connector.connect(**test_config)
                    conn.close()
                    st.success("Connection successful! Update your code with these settings.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

def display_logout_button():
    if st.sidebar.button("Logout", key="btn_logout"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main app
def main():
    st.sidebar.title("Navigation")
    
    if st.session_state['user_id'] is None:
        display_login_register()
    else:
        display_logout_button()
        
        if st.session_state['role'] == "admin":
            pages = ["Home", "Admin Panel"]
            selection = st.sidebar.radio("Go to", pages)
            
            if selection == "Home":
                display_user_home()
            elif selection == "Admin Panel":
                display_admin_panel()
        else:
            display_user_home()

if __name__ == "__main__":
    main()
