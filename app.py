import streamlit as st
import sqlite3
import pandas as pd
import time
import hashlib
import uuid
from datetime import datetime
import langchain
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import os

# Set page configuration
st.set_page_config(
    page_title="Trivia Quiz",
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
    st.session_state['quiz_timer'] = 300  # 10 minutes in seconds
    
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

# Database setup
def init_db():
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    
    # Create users table
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
        name TEXT NOT NULL
    )
    ''')
    
    # Create subjects table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        semester_id INTEGER,
        FOREIGN KEY (semester_id) REFERENCES semesters (id)
    )
    ''')
    
    # Create quizzes table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quizzes (
        id TEXT PRIMARY KEY,
        subject_id INTEGER,
        created_at TIMESTAMP,
        FOREIGN KEY (subject_id) REFERENCES subjects (id)
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
        FOREIGN KEY (quiz_id) REFERENCES quizzes (id)
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
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (quiz_id) REFERENCES quizzes (id)
    )
    ''')
    
    # Insert initial admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_id = str(uuid.uuid4())
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                      (admin_id, 'admin', password_hash, 'admin'))
    
    # Insert initial semesters if not exists
    cursor.execute("SELECT * FROM semesters")
    if not cursor.fetchall():
        for i in range(1, 9):
            cursor.execute("INSERT INTO semesters (name) VALUES (?)", (f"Semester {i}",))
    
    # Insert sample subjects if not exists
    cursor.execute("SELECT * FROM subjects")
    if not cursor.fetchall():
        subjects = [
            ("Mathematics", 1), ("Physics", 1), ("Chemistry", 1), ("Biology", 1), ("History", 1), ("Geography", 1),
            ("Computer Science", 2), ("Statistics", 2), ("Economics", 2), ("Psychology", 2), ("Sociology", 2), ("English", 2),
            # Add more subjects for other semesters...
        ]
        cursor.executemany("INSERT INTO subjects (name, semester_id) VALUES (?, ?)", subjects)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        return {"id": user[0], "username": username, "role": user[2]}
    return None

def register_user(username, password, role="user"):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                      (user_id, username, password_hash, role))
        conn.commit()
        conn.close()
        return {"id": user_id, "username": username, "role": role}
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_semesters():
    conn = sqlite3.connect('trivia_quiz.db')
    semesters = pd.read_sql_query("SELECT * FROM semesters ORDER BY name", conn)
    conn.close()
    return semesters

def get_subjects_by_semester(semester_id):
    conn = sqlite3.connect('trivia_quiz.db')
    subjects = pd.read_sql_query(
        "SELECT * FROM subjects WHERE semester_id = ? ORDER BY name", 
        conn, 
        params=(semester_id,)
    )
    conn.close()
    return subjects

def get_all_subjects():
    conn = sqlite3.connect('trivia_quiz.db')
    subjects = pd.read_sql_query(
        """
        SELECT subjects.id, subjects.name, semesters.name as semester 
        FROM subjects 
        JOIN semesters ON subjects.semester_id = semesters.id 
        ORDER BY semesters.name, subjects.name
        """, 
        conn
    )
    conn.close()
    return subjects

def get_all_users():
    conn = sqlite3.connect('trivia_quiz.db')
    users = pd.read_sql_query("SELECT id, username, role FROM users ORDER BY username", conn)
    conn.close()
    return users

def get_all_results():
    conn = sqlite3.connect('trivia_quiz.db')
    try:
        results = pd.read_sql_query(
            """
            SELECT 
                results.id,
                users.username,
                subjects.name as subject,
                semesters.name as semester,
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
            """, 
            conn
        )
        return results
    except Exception as e:
        # If the join fails, return an empty DataFrame with the expected columns
        columns = ['id', 'username', 'subject', 'semester', 'score', 'total_questions', 'completed_at', 'time_taken']
        return pd.DataFrame(columns=columns)
    finally:
        conn.close()
def get_user_results(user_id):
    conn = sqlite3.connect('trivia_quiz.db')
    results = pd.read_sql_query(
        """
        SELECT 
            results.id,
            subjects.name as subject,
            semesters.name as semester,
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
        """, 
        conn,
        params=(user_id,)
    )
    conn.close()
    return results

def generate_quiz(subject_id, subject_name):
    # Initialize Groq LLM
    groq_api_key ="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"
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
    
    Your questions should be academically rigorous, clear, and directly related to the subject.
    
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
        response = chain.run(subject=subject_name)
        quiz_data = pd.read_json(response)
        
        # Store quiz in database
        quiz_id = str(uuid.uuid4())
        conn = sqlite3.connect('trivia_quiz.db')
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO quizzes (id, subject_id, created_at) VALUES (?, ?, ?)",
            (quiz_id, subject_id, datetime.now())
        )
        
        for _, row in quiz_data.iterrows():
            cursor.execute(
                """
                INSERT INTO questions 
                (quiz_id, question_text, option_a, option_b, option_c, option_d, correct_answer) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quiz_id, 
                    row['question_text'], 
                    row['option_a'], 
                    row['option_b'], 
                    row['option_c'], 
                    row['option_d'], 
                    row['correct_answer']
                )
            )
        
        conn.commit()
        conn.close()
        
        return {"quiz_id": quiz_id, "questions": quiz_data.to_dict('records')}
    
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def get_quiz_questions(quiz_id):
    conn = sqlite3.connect('trivia_quiz.db')
    questions = pd.read_sql_query(
        "SELECT * FROM questions WHERE quiz_id = ?", 
        conn, 
        params=(quiz_id,)
    )
    conn.close()
    return questions.to_dict('records')

def save_quiz_result(user_id, quiz_id, score, total_questions, time_taken):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO results 
        (user_id, quiz_id, score, total_questions, completed_at, time_taken) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, quiz_id, score, total_questions, datetime.now(), time_taken)
    )
    
    conn.commit()
    conn.close()

def update_user_role(user_id, new_role):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def add_semester(name):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO semesters (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def add_subject(name, semester_id):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO subjects (name, semester_id) VALUES (?, ?)", (name, semester_id))
    conn.commit()
    conn.close()

def delete_subject(subject_id):
    conn = sqlite3.connect('trivia_quiz.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    conn.commit()
    conn.close()

# UI Components
def display_login_register():
    st.title("🧠 Trivia Quiz")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
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
        
        if st.button("Register"):
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
    st.title(f"🧠 Welcome to Trivia Quiz, {st.session_state['username']}!")
    
    # Reset quiz state if returning to home
    if st.session_state['quiz_completed'] and st.button("Return to Home"):
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
        
        # Get semesters
        semesters = get_semesters()
        semester_options = semesters['name'].tolist()
        
        selected_semester_name = st.selectbox(
            "Select Semester", 
            options=semester_options,
            index=0 if st.session_state['selected_semester'] is None else semester_options.index(st.session_state['selected_semester'])
        )
        
        st.session_state['selected_semester'] = selected_semester_name
        
        # Get semester ID
        semester_id = semesters[semesters['name'] == selected_semester_name]['id'].iloc[0]
        
        # Get subjects for selected semester
        subjects = get_subjects_by_semester(semester_id)
        if not subjects.empty:
            subject_options = subjects['name'].tolist()
            
            selected_subject_name = st.selectbox(
                "Select Subject", 
                options=subject_options,
                index=0 if st.session_state['selected_subject'] is None else 
                (subject_options.index(st.session_state['selected_subject']) if st.session_state['selected_subject'] in subject_options else 0)
            )
            
            st.session_state['selected_subject'] = selected_subject_name
            
            # Get subject ID
            subject_id = subjects[subjects['name'] == selected_subject_name]['id'].iloc[0]
            
            if st.button("Start Quiz"):
                with st.spinner("Generating quiz questions..."):
                    quiz = generate_quiz(subject_id, selected_subject_name)
                    
                    if quiz:
                        st.session_state['quiz_started'] = True
                        st.session_state['quiz_timer'] = 600  # 10 minutes
                        st.session_state['quiz_questions'] = quiz["questions"]
                        st.session_state['current_quiz_id'] = quiz["quiz_id"]
                        st.session_state['quiz_start_time'] = time.time()
                        st.rerun()
                    else:
                        st.error("Failed to generate quiz. Please try again.")
        else:
            st.info("No subjects available for the selected semester.")
    
    # Display user's past results
    if not st.session_state['quiz_started'] and not st.session_state['quiz_completed']:
        st.subheader("Your Previous Results")
        user_results = get_user_results(st.session_state['user_id'])
        
        if not user_results.empty:
            # Format timestamps
            user_results['completed_at'] = pd.to_datetime(user_results['completed_at'])
            user_results['formatted_date'] = user_results['completed_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Format time taken
            user_results['time_taken_min'] = user_results['time_taken'] // 60
            user_results['time_taken_sec'] = user_results['time_taken'] % 60
            user_results['formatted_time'] = user_results.apply(
                lambda x: f"{x['time_taken_min']}m {x['time_taken_sec']}s", axis=1
            )
            
            # Calculate percentage
            user_results['percentage'] = (user_results['score'] / user_results['total_questions'] * 100).round(1)
            
            # Display results in a table
            st.dataframe(
                user_results[['formatted_date', 'subject', 'semester', 'score', 'total_questions', 'percentage', 'formatted_time']]
                .rename(columns={
                    'formatted_date': 'Date', 
                    'subject': 'Subject', 
                    'semester': 'Semester', 
                    'score': 'Score', 
                    'total_questions': 'Total', 
                    'percentage': 'Percentage', 
                    'formatted_time': 'Time Taken'
                })
            )
        else:
            st.info("You haven't taken any quizzes yet.")

# Replace the radio button code in the display_quiz() function with this updated version:

def display_quiz():
    st.title(f"Quiz: {st.session_state['selected_subject']}")
    
    # Display timer
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    remaining_time = max(0, 600 - elapsed_time)  # 10 minutes in seconds
    
    minutes = remaining_time // 60
    seconds = remaining_time % 60
    
    st.markdown(f"### ⏱️ Time remaining: {minutes:02d}:{seconds:02d}")
    
    # Create a progress bar for the timer
    progress = st.progress(remaining_time / 600)
    
    # Auto-submit when time is up
    if remaining_time <= 0 and not st.session_state['quiz_completed']:
        st.warning("Time's up! Your quiz will be submitted automatically.")
        submit_quiz()
        st.rerun()
    
    # Display questions
    st.subheader("Questions")
    
    questions = st.session_state['quiz_questions']
    
    for i, question in enumerate(questions):
        st.markdown(f"**Question {i+1}**: {question['question_text']}")
        
        # Use a unique key for each question
        answer_key = f"q_{i}"
        options = {
            "A": question['option_a'],
            "B": question['option_b'],
            "C": question['option_c'],
            "D": question['option_d']
        }
        
        # Get selected option from session state if it exists
        selected_option = None
        if i in st.session_state['user_answers']:
            selected_option = st.session_state['user_answers'][i]
        
        # Set index for radio button
        selected_index = None
        if selected_option and selected_option in list(options.keys()):
            selected_index = list(options.keys()).index(selected_option)
        
        # Create radio button with proper index
        selected_option = st.radio(
            "Select your answer:",
            options=list(options.keys()),
            format_func=lambda x: f"{x}: {options[x]}",
            key=answer_key,
            index=selected_index
        )
        
        if selected_option:
            st.session_state['user_answers'][i] = selected_option
        
        st.divider()
    
    # Submit button
    if st.button("Submit Quiz"):
        submit_quiz()
        st.rerun()
    
    # Update timer progress bar
    progress.progress(remaining_time / 600)
def submit_quiz():
    questions = st.session_state['quiz_questions']
    user_answers = st.session_state['user_answers']
    
    score = 0
    for i, question in enumerate(questions):
        if i in user_answers and user_answers[i] == question['correct_answer']:
            score += 1
    
    # Calculate time taken
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    time_taken = min(elapsed_time, 600)  # Cap at 10 minutes
    
    # Save result to database
    save_quiz_result(
        st.session_state['user_id'], 
        st.session_state['current_quiz_id'], 
        score, 
        len(questions), 
        time_taken
    )
    
    # Update session state
    st.session_state['quiz_completed'] = True
    st.session_state['quiz_started'] = False
    st.session_state['quiz_score'] = score
    st.session_state['quiz_total'] = len(questions)
    st.session_state['quiz_time_taken'] = time_taken

def display_quiz_results():
    st.title("Quiz Results")
    
    score = st.session_state['quiz_score']
    total = st.session_state['quiz_total']
    percentage = (score / total) * 100
    
    # Display score
    st.markdown(f"### Your Score: {score}/{total} ({percentage:.1f}%)")
    
    # Display time taken
    time_taken = st.session_state['quiz_time_taken']
    minutes = time_taken // 60
    seconds = time_taken % 60
    st.markdown(f"### Time Taken: {minutes:02d}:{seconds:02d}")
    
    # Display detailed results
    st.subheader("Detailed Results")
    
    questions = st.session_state['quiz_questions']
    user_answers = st.session_state['user_answers']
    
    for i, question in enumerate(questions):
        user_answer = user_answers.get(i, "Not answered")
        correct_answer = question['correct_answer']
        
        if user_answer == correct_answer:
            st.markdown(f"**Question {i+1}**: ✅ Correct")
        else:
            st.markdown(f"**Question {i+1}**: ❌ Incorrect")
        
        st.markdown(f"- **Question**: {question['question_text']}")
        st.markdown(f"- **Your Answer**: {user_answer if user_answer != 'Not answered' else '❓'} ({question[f'option_{user_answer.lower()}'] if user_answer != 'Not answered' else 'Not provided'})")
        st.markdown(f"- **Correct Answer**: {correct_answer} ({question[f'option_{correct_answer.lower()}']})")
        st.divider()
    
    if st.button("Return to Home"):
        st.session_state['quiz_completed'] = False
        st.session_state['quiz_questions'] = []
        st.session_state['user_answers'] = {}
        st.session_state['current_quiz_id'] = None
        st.session_state['selected_semester'] = None
        st.session_state['selected_subject'] = None
        st.rerun()

def display_admin_panel():
    st.title("Admin Panel")
    
    admin_tabs = st.tabs(["User Management", "Quiz Management", "Results Management"])
    
    with admin_tabs[0]:
        st.subheader("User Management")
        
        # Add new user
        with st.expander("Add New User"):
            new_username = st.text_input("Username", key="admin_new_username")
            new_password = st.text_input("Password", type="password", key="admin_new_password")
            new_role = st.selectbox("Role", ["user", "admin"], key="admin_new_role")
            
            if st.button("Add User"):
                if not new_username or not new_password:
                    st.error("Username and password are required")
                else:
                    user = register_user(new_username, new_password, new_role)
                    if user:
                        st.success(f"User '{new_username}' added successfully")
                        st.rerun()
                    else:
                        st.error("Username already exists")
        
        # Display and manage users
        users = get_all_users()
        
        if not users.empty:
            st.dataframe(users)
            
            # User actions
            col1, col2 = st.columns(2)
            
            with col1:
                user_to_modify = st.selectbox(
                    "Select User to Modify",
                    options=users['username'].tolist(),
                    key="user_to_modify"
                )
                
                # Get user ID
                user_id = users[users['username'] == user_to_modify]['id'].iloc[0]
                current_role = users[users['username'] == user_to_modify]['role'].iloc[0]
                
            with col2:
                new_role = st.selectbox(
                    "New Role",
                    options=["user", "admin"],
                    index=0 if current_role == "user" else 1,
                    key="new_role"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Update Role"):
                    update_user_role(user_id, new_role)
                    st.success(f"User '{user_to_modify}' updated to role '{new_role}'")
                    st.rerun()
            
            with col2:
                if st.button("Delete User"):
                    if user_to_modify == st.session_state['username']:
                        st.error("You cannot delete your own account")
                    else:
                        delete_user(user_id)
                        st.success(f"User '{user_to_modify}' deleted")
                        st.rerun()
        else:
            st.info("No users found")
    
    with admin_tabs[1]:
        st.subheader("Quiz Management")
        
        # Add new semester
        with st.expander("Add New Semester"):
            new_semester = st.text_input("Semester Name", key="new_semester")
            
            if st.button("Add Semester"):
                if not new_semester:
                    st.error("Semester name is required")
                else:
                    add_semester(new_semester)
                    st.success(f"Semester '{new_semester}' added successfully")
                    st.rerun()
        
        # Add new subject
        with st.expander("Add New Subject"):
            # Get semesters for dropdown
            semesters = get_semesters()
            semester_options = semesters['name'].tolist()
            
            selected_semester = st.selectbox("Select Semester", options=semester_options, key="subject_semester")
            semester_id = semesters[semesters['name'] == selected_semester]['id'].iloc[0]
            
            new_subject = st.text_input("Subject Name", key="new_subject")
            
            if st.button("Add Subject"):
                if not new_subject:
                    st.error("Subject name is required")
                else:
                    add_subject(new_subject, semester_id)
                    st.success(f"Subject '{new_subject}' added to '{selected_semester}' successfully")
                    st.rerun()
        
        # Display and manage subject
        # Display and manage subjects
        subjects = get_all_subjects()
        
        if not subjects.empty:
            st.dataframe(subjects)
            
            # Subject actions
            subject_to_delete = st.selectbox(
                "Select Subject to Delete",
                options=subjects['name'].tolist(),
                key="subject_to_delete"
            )
            
            # Get subject ID
            subject_id = subjects[subjects['name'] == subject_to_delete]['id'].iloc[0]
            
            if st.button("Delete Subject"):
                delete_subject(subject_id)
                st.success(f"Subject '{subject_to_delete}' deleted")
                st.rerun()
        else:
            st.info("No subjects found")
    
    
         
# Add this code to the admin_tabs[2] section to debug the results issue
    
    with admin_tabs[2]:
        st.subheader("Results Management")
    
    # Debug information
        st.subheader("Debugging Information")
    
    # Check if results table has data
        conn = sqlite3.connect('trivia_quiz.db')
        cursor = conn.cursor()
    
    # Check raw results table
        cursor.execute("SELECT COUNT(*) FROM results")
        results_count = cursor.fetchone()[0]
        st.write(f"Number of records in results table: {results_count}")
    
        if results_count > 0:
        # If there are results, check each join separately
            cursor.execute("""
                SELECT r.id, r.user_id, r.quiz_id, r.score, r.total_questions, r.completed_at, r.time_taken,
                   u.username
                FROM results r
                LEFT JOIN users u ON r.user_id = u.id
                LIMIT 5
            """)
            results_users = cursor.fetchall()
            st.write("Sample results with users join:")
            st.write(results_users)
        
            cursor.execute("""
            SELECT r.id, r.quiz_id, q.subject_id
            FROM results r
            LEFT JOIN quizzes q ON r.quiz_id = q.id
            LIMIT 5
        """)
            results_quizzes = cursor.fetchall()
            st.write("Sample results with quizzes join:")
            st.write(results_quizzes)
        
            if len(results_quizzes) > 0:
            # Check subject data
                sample_subject_id = results_quizzes[0][2]
                cursor.execute("""
                SELECT s.id, s.name, s.semester_id
                FROM subjects s
                WHERE s.id = ?
            """, (sample_subject_id,))
                subject_data = cursor.fetchone()
                st.write(f"Subject data for ID {sample_subject_id}:")
                st.write(subject_data)
            
                if subject_data:
                # Check semester data
                    semester_id = subject_data[2]
                    cursor.execute("""
                        SELECT id, name
                    FROM semesters
                    WHERE id = ?
                """, (semester_id,))
                    semester_data = cursor.fetchone()
                    st.write(f"Semester data for ID {semester_id}:")
                    st.write(semester_data)
    
    # Try a simpler query
        st.subheader("Simplified Results")
        try:
            simple_results = pd.read_sql_query(
            """
            SELECT 
                results.id,
                results.user_id,
                results.quiz_id,
                results.score,
                results.total_questions,
                results.completed_at,
                results.time_taken
            FROM results
            ORDER BY results.completed_at DESC
            """, 
            conn
        )
            st.write("Raw results data:")
            st.dataframe(simple_results)
        except Exception as e:
            st.error(f"Error executing simple query: {e}")
    
        conn.close()
    
    # Continue with the original code...
    # Display all results
        results = get_all_results()
    
        if not results.empty:
            st.success("Results retrieved successfully!")
        # Rest of the code remains the same...
def logout():
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    st.session_state['role'] = None
    st.session_state['quiz_started'] = False
    st.session_state['quiz_completed'] = False
    st.session_state['quiz_questions'] = []
    st.session_state['user_answers'] = {}
    st.rerun()

# Main application
def main():
    # Sidebar
    if st.session_state['username']:
        st.sidebar.title(f"Hello, {st.session_state['username']}")
        st.sidebar.markdown(f"**Role**: {st.session_state['role'].capitalize()}")
        
        if st.sidebar.button("Logout"):
            logout()
    
    # Main content
    if not st.session_state['user_id']:
        display_login_register()
    else:
        if st.session_state['role'] == 'admin':
            display_admin_panel()
        else:
            if st.session_state['quiz_started']:
                display_quiz()
            elif st.session_state['quiz_completed']:
                display_quiz_results()
            else:
                display_user_home()

if __name__ == "__main__":
    main()
