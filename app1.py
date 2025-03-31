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

# Set page configuration
st.set_page_config(
    page_title="Academic Quiz Generator",
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

# Database setup
def init_db():
    conn = sqlite3.connect('academic_quiz.db')
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
    
    # Insert initial admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        admin_id = str(uuid.uuid4())
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                      (admin_id, 'admin', password_hash, 'admin'))
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        return {"id": user[0], "username": username, "role": user[2]}
    return None

def register_user(username, password, role="user"):
    conn = sqlite3.connect('academic_quiz.db')
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
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM semesters ORDER BY name")
    semesters = cursor.fetchall()
    conn.close()
    
    if not semesters:
        return pd.DataFrame(columns=['id', 'name'])
    
    return pd.DataFrame(semesters, columns=['id', 'name'])

def get_subjects_by_semester(semester_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    
    try:
        # Make sure semester_id is an integer
        semester_id = int(semester_id)
        
        # Debugging - Print the SQL query parameters
        print(f"Fetching subjects for semester_id: {semester_id}")
        
        # Execute the query with parameter binding
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
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT subjects.id, subjects.name, semesters.name as semester 
        FROM subjects 
        JOIN semesters ON subjects.semester_id = semesters.id 
        ORDER BY semesters.name, subjects.name
    """)
    subjects = cursor.fetchall()
    conn.close()
    
    if not subjects:
        return pd.DataFrame(columns=['id', 'name', 'semester'])
    
    return pd.DataFrame(subjects, columns=['id', 'name', 'semester'])

def get_all_users():
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users ORDER BY username")
    users = cursor.fetchall()
    conn.close()
    
    if not users:
        return pd.DataFrame(columns=['id', 'username', 'role'])
    
    return pd.DataFrame(users, columns=['id', 'username', 'role'])

def get_all_results():
    conn = sqlite3.connect('academic_quiz.db')
    try:
        cursor = conn.cursor()
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
    conn = sqlite3.connect('academic_quiz.db')
    try:
        cursor = conn.cursor()
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
                conn = sqlite3.connect('academic_quiz.db')
                cursor = conn.cursor()
                
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
                conn.close()
                
                return {"quiz_id": quiz_id, "questions": quiz_data}
                
            except json.JSONDecodeError:
                st.error("Failed to parse AI response as JSON.")
                return None
    
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def get_quiz_questions(quiz_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
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
    conn.close()
    
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

def save_quiz_result(user_id, quiz_id, score, total_questions, time_taken):
    conn = sqlite3.connect('academic_quiz.db')
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
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def add_semester(name):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO semesters (name) VALUES (?)", (name,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def add_subject(name, semester_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO subjects (name, semester_id) VALUES (?, ?)", (name, semester_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding subject: {e}")
        return False
    finally:
        conn.close()

def delete_subject(subject_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
    conn.commit()
    conn.close()

def delete_semester(semester_id):
    conn = sqlite3.connect('academic_quiz.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM semesters WHERE id = ?", (semester_id,))
    conn.commit()
    conn.close()

# UI Components
def display_login_register():
    st.title("🧠 Academic Quiz Generator")
    
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
    st.title(f"🧠 Welcome to Academic Quiz Generator, {st.session_state['username']}!")
    
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
            index=selected_semester_index
        )
        
        # Update selected semester in session state
        if st.session_state['selected_semester'] != selected_semester_name:
            st.session_state['selected_semester'] = selected_semester_name
            # Reset subject selection when semester changes
            st.session_state['selected_subject'] = None
            # No need for st.rerun() here as we continue in the function
        else:
            st.session_state['selected_semester'] = selected_semester_name
        
        # Get semester ID - Ensure it's an integer
        semester_id = int(semesters[semesters['name'] == selected_semester_name]['id'].iloc[0])
        
        # Debugging - Print semester_id to console
        print(f"Selected semester ID: {semester_id}")
        
        # Get subjects for selected semester - Always fetch fresh data
        subjects = get_subjects_by_semester(semester_id)
        
        # Debugging - Print subjects DataFrame
        print(f"Subjects DataFrame: {subjects}")
        
        if subjects.empty:
            st.info("No subjects available for the selected semester. Please ask an administrator to add subjects.")
            
            # Display user's past results
            display_user_results()
            return
        
        subject_options = subjects['name'].tolist()
        
        # Debugging - Print subject options
        print(f"Subject options: {subject_options}")
        
        # Default to first subject if none selected or if previous selection is not in current list
        selected_subject_index = 0
        if st.session_state['selected_subject'] is not None:
            try:
                selected_subject_index = subject_options.index(st.session_state['selected_subject'])
            except ValueError:
                selected_subject_index = 0
        
        selected_subject_name = st.selectbox(
            "Select Subject", 
            options=subject_options,
            index=selected_subject_index
        )
        
        # Update selected subject in session state
        st.session_state['selected_subject'] = selected_subject_name
        
        # Get subject ID
        subject_id = int(subjects[subjects['name'] == selected_subject_name]['id'].iloc[0])
        
        # Debugging - Print subject ID
        print(f"Selected subject ID: {subject_id}")
        
        if st.button("Start Quiz"):
            quiz = generate_quiz(subject_id, selected_subject_name)
            
            if quiz:
                st.session_state['quiz_started'] = True
                st.session_state['quiz_timer'] = 300  # 5 minutes
                st.session_state['quiz_questions'] = quiz["questions"]
                st.session_state['current_quiz_id'] = quiz["quiz_id"]
                st.session_state['quiz_start_time'] = time.time()
                st.session_state['user_answers'] = {}  # Clear previous answers
                st.rerun()
            else:
                st.error("Failed to generate quiz. Please try again.")
    
    # Display user's past results
    if not st.session_state['quiz_started'] and not st.session_state['quiz_completed']:
        display_user_results()
def display_user_results():
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

def display_quiz():
    st.title(f"Quiz: {st.session_state['selected_subject']}")
    
    # Display timer
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    remaining_time = max(0, 300 - elapsed_time)  # 5 minutes in seconds
    
    minutes = remaining_time // 60
    seconds = remaining_time % 60
    
    st.markdown(f"### ⏱️ Time remaining: {minutes:02d}:{seconds:02d}")
    
    # Create a progress bar for the timer
    progress = st.progress(remaining_time / 300)
    
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
    progress.progress(remaining_time / 300)

def submit_quiz():
    questions = st.session_state['quiz_questions']
    user_answers = st.session_state['user_answers']
    
    score = 0
    for i, question in enumerate(questions):
        if i in user_answers and user_answers[i] == question['correct_answer']:
            score += 1
    
    # Calculate time taken
    elapsed_time = int(time.time() - st.session_state['quiz_start_time'])
    time_taken = min(elapsed_time, 300)  # Cap at 5 minutes
    
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
        
        # User's answer
        if user_answer != "Not answered":
            user_option = question[f'option_{user_answer.lower()}']
            st.markdown(f"- **Your Answer**: {user_answer} ({user_option})")
        else:
            st.markdown(f"- **Your Answer**: ❓ Not provided")
        
        # Correct answer
        correct_option = question[f'option_{correct_answer.lower()}']
        st.markdown(f"- **Correct Answer**: {correct_answer} ({correct_option})")
        st.divider()
    
    if st.button("Return to Home"):
        st.session_state['quiz_started'] = False
        st.session_state['quiz_completed'] = False
        st.session_state['quiz_questions'] = []
        st.session_state['user_answers'] = {}
        st.session_state['current_quiz_id'] = None
        st.session_state['selected_semester'] = None
        st.session_state['selected_subject'] = None
        st.rerun()

def display_admin_panel():
    st.title("🧠 Admin Panel")
    
    tabs = st.tabs(["User Management", "Semester/Subject Management", "Results"])
    
    # User Management Tab
    with tabs[0]:
        st.subheader("User Management")
        
        users_df = get_all_users()
        
        if not users_df.empty:
            # Exclude current user from the list for safety
            users_df = users_df[users_df['id'] != st.session_state['user_id']]
            
            if not users_df.empty:
                # Create selectbox for users
                selected_user_id = st.selectbox(
                    "Select User",
                    options=users_df['id'].tolist(),
                    format_func=lambda x: f"{users_df[users_df['id'] == x]['username'].iloc[0]} ({users_df[users_df['id'] == x]['role'].iloc[0]})"
                )
                
                selected_user = users_df[users_df['id'] == selected_user_id].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    new_role = st.selectbox(
                        "Role",
                        options=["user", "admin"],
                        index=0 if selected_user['role'] == "user" else 1
                    )
                    
                    if st.button("Update Role"):
                        update_user_role(selected_user_id, new_role)
                        st.success(f"Updated {selected_user['username']}'s role to {new_role}")
                        st.rerun()
                
                with col2:
                    if st.button("Delete User", type="primary", help="This action cannot be undone"):
                        delete_user(selected_user_id)
                        st.success(f"Deleted user {selected_user['username']}")
                        st.rerun()
            else:
                st.info("No other users to manage")
        else:
            st.info("No users to display")
    
    # Semester/Subject Management Tab
    with tabs[1]:
        st.subheader("Semester/Subject Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Add Semester")
            new_semester = st.text_input("Semester Name (e.g., 'Semester 1', 'First Year')")
            
            if st.button("Add Semester"):
                if new_semester:
                    if add_semester(new_semester):
                        st.success(f"Added semester: {new_semester}")
                        st.rerun()
                    else:
                        st.error(f"Semester '{new_semester}' already exists")
                else:
                    st.error("Please enter a semester name")
        
        with col2:
            st.markdown("### Add Subject")
            
            # Get semesters for dropdown
            semesters = get_semesters()
            
            if not semesters.empty:
                semester_id = st.selectbox(
                    "Select Semester",
                    options=semesters['id'].tolist(),
                    format_func=lambda x: semesters[semesters['id'] == x]['name'].iloc[0]
                )
                
                new_subject = st.text_input("Subject Name")
                
                if st.button("Add Subject"):
                    if new_subject:
                        if add_subject(new_subject, semester_id):
                            st.success(f"Added subject: {new_subject}")
                            st.rerun()
                        else:
                            st.error(f"Failed to add subject '{new_subject}'")
                    else:
                        st.error("Please enter a subject name")
            else:
                st.warning("You need to create at least one semester first")
        
        st.divider()
        
        # Display existing semesters and subjects
        st.markdown("### Existing Semesters and Subjects")
        
        semesters = get_semesters()
        if not semesters.empty:
            for _, semester in semesters.iterrows():
                with st.expander(f"Semester: {semester['name']}"):
                    subjects = get_subjects_by_semester(semester['id'])
                    
                    if not subjects.empty:
                        for i, subject in subjects.iterrows():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"• {subject['name']}")
                            
                            with col2:
                                if st.button("Delete", key=f"del_subj_{subject['id']}"):
                                    delete_subject(subject['id'])
                                    st.success(f"Deleted subject: {subject['name']}")
                                    st.rerun()
                    else:
                        st.info("No subjects in this semester")
                    
                    if st.button("Delete Semester", key=f"del_sem_{semester['id']}"):
                        delete_semester(semester['id'])
                        st.success(f"Deleted semester: {semester['name']}")
                        st.rerun()
        else:
            st.info("No semesters created yet")
    
    # Results Tab
    with tabs[2]:
        st.subheader("Quiz Results")
        
        results_df = get_all_results()
        
        if not results_df.empty:
            # Format timestamps
            results_df['completed_at'] = pd.to_datetime(results_df['completed_at'])
            results_df['formatted_date'] = results_df['completed_at'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Format time taken
            results_df['time_taken_min'] = results_df['time_taken'] // 60
            results_df['time_taken_sec'] = results_df['time_taken'] % 60
            results_df['formatted_time'] = results_df.apply(
                lambda x: f"{x['time_taken_min']}m {x['time_taken_sec']}s", axis=1
            )
            
            # Calculate percentage
            results_df['percentage'] = (results_df['score'] / results_df['total_questions'] * 100).round(1)
            
            # Display results in a table
            st.dataframe(
                results_df[['username', 'subject', 'semester', 'score', 'total_questions', 'percentage', 'formatted_time', 'formatted_date']]
                .rename(columns={
                    'username': 'User',
                    'subject': 'Subject', 
                    'semester': 'Semester', 
                    'score': 'Score', 
                    'total_questions': 'Total', 
                    'percentage': 'Percentage', 
                    'formatted_time': 'Time Taken',
                    'formatted_date': 'Date'
                })
            )
        else:
            st.info("No quiz results yet")

# Main App Logic
def main():
    # Set custom app styles
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton button {
            border-radius: 5px;
        }
        .stProgress > div > div {
            background-color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Show logout button if user is logged in
    if st.session_state['user_id']:
        if st.sidebar.button("Logout"):
            st.session_state['user_id'] = None
            st.session_state['username'] = None
            st.session_state['role'] = None
            st.session_state['quiz_started'] = False
            st.session_state['quiz_completed'] = False
            st.session_state['quiz_questions'] = []
            st.session_state['user_answers'] = {}
            st.session_state['current_quiz_id'] = None
            st.session_state['selected_semester'] = None
            st.session_state['selected_subject'] = None
            st.rerun()
    
    # Main application logic
    if not st.session_state['user_id']:
        # Not logged in, show login/register screen
        display_login_register()
    else:
        # Logged in
        if st.session_state['role'] == 'admin':
            display_admin_panel()
        else:
            # Regular user
            if st.session_state['quiz_started']:
                display_quiz()
            elif st.session_state['quiz_completed']:
                display_quiz_results()
            else:
                display_user_home()

if __name__ == "__main__":
    main()
