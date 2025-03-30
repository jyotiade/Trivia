# Trivia Quiz Application
A Streamlit web application that allows students to take quizzes and provides an administrative panel for managing users, quizzes, and results.

## Features

- **User Authentication**: Secure user login and registration
- **Quiz Generation**: AI-powered quiz questions based on selected subjects
- **User Interface**: Clean, intuitive interface with timer and result display
- **Admin Panel**: Comprehensive management of users, subjects, questions, and results
- **Data Storage**: Local SQLite database for storing all application data

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Groq API key (create one at https://console.groq.com)

### Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your Groq API key as an environment variable:

```bash
# Linux/macOS
export GROQ_API_KEY="your-groq-api-key"

# Windows
set GROQ_API_KEY=your-groq-api-key
```

### Running the Application

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open a web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

### Default Login Credentials

- Admin User:
  - Username: admin
  - Password: admin

## Usage

### Student Users

1. Register a new account or log in with existing credentials
2. Navigate to the home page to see available semesters and subjects
3. Select a subject to start a 10-minute quiz
4. Answer the questions and submit before the timer runs out
5. View your results and review your answers
6. Check the leaderboard to see how you compare to other users
7. View your quiz history to track your progress

### Admin Users

1. Log in with admin credentials
2. Use the navigation sidebar to access admin features:
   - **User Management**: Add, edit, or delete user accounts
   - **Subject Management**: Add or remove semesters and subjects
   - **Question Management**: Add, edit, or delete questions; generate new questions with AI
   - **Results View**: View and analyze quiz results for all users

## System Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with SQLite database
- **AI Integration**: Langchain with Groq LLM for generating quiz questions

## Customization

- Modify the default subjects and semesters in the `init_db()` function
- Adjust the quiz duration by changing the `quiz_time_limit` variable
- Customize the UI appearance using Streamlit's theming capabilities
