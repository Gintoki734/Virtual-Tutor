#The Program was build using known knowledge of the languages used and the following references
#https://www.youtube.com/watch?app=desktop&v=bluclMxiUkA
#https://docs.python.org/3/library/subprocess.html
#https://github.com/fivestarspicy/chat-with-notes/blob/main/app.py
#https://www.analyticsvidhya.com/blog/2024/04/how-to-access-llama3-with-flask/
#https://medium.com/@vishal_007/build-a-chatbot-with-custom-data-sources-using-llamaindex-and-openai-f70c6841cd43
#https://stackoverflow.com/questions/73545218/utf-8-encoding-exception-with-subprocess-run


from flask import Flask, render_template, request, jsonify
import os
import markdown
import json
import pandas as pd
import subprocess  #Used to call local commands

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import time

app = Flask(__name__)

process = None

# File paths where the data will be stored
LOG_FILE = "user_page_time.csv"
LOG_QUIZ = "user_quiz_performance.csv"
LOG_LLM = "user_llm_time.csv"

# Define available courses - https://www.w3schools.com/python/python_dictionaries.asp
COURSES = {
    "Talking about yourself": {"content": "data/Talking about yourself.txt", "quiz": "data/Talking about yourself.json"},
    "Talking about family": {"content": "data/Talking about family.txt", "quiz": "data/Talking about family.json"},
    "Animals and pets": {"content": "data/Animals and pets.txt", "quiz": "data/Animals and pets.json"},
    "Classroom language": {"content": "data/Classroom language.txt", "quiz": "data/Classroom language.json"},
    "Weather and seasons": {"content": "data/Weather and seasons.txt", "quiz": "data/Weather and seasons.json"},
    "Numbers and numeracy": {"content": "data/Numbers and numeracy.txt", "quiz": "data/Numbers and numeracy.json"},
    "Days, months and dates": {"content": "data/Days, months and dates.txt", "quiz": "data/Days, months and dates.json"},
    "Colours": {"content": "data/Colours.txt", "quiz": "data/Colours.json"},
    "Clothes": {"content": "data/Clothes.txt", "quiz": "data/Clothes.json"},
    "Food and drink": {"content": "data/Food and drink.txt", "quiz": "data/Food and drink.json"},
    "Exploring the town": {"content": "data/Exploring the town.txt", "quiz": "data/Exploring the town.json"}
}

#Fuzzy Logic
# Variables to judge the whole quiz performance
quiz_score = ctrl.Antecedent(np.arange(0, 5, 1), 'quiz_score') 
total_time_on_quiz = ctrl.Antecedent(np.arange(0, 71, 1), 'total_time_on_quiz')  
total_hesitance = ctrl.Antecedent(np.arange(0, 12, 1), 'hesitant')  

# Defining the output variable
result = ctrl.Consequent(np.arange(0, 101, 1), 'result') 

# Defining membership functions for quiz_score
quiz_score['low'] = fuzz.trimf(quiz_score.universe, [0, 0, 2])
quiz_score['medium'] = fuzz.trimf(quiz_score.universe, [1, 2, 3])
quiz_score['high'] = fuzz.trimf(quiz_score.universe, [2, 4, 4]) 

# Defining membership functions for total_time_on_quiz
total_time_on_quiz['low'] = fuzz.trimf(total_time_on_quiz.universe, [0, 0, 30])
total_time_on_quiz['medium'] = fuzz.trimf(total_time_on_quiz.universe, [20, 40, 60])
total_time_on_quiz['high'] = fuzz.trimf(total_time_on_quiz.universe, [50, 70, 70])

# Defining Membership functions for total_hesitation
total_hesitance['low'] = fuzz.trimf(total_hesitance.universe, [0, 0, 5])
total_hesitance['medium'] = fuzz.trimf(total_hesitance.universe, [4, 6, 8])
total_hesitance['high'] = fuzz.trimf(total_hesitance.universe, [7, 9, 11])


# Defining membership functions for result
result['low'] = fuzz.trimf(result.universe, [0, 0, 40])
result['medium'] = fuzz.trimf(result.universe, [30, 50, 70])
result['high'] = fuzz.trimf(result.universe, [60, 80, 100])

# Low range rules
rule1 = ctrl.Rule(quiz_score['low'] & total_hesitance['low'] & total_time_on_quiz['low'], result['low'])
rule2 = ctrl.Rule(quiz_score['low'] & total_hesitance['low'] & total_time_on_quiz['medium'], result['low'])
rule3 = ctrl.Rule(quiz_score['low'] & total_hesitance['low'] & total_time_on_quiz['high'], result['low'])
rule4 = ctrl.Rule(quiz_score['low'] & total_hesitance['medium'] & total_time_on_quiz['low'], result['low'])
rule5 = ctrl.Rule(quiz_score['low'] & total_hesitance['medium'] & total_time_on_quiz['medium'], result['low'])
rule6 = ctrl.Rule(quiz_score['low'] & total_hesitance['medium'] & total_time_on_quiz['high'], result['low'])
rule7 = ctrl.Rule(quiz_score['low'] & total_hesitance['high'] & total_time_on_quiz['low'], result['low'])
rule8 = ctrl.Rule(quiz_score['low'] & total_hesitance['high'] & total_time_on_quiz['medium'], result['low'])
rule9 = ctrl.Rule(quiz_score['low'] & total_hesitance['high'] & total_time_on_quiz['high'], result['low'])

# Medium range rules
rule10 = ctrl.Rule(quiz_score['medium'] & total_hesitance['low'] & total_time_on_quiz['low'], result['medium'])
rule11 = ctrl.Rule(quiz_score['medium'] & total_hesitance['low'] & total_time_on_quiz['medium'], result['medium'])
rule12 = ctrl.Rule(quiz_score['medium'] & total_hesitance['low'] & total_time_on_quiz['high'], result['medium'])
rule13 = ctrl.Rule(quiz_score['medium'] & total_hesitance['medium'] & total_time_on_quiz['low'], result['medium'])
rule14 = ctrl.Rule(quiz_score['medium'] & total_hesitance['medium'] & total_time_on_quiz['medium'], result['medium'])
rule15 = ctrl.Rule(quiz_score['medium'] & total_hesitance['medium'] & total_time_on_quiz['high'], result['medium'])
rule16 = ctrl.Rule(quiz_score['medium'] & total_hesitance['high'] & total_time_on_quiz['low'], result['medium'])
rule17 = ctrl.Rule(quiz_score['medium'] & total_hesitance['high'] & total_time_on_quiz['medium'], result['medium'])
rule18 = ctrl.Rule(quiz_score['medium'] & total_hesitance['high'] & total_time_on_quiz['high'], result['medium'])

# High range rules
rule19 = ctrl.Rule(quiz_score['high'] & total_hesitance['low'] & total_time_on_quiz['low'], result['high'])
rule20 = ctrl.Rule(quiz_score['high'] & total_hesitance['low'] & total_time_on_quiz['medium'], result['high'])
rule21 = ctrl.Rule(quiz_score['high'] & total_hesitance['low'] & total_time_on_quiz['high'], result['high'])
rule22 = ctrl.Rule(quiz_score['high'] & total_hesitance['medium'] & total_time_on_quiz['low'], result['high'])
rule23 = ctrl.Rule(quiz_score['high'] & total_hesitance['medium'] & total_time_on_quiz['medium'], result['high'])
rule24 = ctrl.Rule(quiz_score['high'] & total_hesitance['medium'] & total_time_on_quiz['high'], result['high'])
rule25 = ctrl.Rule(quiz_score['high'] & total_hesitance['high'] & total_time_on_quiz['low'], result['high'])
rule26 = ctrl.Rule(quiz_score['high'] & total_hesitance['high'] & total_time_on_quiz['medium'], result['high'])
rule27 = ctrl.Rule(quiz_score['high'] & total_hesitance['high'] & total_time_on_quiz['high'], result['medium'])


# Create the fuzzy inference system
result_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,rule21,rule22,rule23,rule24,rule25,rule26, rule27])
result_simulation = ctrl.ControlSystemSimulation(result_ctrl)

# Function to give feedback based on their result at the end
@app.route('/total_results', methods=['POST'])
def total_results():
    try:
        # Store the received values
        data = request.get_json()
        total_time = float(data.get('total_time', 0.0))
        score = int(data.get('total_score', 0))
        total_options_Clicked = int(data.get('optionsClicked', 0))
        total_question = data.get('totalQ', 0)

        # Input values
        result_simulation.input['quiz_score'] = score
        result_simulation.input['total_time_on_quiz'] = total_time
        result_simulation.input['hesitant'] = total_options_Clicked

        # If statements to make sure the variables stay on range
        if score < 0:
            score = 0
        elif score > 4:
            score = 4

        if total_time < 0:
            total_time = 0
        elif total_time > 70:
            total_time = 70

        if total_options_Clicked < 0:
            total_options_Clicked = 0
        elif total_options_Clicked > 11:
            total_options_Clicked = 11

        # Compute the result
        result_simulation.compute()

        # If the fuzzy logic outputs an error during the calculation, the result will be a neutral of 50
        try:
            # Call fuzzy logic function and get result
            fuzzy_result = round(result_simulation.output['result'], 2)
        except Exception as err:
            fuzzy_result = 50

        # Decide on a prompt based on the result
        if fuzzy_result > 88:
            prompt = "I have performed excellent overall. Praise me in a few sentences"
        elif fuzzy_result < 33:
            prompt = f"I didn't perform well for these questions - {total_question}. Create a small course focusing in this topic."
        else:
            prompt = "I performed good overall. Motivate me to further get a better result in a few sentences"

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer, 2) # Round the it to 2 decimal point

        prompt_word_count = len(prompt) # Count the number of letters used

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken","Output_Word_Count"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "SessionID": [len(df) + 1],
            "Course": [0],
            "Feedback": [0],
            "Final_feedback": [1],
            "Prompt_word_count": [prompt_word_count],
            "Time_taken": [time_taken],
            "Output_Word_Count": [len(output)]
        })

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)

        return jsonify({"response": output})

    except Exception as e:
        print(e, "\n")
        return jsonify({"Error": str(e)}), 500

           

# Function to get feedback per answer
@app.route('/per_Q_results', methods=['POST'])
def per_Q_results():
    try:
        # Store the received values
        data = request.get_json()
        correct_answers = int(data.get('correct_answers', 0))
        options_Clicked = int(data.get('optionsClicked', 0))
        current_question = data.get('current_Q', "")
        options = data.get('C_option', [])
        answer = data.get('selected_answer', "")
        qtime = data.get('time_taken', 0)


        # If they get it right
        if correct_answers > 0:
            # They hesitated
            if (options_Clicked > 2) or (qtime > 10):
                prompt = f"I got this question correct '{current_question}' but I hesitated in these options - {options} the answer I selected was {answer}.In few words tell me i got it correct and state why the other options were wrong"
            else:
                prompt = f"I got this question correct for '{current_question}', the options were - {options} and the answer I selected was {answer}. Praise and motivate me in a few words"
        # If they get it wrong
        else:
            # They hesitated
            if (options_Clicked > 2) or (qtime > 10):
                prompt = f"I got this question incorrect '{current_question}' and I hesitated in these options - {options} the answer I selected was {answer}. In few words tell me i got it incorrect and state why the other options were wrong"
            else:
                prompt = f"I got this question incorrect '{current_question}' create a small explanation on why I got it wrong among these options - {options}, the answer I selected was {answer}"

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer,2) # Round the it to 2 decimal point

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken","Output_Word_Count"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "SessionID": [len(df) + 1],
            "Course": [0],
            "Feedback": [1],
            "Final_feedback": [0],
            "Prompt_word_count": [len(prompt)],
            "Time_taken": [time_taken],
            "Output_Word_Count": [len(output)]
        })

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)

        return jsonify({"response": output})

    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

           
#Function to interact with Ollama locally - https://github.com/ollama/ollama/issues/1474
def prompt_ollama(user_input):
    try:
        #Call Ollama using subprocess
        command = f"ollama run llama2"
        #Add the utf 8 reference here
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, encoding='utf-8')
        # Send the user input to Ollama
        process.stdin.write(user_input + "\n")
        process.stdin.flush()
        # Capture the full output after completion
        result, error = process.communicate()
        if process.returncode != 0:
            return f"Error: {error.strip()}"  # Capture the error
        else:
            return result.strip() #Capture the output from Ollama
    except Exception as e:
        return f"Exception: {str(e)}"
    

# https://stackoverflow.com/questions/47048906/convert-markdown-tables-to-html-tables-using-python
# Function to read and convert markdown content
def load_course_content(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return markdown.markdown(file.read(), extensions=["tables"])
    return "Content not available."

# Function to load quiz questions
def load_quiz(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"error": "Quiz not available."}

# Dispalys the first page
@app.route("/")
def home():
    return render_template("landingpage.html")

# Dispalys the home page
@app.route("/index")
def index():
    return render_template("index.html", courses=COURSES)

# Function to receive prompt from the user and give the output back to the user
@app.route('/send_value', methods=['POST'])
def send_value():
    try:
        # Get the value sent from the user
        prompt = request.json.get('value')

        # Record time taked for the prompt to generate an output
        start_timer = time.time()
        output = prompt_ollama(prompt) # Ask the LLM
        end_timer = time.time()

        time_taken = round(end_timer - start_timer,2) # Round the it to 2 decimal point

        columns = ["SessionID", "Course", "Feedback", "Final_feedback", "Prompt_word_count", "Time_taken","Output_Word_Count"]

        # Check if the file exists and read it, otherwise initialize an empty DataFrame
        if os.path.exists(LOG_LLM):
            df = pd.read_csv(LOG_LLM)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "SessionID": [len(df) + 1],
            "Course": [1],
            "Feedback": [0],
            "Final_feedback": [0],
            "Prompt_word_count": [len(prompt)],
            "Time_taken": [time_taken],
            "Output_Word_Count": [len(output)]
        })

        # Function to write to the csv file
        write_to_csv(LOG_LLM, new_entry, columns)
    
        return jsonify({'response': output})
    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

# Displays course content
@app.route("/course/<course_name>")
def course(course_name):
    if course_name in COURSES:
        content = load_course_content(COURSES[course_name]["content"])
        return render_template("course.html", course_name=course_name, content=content)
    return "Course not found", 404

# Displays quiz content for that specific course
@app.route("/quiz/<course_name>")
def quiz(course_name):
    if course_name in COURSES:
        quiz_data = load_quiz(COURSES[course_name]["quiz"])
        return render_template("quiz.html", course_name=course_name, quiz_data=quiz_data)
    return "Quiz not found", 404

# Writes the data collected from the course page into the csv file
@app.route('/log_time', methods=['POST'])
def log_time():
    # Handles any error occured
    try:
        # Getting the data in JSON format and retriving it and storing temporary data incase there is a void/missing data
        data = request.get_json()
        duration = data.get("duration", 0)
        course_name = data.get("courseName", "Unknown Course")
        date = data.get("date", "Unknown Date")
        time = data.get("time", "Unknown Time")

        columns = ["Session ID", "Time Spent (seconds)", "Course Name", "Date", "Time"]

        # Check if the file exists and read it otherwise initialize an empty DataFrame
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log
        new_entry = pd.DataFrame({
            "Session ID": [len(df) + 1], 
            "Time Spent (seconds)": [duration],
            "Course Name": [course_name],
            "Date": [date],
            "Time": [time]
        })

        # Function to write to the csv file
        write_to_csv(LOG_FILE,new_entry,columns)

        return jsonify({"Message": "Quiz logged"}), 200

    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

# Function to record the quiz result
@app.route('/log_quiz', methods=['POST'])
def log_quiz():
    # Handles any error occured
    try:
        # Getting the data in JSON format and retriving it and storing temporary data incase there is a void/missing data
        data = request.get_json()
        score = data.get("quiz_score", 0)
        course_name = data.get("courseName", "Unknown Course")
        date = data.get("date", "Unknown Date")
        time = data.get("time", "Unknown Time")
        optionClicked = data.get("clickCounts", [0] * 4)
        quiztime = data.get("quiztimetaken", [0] * 4)

        columns=[
                "Session ID", "Score", "Course Name", "Date", "Time",
                "OptionsClicked_Question_1", "OptionsClicked_Question_2", 
                "OptionsClicked_Question_3", "OptionsClicked_Question_4",
                "Time_Question_1", "Time_Question_2", "Time_Question_3", "Time_Question_4"
            ]

        # Check if the file exists and read it otherwise initialize an empty DataFrame
        if os.path.exists(LOG_QUIZ):
            df = pd.read_csv(LOG_QUIZ)  # Read the CSV file
        else:
            df = pd.DataFrame(columns=columns)

        # Data entry for the log            
        new_entry = pd.DataFrame({
            "Session ID": [len(df) + 1], 
            "Score": [score],
            "Course Name": [course_name],
            "Date" : [date],
            "Time": [time]
            })
        for i in range(4):
            new_entry[f"OptionsClicked_Question_{i+1}"] = optionClicked[i]
            new_entry[f"Time_Question_{i+1}"] = quiztime[i]

        # Function to write to the csv file
        write_to_csv(LOG_QUIZ, new_entry, columns)

        return jsonify({"Message": "Quiz logged"}), 200
    except Exception as e:
        print("\n",e)
        return jsonify({"Error": str(e)}), 500

# Function to write to any csv file
def write_to_csv(file_path, new_entry, columns):
    # Ensure the file exists and read it, otherwise initialize a DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)

    # Append new entry and save it to CSV
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    #Call Ollama using subprocess
    command = f"ollama run llama2"
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, encoding='utf-8')
    app.run(debug=True)

