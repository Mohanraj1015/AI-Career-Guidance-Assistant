# ============================================================
# AI CAREER GUIDANCE ASSISTANT
# Internship Project: AI-SS-005
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CAREER KNOWLEDGE BASE
# ============================================================

careers = {

    "AI / ML Engineer": {
        "description": "Build Artificial Intelligence and Machine Learning applications.",
        "skills": ["Python", "Machine Learning", "Deep Learning", "Statistics"],
        "salary": "₹6 - ₹25 LPA",
        "demand": "Very High"
    },

    "Data Scientist": {
        "description": "Analyze data and build models to find useful insights.",
        "skills": ["Python", "Statistics", "SQL", "Machine Learning"],
        "salary": "₹7 - ₹30 LPA",
        "demand": "Very High"
    },

    "Software Engineer": {
        "description": "Develop and maintain software applications.",
        "skills": ["Python", "Java", "C++", "Data Structures", "Algorithms"],
        "salary": "₹5 - ₹20 LPA",
        "demand": "High"
    },

    "Data Analyst": {
        "description": "Analyze data and create reports for business decisions.",
        "skills": ["Excel", "SQL", "Python", "Statistics", "Power BI"],
        "salary": "₹4 - ₹15 LPA",
        "demand": "High"
    },

    "Cyber Security Engineer": {
        "description": "Protect computer systems and networks from cyber attacks.",
        "skills": ["Networking", "Linux", "Python", "Cyber Security"],
        "salary": "₹5 - ₹25 LPA",
        "demand": "Very High"
    },

    "Web Developer": {
        "description": "Create websites and web applications.",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
        "salary": "₹4 - ₹18 LPA",
        "demand": "High"
    },

    "Cloud Engineer": {
        "description": "Design and manage cloud infrastructure and services.",
        "skills": ["AWS", "Azure", "Linux", "Docker", "Networking"],
        "salary": "₹6 - ₹25 LPA",
        "demand": "Very High"
    },

    "DevOps Engineer": {
        "description": "Automate software development and cloud deployment processes.",
        "skills": ["Linux", "Docker", "Kubernetes", "AWS", "CI/CD"],
        "salary": "₹6 - ₹25 LPA",
        "demand": "High"
    }
}


# ============================================================
# 2. INTENT DATA
# ============================================================

intent_data = {

    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening"
    ],

    "career_options": [
        "what careers are available",
        "career options",
        "show career paths",
        "what jobs can I choose",
        "career path"
    ],

    "recommendation": [
        "recommend a career",
        "suggest a career",
        "which career is best for me",
        "what career should I choose",
        "career recommendation"
    ],

    "skills": [
        "what skills are required",
        "skills needed",
        "required skills",
        "what should I learn"
    ],

    "salary": [
        "salary",
        "salary range",
        "how much can I earn",
        "pay",
        "income",
        "package"
    ],

    "trends": [
        "career trends",
        "industry trends",
        "future careers",
        "which field is growing",
        "high demand careers",
        "future jobs"
    ],

    "thanks": [
        "thank you",
        "thanks"
    ]
}


# ============================================================
# 3. NLP MODEL
# ============================================================

training_sentences = []
training_labels = []

for intent, sentences in intent_data.items():

    for sentence in sentences:
        training_sentences.append(sentence)
        training_labels.append(intent)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

training_vectors = vectorizer.fit_transform(training_sentences)


def detect_intent(user_text):

    user_vector = vectorizer.transform([user_text])

    similarity = cosine_similarity(
        user_vector,
        training_vectors
    )

    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence < 0.15:
        return "unknown"

    return training_labels[best_match]


# ============================================================
# 4. CAREER RECOMMENDATION
# ============================================================

def recommend_career(user_text):

    user_text = user_text.lower()
    scores = {}

    for career, information in careers.items():

        score = 0

        for skill in information["skills"]:

            if skill.lower() in user_text:
                score += 2

        if career.lower() in user_text:
            score += 5

        scores[career] = score


    sorted_careers = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    if sorted_careers[0][1] == 0:
        return list(careers.keys())[:3]


    return [
        career
        for career, score in sorted_careers[:3]
    ]


# ============================================================
# 5. CHATBOT RESPONSE
# ============================================================

def chatbot_response(user_text):

    intent = detect_intent(user_text)


    if intent == "greeting":

        return """
Hello! 👋

I am your AI Career Guidance Assistant.

I can help you with:
• Career options
• Career recommendations
• Required skills
• Salary information
• Industry trends
"""


    elif intent == "career_options":

        response = "\nAVAILABLE CAREER PATHS\n"
        response += "=" * 40 + "\n"

        for number, career in enumerate(careers.keys(), 1):
            response += f"{number}. {career}\n"

        return response


    elif intent == "recommendation":

        recommended = recommend_career(user_text)

        response = "\nRECOMMENDED CAREERS\n"
        response += "=" * 40 + "\n"

        for career in recommended:

            data = careers[career]

            response += f"\n🎯 {career}\n"
            response += f"Description: {data['description']}\n"
            response += f"Skills: {', '.join(data['skills'])}\n"
            response += f"Salary: {data['salary']}\n"
            response += f"Demand: {data['demand']}\n"

        return response


    elif intent == "skills":

        response = "\nREQUIRED CAREER SKILLS\n"
        response += "=" * 40 + "\n"

        for career, data in careers.items():

            response += f"\n{career}\n"
            response += "→ " + ", ".join(data["skills"]) + "\n"

        return response


    elif intent == "salary":

        response = "\nSALARY INFORMATION\n"
        response += "=" * 40 + "\n"

        for career, data in careers.items():

            response += f"{career}: {data['salary']}\n"

        return response


    elif intent == "trends":

        return """
CURRENT CAREER TRENDS
========================================

1. Artificial Intelligence & Machine Learning
   Demand: Very High

2. Cloud Computing
   Demand: Very High

3. Cyber Security
   Demand: Very High

4. Data Science & Analytics
   Demand: Very High

5. DevOps
   Demand: High

6. Software Development
   Demand: High
"""


    elif intent == "thanks":

        return """
You're welcome! 😊

All the best for your career journey! 🚀
"""


    else:

        return """
Sorry, I couldn't understand your question.

Try asking:

• What career options are available?
• Recommend a career for Python
• What skills are required?
• What is the salary?
• What are the current industry trends?
"""


# ============================================================
# 6. START CHATBOT
# ============================================================

def start_chatbot():

    print("=" * 60)
    print("AI CAREER GUIDANCE ASSISTANT")
    print("=" * 60)

    print("""
Welcome!

Ask me anything about careers.

Type 'exit' to close the chatbot.
""")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower().strip() == "exit":

            print("\nAssistant: Goodbye! 👋")
            break

        response = chatbot_response(user_input)

        print("\nAssistant:")
        print(response)


# ============================================================
# 7. RUN PROJECT
# ============================================================

if __name__ == "__main__":
    start_chatbot()
