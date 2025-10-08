import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Hardcoded standard resume text
standard_resume_text = """
John Doe
Software Engineer
john.doe@email.com | (123) 456-7890 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Results-driven Software Engineer with 5+ years of experience developing scalable web applications. 
Proficient in JavaScript, React, Node.js, and cloud technologies. Passionate about creating 
user-friendly solutions and optimizing application performance.

TECHNICAL SKILLS
• Programming Languages: JavaScript, TypeScript, Python, Java
• Frontend: React, Angular, Vue.js, HTML5, CSS3, Tailwind CSS
• Backend: Node.js, Express.js, Django, Spring Boot
• Databases: MongoDB, PostgreSQL, MySQL, Redis
• Cloud: AWS, Google Cloud, Azure, Docker, Kubernetes
• Tools: Git, Jenkins, JIRA, Agile Methodologies

PROFESSIONAL EXPERIENCE
Senior Software Engineer | Tech Corp Inc. | 2021 - Present
• Led development of microservices architecture serving 1M+ users
• Improved application performance by 40% through optimization
• Mentored junior developers and conducted code reviews
• Implemented CI/CD pipelines reducing deployment time by 60%

Software Engineer | StartupXYZ | 2019 - 2021
• Developed responsive web applications using React and Node.js
• Collaborated with cross-functional teams to deliver projects on time
• Integrated third-party APIs and payment gateways
• Participated in agile development processes

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2015 - 2019
GPA: 3.8/4.0

CERTIFICATIONS
• AWS Certified Developer Associate (2022)
• Google Cloud Professional Developer (2021)
"""

# Preprocess text: tokenize, remove stopwords and punctuation
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_tokens)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

# Check for presence of key sections
def check_sections(resume_text):
    sections = {
        "Professional Summary": re.search(r'professional summary|summary|objective', resume_text, re.IGNORECASE),
        "Technical Skills": re.search(r'technical skills|skills|key skills', resume_text, re.IGNORECASE),
        "Professional Experience": re.search(r'professional experience|work experience|experience|employment history', resume_text, re.IGNORECASE),
        "Education": re.search(r'education|academic background', resume_text, re.IGNORECASE),
        "Certifications": re.search(r'certifications|certificates|credentials', resume_text, re.IGNORECASE)
    }
    missing_sections = [section for section, found in sections.items() if not found]
    return missing_sections

# Chatbot response logic
def chatbot_response(user_query, resume_text, missing_keywords, missing_sections, rating, matching_keywords):
    query = user_query.lower().strip()
    response = "I'm here to help with your resume! "

    # Patterns for common questions
    if re.search(r'improve.*skills|skills.*section|add.*skills', query):
        if "Technical Skills" in missing_sections:
            response += "Your resume is missing a 'Technical Skills' section. Add one like in the standard resume, listing specific skills (e.g., JavaScript, React, AWS)."
        else:
            response += f"Enhance your skills section by adding relevant keywords like: {', '.join(missing_keywords[:5])}. Tailor them to match the standard resume's skills (e.g., TypeScript, Docker)."
    
    elif re.search(r'improve.*experience|work.*history|experience.*section', query):
        if "Professional Experience" in missing_sections:
            response += "Your resume lacks a 'Professional Experience' section. Include detailed roles like the standard resume, with quantifiable achievements (e.g., 'Improved performance by 40%')."
        else:
            response += "Strengthen your experience section with action verbs and metrics, like 'Led microservices development' or 'Reduced deployment time by 60%' as seen in the standard resume."
    
    elif re.search(r'missing.*section|section.*missing|structure', query):
        if missing_sections:
            response += f"Your resume is missing these sections: {', '.join(missing_sections)}. Add them to match the standard resume's structure (Professional Summary, Technical Skills, etc.)."
        else:
            response += "Your resume has all key sections! Ensure each is detailed and aligns with the standard resume's format."
    
    elif re.search(r'keywords|missing.*keywords|relevant.*skills', query):
        if missing_keywords:
            response += f"Add these missing keywords to align with the standard resume: {', '.join(missing_keywords[:5])}. Focus on technical terms like 'Kubernetes' or 'Node.js'."
        else:
            response += "Your resume includes most relevant keywords! Double-check for niche skills like 'Tailwind CSS' or 'Jenkins' from the standard."
    
    elif re.search(r'rating|score|how.*good', query):
        response += f"Your resume scored {rating}% compared to the standard Software Engineer resume. To improve, focus on {', '.join(missing_sections) if missing_sections else 'adding more quantifiable achievements'} and keywords like {', '.join(missing_keywords[:3]) if missing_keywords else 'specific tools'}."
    
    elif re.search(r'length|too.*long|too.*short', query):
        word_count = len(word_tokenize(resume_text))
        if word_count < 300:
            response += "Your resume is short (word count: {word_count}). Add more details to skills and experience, like the standard resume's detailed descriptions."
        elif word_count > 1000:
            response += "Your resume is lengthy (word count: {word_count}). Be concise, focusing on key achievements as in the standard resume."
        else:
            response += "Your resume length is good (word count: {word_count}). Ensure content is impactful, like the standard's quantifiable metrics."
    
    else:
        response += "Could you clarify your question? Ask about specific sections (e.g., skills, experience), keywords, or your rating to get tailored advice."

    return response

# Main app
st.title("AI Resume Review Bot - Software Engineer Edition")

st.write("Upload your resume to get a rating compared to a standard Software Engineer resume, plus areas for improvement. Chat with the bot below to ask about specific changes!")

# Upload resume
resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

# Initialize session state for analysis results and chat history
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        'resume_text': None,
        'rating': None,
        'matching_keywords': [],
        'missing_keywords': [],
        'missing_sections': []
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Process resume
if st.button("Review Resume") and resume_file:
    # Extract and preprocess resume text
    resume_text = extract_text_from_pdf(resume_file)
    processed_resume = preprocess_text(resume_text)
    
    # Preprocess standard resume
    processed_standard = preprocess_text(standard_resume_text)
    
    # Use TF-IDF for vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_standard])
    
    # Compute similarity score (rating)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    rating = round(similarity * 100, 2)
    
    # Get feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top keywords from resume and standard
    resume_tfidf = tfidf_matrix[0].toarray()[0]
    standard_tfidf = tfidf_matrix[1].toarray()[0]
    
    # Top matching keywords
    matching_keywords = [feature_names[i] for i in range(len(feature_names)) if resume_tfidf[i] > 0 and standard_tfidf[i] > 0]
    matching_keywords = sorted(matching_keywords, key=lambda x: min(resume_tfidf[feature_names.tolist().index(x)], standard_tfidf[feature_names.tolist().index(x)]), reverse=True)[:10]
    
    # Missing keywords
    missing_keywords = [feature_names[i] for i in range(len(feature_names)) if standard_tfidf[i] > 0.1 and resume_tfidf[i] < 0.05]
    
    # Check missing sections
    missing_sections = check_sections(resume_text)
    
    # Store results in session state
    st.session_state.analysis_results = {
        'resume_text': resume_text,
        'rating': rating,
        'matching_keywords': matching_keywords,
        'missing_keywords': missing_keywords,
        'missing_sections': missing_sections
    }
    
    # Rule-based suggestions
    suggestions = []
    word_count = len(word_tokenize(resume_text))
    if word_count < 300:
        suggestions.append("Your resume is quite short. Add more details to experiences and skills to match the depth in the standard.")
    if word_count > 1000:
        suggestions.append("Your resume is lengthy. Be more concise, focusing on quantifiable achievements like in the standard.")
    if missing_sections:
        suggestions.append(f"Add these missing sections: {', '.join(missing_sections)}. The standard resume includes them for completeness.")
    if len(missing_keywords) > 5:
        suggestions.append("Incorporate more relevant keywords (e.g., from skills or experience) to boost alignment.")
    if not re.search(r'\d+%|\d+M|\d+K|reduced by|improved by|led|developed|implemented', resume_text, re.IGNORECASE):
        suggestions.append("Use quantifiable achievements (e.g., 'improved by 40%') as seen in the standard resume's experience section.")
    
    # Display results
    st.subheader("Review Results")
    st.write(f"**Overall Rating:** {rating}% (based on keyword and content similarity to the standard Software Engineer resume)")
    
    st.subheader("Top Matching Keywords")
    st.write(", ".join(matching_keywords) if matching_keywords else "No strong matches found.")
    
    st.subheader("Missing Keywords (Consider Adding These)")
    st.write(", ".join(missing_keywords) if missing_keywords else "No major gaps detected.")
    
    st.subheader("Areas to Improve")
    for sug in suggestions:
        st.write(f"- {sug}")
    if not suggestions:
        st.write("Your resume aligns well with the standard—great job!")
    
    # Option to view raw resume text
    with st.expander("View Extracted Resume Text"):
        st.text(resume_text)
    
    # Option to view standard resume
    with st.expander("View Standard Reference Resume"):
        st.text(standard_resume_text)

# Chatbot interface
st.subheader("Chat with the Resume Advisor")
user_query = st.text_input("Ask about improving your resume (e.g., 'How do I improve my skills section?')")

if user_query and st.session_state.analysis_results['resume_text']:
    # Generate chatbot response
    response = chatbot_response(
        user_query,
        st.session_state.analysis_results['resume_text'],
        st.session_state.analysis_results['missing_keywords'],
        st.session_state.analysis_results['missing_sections'],
        st.session_state.analysis_results['rating'],
        st.session_state.analysis_results['matching_keywords']
    )
    
    # Add to chat history
    st.session_state.chat_history.append({"user": user_query, "bot": response})
    
    # Display chat history
    for chat in st.session_state.chat_history[-5:]:  # Show last 5 exchanges
        st.write(f"**You**: {chat['user']}")
        st.write(f"**Bot**: {chat['bot']}")
elif user_query and not st.session_state.analysis_results['resume_text']:
    st.write("Please upload and review a resume first to enable the chatbot.")