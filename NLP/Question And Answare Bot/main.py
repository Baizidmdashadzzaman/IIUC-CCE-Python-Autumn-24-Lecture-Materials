# pip install transformers torch

# Import the pipeline from Hugging Face transformers
from transformers import pipeline

# Load the pre-trained BERT model fine-tuned on the SQuAD dataset
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load CV text from a file (assuming it's stored in 'cv.txt')
def load_cv(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Function to ask questions about the CV and get answers
def ask_question(question, context):
    result = qa_model(question=question, context=context)
    return result['answer']

# Load your CV text
cv_text = load_cv('cv.txt')

# Start an interactive loop to ask questions about your CV
print("Ask any question about your CV (type 'exit' to quit):\n")
while True:
    # Take user input
    question = input("Your question: ")

    # Exit condition
    if question.lower() == 'exit':
        print("Exiting the QA system.")
        break

    # Get the answer from the CV
    try:
        answer = ask_question(question, cv_text)
        print(f"Answer: {answer}\n")
    except Exception as e:
        print(f"Sorry, I couldn't find an answer. Error: {str(e)}\n")
