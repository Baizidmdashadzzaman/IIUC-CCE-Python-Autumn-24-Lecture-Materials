import os
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyB1EZYIL91NYUMMrRkhvThf4nU3imKSTnQ")

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Start a chat session
chat_session = model.start_chat(
    history=[]
)

# response = chat_session.send_message("what is ai?")
# print(response.text)

# Dynamic Q&A loop
print("You can now ask questions! Type 'exit' to quit.")
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # Send the user input to the model
    response = chat_session.send_message(user_input)

    # Print the model's response
    print("AI: ", response.text)
