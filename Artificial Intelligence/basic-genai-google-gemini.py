
# api key : AIzaSyB1EZYIL91NYUMMrRkhvThf4nU3imKSTnQ

"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyB1EZYIL91NYUMMrRkhvThf4nU3imKSTnQ")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("what is ai?")

print(response.text)