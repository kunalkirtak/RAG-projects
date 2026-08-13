
import os
import google.generativeai as genai

genai.configure(
    api_key=os.environ["GEMINI_API_KEY"]
)

class GeminiClient:

    def __init__(self):

        self.model = genai.GenerativeModel(
            "gemini-3.5-flash"
        )

    def generate(self, prompt):

        response = self.model.generate_content(

            prompt,

            generation_config={

                "temperature":0.2,

                "max_output_tokens":1024

            }

        )

        return response.text
