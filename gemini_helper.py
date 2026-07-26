
import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_medical_report(disease, confidence):
    """
    Generate an AI-powered medical report using Google Gemini.
    """

    prompt = f"""
You are an experienced medical AI assistant.

A chest X-ray deep learning model has produced the following result:

Disease Prediction: {disease}
Confidence Score: {confidence:.2f}%

Generate a professional medical report using the following sections.

# Prediction Summary
Briefly summarize the prediction.

# Clinical Interpretation
Explain what this prediction means in simple language.

# Recommended Next Steps
Suggest appropriate next steps for the patient.

# Lifestyle Advice
Provide general lifestyle and health recommendations.

# Disclaimer
Mention that this AI-generated report is for educational purposes only and cannot replace professional medical advice.

Keep the report professional, concise, and easy to understand.

Return the report as plain text only.
Do not use Markdown formatting like #, ##, **, *, or bullet symbols.
Use clear headings and paragraphs only.
"""

    try:

        response = client.models.generate_content(
               model="models/gemini-3-flash-preview",
               contents=prompt
        )

        if response.text:
            return response.text

        return "No report was generated."

    except Exception as e:

        print("Gemini Error:", e)

        return f"Error generating report: {str(e)}"