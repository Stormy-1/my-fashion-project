import os
import re
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Load API token
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

prompt_template = PromptTemplate(
    input_variables=["age", "gender", "height", "weight", "bmi", "occasion", "facial_features"],
    template=(
        "You are a highly personalized fashion recommendation assistant, like Jarvis, trained to suggest outfits based on detailed user attributes. "
        "Your recommendations must prioritize individual comfort, style, and suitability based on physical features, occasion, and body profile.\n\n"

        "User Details:\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Height: {height} cm\n"
        "- Weight: {weight} kg\n"
        "- BMI: {bmi:.2f}\n"
        "- Occasion: {occasion}\n"
        "- Facial Features: {facial_features}\n\n"

        "IMPORTANT INSTRUCTIONS:\n"
        "1. Provide EXACTLY 3 outfit recommendations\n"
        "2. Each recommendation MUST follow this exact format:\n"
        "   Product Name: [specific product name]\n"
        "   Color Palette: [2-3 colors]\n"
        "   Fit: [recommended fit and size, e.g., Relaxed Fit - XL]\n"
        "   Gender: [gender, e.g., Male or Female]\n"
        "3. Do NOT include any other text, explanations, or conversation\n"
        "4. Focus on {occasion}-appropriate outfits\n"
        "5. Recommendations MUST strongly reflect:\n"
        "   - The user's BMI ({bmi:.1f}) to ensure fit and comfort\n"
        "   - Facial features ({facial_features}) to align colors and styles\n"
        "   - Gender, age, height, and weight to tailor the look to their body type and fashion sensibility\n\n"

        "Example output:\n"
        "1. Product Name: Slim Fit Cotton Kurta with Churidar\n"
        "   Color Palette: Cream, Gold, Maroon\n"
        "   Fit: Regular Fit - M\n\n"
        "   Gender: MALE\n\n"
        "2. Product Name: Embroidered Silk Sherwani\n"
        "   Color Palette: Ivory, Gold\n"
        "   Fit: Tailored Fit  - L\n\n"
        "   Gender: MALE\n\n"
        "3. Product Name: Linen Kurta with Dhoti Pants\n"
        "   Color Palette: Beige, Brown, White\n"
        "   Fit: Relaxed Fit - XL\n\n"
        "   Gender: MALE\n\n"
        "Now provide 3 recommendations for {occasion}:"
    )
)


# Load LLM
try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        max_new_tokens=512,
        temperature=0.0,
        repetition_penalty=1.1,
    )
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Hugging Face LLM: {e}")

# Build chain properly
chain = prompt_template | chat_model

def run_fashion_llm(age: int, gender: str, height: float, weight: float, bmi: float, occasion: str, facial_features: dict):
    try:
        facial_features_str = ", ".join([f"{k}: {v:.2f}" for k, v in facial_features.items()])
        inputs = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "occasion": occasion,
            "facial_features": facial_features_str,
        }

        raw_llm_response = chain.invoke(inputs)

        result_text = raw_llm_response.content if hasattr(raw_llm_response, 'content') else str(raw_llm_response)
        cleaned_response = result_text.strip()

        # Ensure response starts from first recommendation
        cleaned_response = re.sub(r'^.*?(?=1\.\s*Product Name:)', '', cleaned_response, flags=re.DOTALL).strip()

        if not cleaned_response:
            raise ValueError("Empty or improperly formatted LLM response")

        return cleaned_response

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return None

# Test mode
if __name__ == "__main__":
    print("--- Running llm_module.py standalone ---")

    test_features = {
        "jawline_strength": 0.81,
        "cheekbone_prominence": 0.76,
        "eye_spacing": 0.64,
        "nose_length": 0.59
    }

    print("\n[Scenario 1: Male, 30 years old, Casual]")
    result1 = run_fashion_llm(age=30, gender="Male", height=175, weight=70, bmi=22.9, occasion="casual", facial_features=test_features)
    print(result1 or "Failed to generate.")

    print("\n[Scenario 2: Female, 25 years old, Formal]")
    result2 = run_fashion_llm(age=25, gender="Female", height=165, weight=55, bmi=20.2, occasion="formal", facial_features=test_features)
    print(result2 or "Failed to generate.")