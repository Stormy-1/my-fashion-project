import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    print("ERROR: Hugging Face API token (HF_TOKEN) not found in your .env file.")
    print("Please create a .env file in the same directory as llm.py and add HF_TOKEN='YOUR_TOKEN_HERE'")
    raise ValueError("HF_TOKEN environment variable not set. Cannot initialize HuggingFaceEndpoint.")

prompt_template = PromptTemplate(
    input_variables=["age", "gender", "height", "weight"],
    template=(
        "You are a fashion recommendation expert.\n\n"
        "Given the following user attributes:\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Height: {height} cm\n"
        "- Weight: {weight} kg\n\n"
        "Suggest **3 distinct clothing styles** that would look great on this user.\n"
        "For each style, describe the:\n"
        "  - Garment Type\n"
        "  - Color Palette\n"
        "  - Fit\n"
        "  - Fabric\n"
        "  - Accessories\n"
        "  - A suggested Product Name (e.g., 'blue relaxed-fit cotton t-shirt')\n\n"
        "Format your suggestions as a numbered list of styles. Do not use JSON.\n"
        "Example:\n"
        "1. Style Name: Casual Comfort\n"
        "   - Garment Type: T-shirt and Jeans\n"
        "   - Color Palette: Earthy tones (e.g., olive green, beige, denim blue)\n"
        "   - Fit: Relaxed-fit t-shirt, straight-leg jeans\n"
        "   - Fabric: Soft cotton t-shirt, durable denim jeans\n"
        "   - Accessories: Canvas sneakers, simple watch\n"
        "   - Product Name: Olive green relaxed-fit cotton t-shirt with classic blue straight-leg jeans\n\n"
        "2. Style Name: Smart Everyday\n"
        "   - Garment Type: Polo shirt and Chinos\n"
        "   - Color Palette: Neutrals with a pop of color (e.g., navy, grey, burgundy)\n"
        "   - Fit: Slim-fit polo, tapered chinos\n"
        "   - Fabric: Piqué cotton polo, stretch cotton chinos\n"
        "   - Accessories: Leather loafers, minimalist belt\n"
        "   - Product Name: Navy slim-fit pique polo shirt with grey tapered chinos\n"
    )
)

try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        max_new_tokens=512,
        temperature=0.0, # Set to 0.0 for deterministic output
        repetition_penalty=1.1
    )
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    print(f"ERROR: Failed to initialize Hugging Face LLM: {e}")
    print("Please check your HF_TOKEN, internet connection, and the `repo_id` validity.")
    raise

chain = prompt_template | chat_model

def run_fashion_llm(age: int, gender: str, height: int = 170, weight: int = 65):
    try:
        raw_llm_response = chain.invoke({
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight
        })
        result_text = raw_llm_response.content if hasattr(raw_llm_response, 'content') else str(raw_llm_response)
        return result_text
    except Exception as e:
        return f"Error generating recommendations: {e}"

if __name__ == "__main__":
    print("--- Running llm.py in standalone test mode ---")

    print("\n[Scenario 1: Male, 30 years old]")
    test_result_1 = run_fashion_llm(age=30, gender="Male")
    print(test_result_1)

    print("\n[Scenario 2: Female, 25 years old]")
    test_result_2 = run_fashion_llm(age=25, gender="Female", height=165, weight=55)
    print(test_result_2)