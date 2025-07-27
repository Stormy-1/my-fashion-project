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
    input_variables=["age", "gender", "height", "weight", "bmi", "occasion", "facial_features"],
    template=(
        "You are a highly creative and detail-oriented fashion recommendation expert.\n\n"
        "Given the following user attributes:\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Height: {height} cm\n"
        "- Weight: {weight} kg\n"
        "- BMI: {bmi:.2f}\n"
        "- Occasion: {occasion}\n"
        "- Facial Features: {facial_features}\n\n"
        "Your task is to suggest *3 distinct, highly relevant, and creative clothing styles* that would look great on this user specifically for the given occasion and their facial features.\n"
        "- Use the facial features to recommend styles, colors, accessories, or fits that would enhance or complement the user's appearance.\n"
        "- If the occasion is themed (e.g., anime, sports, cultural, fandom, etc.), incorporate references, colors, or accessories that fit the theme.\n"
        "- Explicitly reference the occasion in each style name and description.\n"
        "- Be as specific as possible with garment types, color palettes, fits, fabrics, and accessories.\n"
        "- If the occasion is related to pop culture, fandom, or a specific character, use creative cues, color schemes, or accessories inspired by that theme.\n"
        "- Make sure each style is unique and tailored to the user's body type, facial features, and the occasion.\n"
        "- *Make sure each style is appropriate for the user's specified gender. If the occasion is themed, adapt the style to fit the gender while still referencing the theme.*\n\n"
        "For each style, provide:\n"
        "  - Style Name (should reference the occasion or theme)\n"
        "  - Garment Type\n"
        "  - Color Palette\n"
        "  - Fit\n"
        "  - Fabric\n"
        "  - Accessories (be creative and occasion-specific)\n"
        "  - A suggested Product Name (e.g., 'orange ninja-inspired hoodie with leaf village headband')\n\n"
        "Format your suggestions as a numbered list of styles. Do not use JSON.\n"
        "Example:\n"
        "1. Style Name: Anime Hero Streetwear (for an anime fan party, male, with sharp jawline and black hair)\n"
        "   - Garment Type: Orange hoodie, navy joggers\n"
        "   - Color Palette: Orange, navy blue, white\n"
        "   - Fit: Relaxed-fit hoodie, tapered joggers\n"
        "   - Fabric: Soft fleece hoodie, stretch cotton joggers\n"
        "   - Accessories: Leaf village headband, ninja sandals\n"
        "   - Product Name: Orange ninja-inspired hoodie with navy joggers and cosplay headband\n\n"
        "2. Style Name: Elegant Gala Attire (for a formal evening event, female, with wavy hair and high cheekbones)\n"
        "   - Garment Type: Black evening gown, silver shawl\n"
        "   - Color Palette: Black, silver, white\n"
        "   - Fit: Fitted bodice, flowing skirt\n"
        "   - Fabric: Silk gown, satin shawl\n"
        "   - Accessories: Silver clutch, pearl earrings\n"
        "   - Product Name: Classic black silk gown with silver accessories\n"
    )
)

try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        max_new_tokens=512,
        temperature=0.0, 
        repetition_penalty=1.1
    )
    chat_model = ChatHuggingFace(llm=llm)
except Exception as e:
    print(f"ERROR: Failed to initialize Hugging Face LLM: {e}")
    print("Please check your HF_TOKEN, internet connection, and the repo_id validity.")
    raise

chain = prompt_template | chat_model

def run_fashion_llm(age: int, gender: str, height: float, weight: float, bmi: float, occasion: str, facial_features: str):
    try:
        raw_llm_response = chain.invoke({
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "occasion": occasion,
            "facial_features": facial_features
        })
        result_text = raw_llm_response.content if hasattr(raw_llm_response, 'content') else str(raw_llm_response)
        return result_text
    except Exception as e:
        return f"Error generating recommendations: {e}"

if __name__ == "__main__":
    print("--- Running llm.py in standalone test mode ---")

    print("\n[Scenario 1: Male, 30 years old]")
    test_result_1 = run_fashion_llm(age=30, gender="Male", height=175, weight=70, bmi=22.9, occasion="casual", facial_features="test")
    print(test_result_1)

    print("\n[Scenario 2: Female, 25 years old]")
    test_result_2 = run_fashion_llm(age=25, gender="Female", height=165, weight=55, bmi=20.2, occasion="formal", facial_features="test")
    print(test_result_2)