import re
import json

def parse_llm_recommendations(text_output, max_recommendations=2):
    """
    Parses the plain text output from the LLM into a structured list of dictionaries,
    where each dictionary represents a clothing style recommendation.
    Limits the number of recommendations to max_recommendations.
    """
    recommendations = []
    style_blocks = re.split(r'\n\s*\d+\.\s*Style Name:', text_output, flags=re.DOTALL)

    for block in style_blocks:
        if not block.strip():
            continue
        style_data = {}
        first_line = block.strip().split('\n')[0]
        style_name_match = re.match(r'(?:Style Name:\s*)?(.+)', first_line)
        if style_name_match:
            style_data['Style Name'] = style_name_match.group(1).strip()
        patterns = {
            'Garment Type': r'- Garment Type: (.+)',
            'Color Palette': r'- Color Palette: (.+)',
            'Fit': r'- Fit: (.+)',
            'Fabric': r'- Fabric: (.+)',
            'Accessories': r'- Accessories: (.+)',
            'Product Name': r'- Product Name: (.+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                style_data[key] = match.group(1).strip()
            else:
                style_data[key] = None
        if style_data and style_data.get('Style Name'):
            recommendations.append(style_data)
        if len(recommendations) >= max_recommendations:
            break
    return recommendations

def save_recommendations_to_json(recommendations, output_file):
    """Save recommendations to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

# Example usage for integration with web_scrapping.py
if __name__ == "__main__":
    # Example LLM output (replace with actual output)
    llm_output = """
    1. Style Name: Urban Explorer\n- Garment Type: Jacket\n- Color Palette: Olive Green\n- Fit: Regular\n- Fabric: Cotton\n- Accessories: Sunglasses, Backpack\n- Product Name: Explorer Cotton Jacket\n\n2. Style Name: Minimalist Chic\n- Garment Type: T-shirt\n- Color Palette: White, Black\n- Fit: Slim\n- Fabric: Linen\n- Accessories: Watch, Leather Belt\n- Product Name: Chic Linen Tee\n\n3. Style Name: Sporty Casual\n- Garment Type: Hoodie\n- Color Palette: Grey\n- Fit: Loose\n- Fabric: Fleece\n- Accessories: Cap, Sneakers\n- Product Name: Sporty Fleece Hoodie\n    """
    recs = parse_llm_recommendations(llm_output, max_recommendations=2)
    save_recommendations_to_json(recs, 'llm_recommendations.json')

