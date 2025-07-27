# Fashion Recommendation API Integration Guide

## Overview
This API provides fashion recommendations based on uploaded images and user parameters (height, weight, occasion). The backend processes the image using deep learning models for age/gender detection, combines it with user inputs, uses LLM for fashion recommendations, and performs web scraping to find actual fashion items.

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API server is running.

**Response:**
```json
{
    "status": "healthy",
    "message": "Fashion Recommendation API is running"
}
```

### 2. Fashion Recommendations (Main Endpoint)
**POST** `/recommend`

Upload an image and get personalized fashion recommendations.

**Request Format:** `multipart/form-data`

**Required Fields:**
- `image`: Image file (PNG, JPG, JPEG, GIF, BMP)
- `height`: Height in centimeters (100-250)
- `weight`: Weight in kilograms (30-300)

**Optional Fields:**
- `occasion`: Event type (default: "casual")

**Example Frontend JavaScript:**
```javascript
async function getFashionRecommendations(imageFile, height, weight, occasion = 'casual') {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('height', height);
    formData.append('weight', weight);
    formData.append('occasion', occasion);
    
    try {
        const response = await fetch('http://localhost:5000/recommend', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('Recommendations received:', result);
            return result;
        } else {
            console.error('API Error:', result.error);
            return null;
        }
    } catch (error) {
        console.error('Network Error:', error);
        return null;
    }
}
```

**Success Response:**
```json
{
    "success": true,
    "user_info": {
        "height": 170,
        "weight": 65,
        "occasion": "casual",
        "bmi": 22.49
    },
    "facial_features": {
        "Attractive": 0.85,
        "Male": 0.15,
        "Young": 0.92,
        "Smiling": 0.78
    },
    "llm_recommendations": [
        {
            "item_type": "Top",
            "description": "Casual cotton t-shirt",
            "color": "Navy blue",
            "style": "Relaxed fit"
        }
    ],
    "scraped_products": [
        {
            "brand": "Nike",
            "description": "Cotton casual t-shirt",
            "price": "$29.99",
            "rating": "4.5/5",
            "product_link": "https://example.com/product",
            "image_url": "https://example.com/image.jpg"
        }
    ],
    "total_products": 15
}
```

**Error Response:**
```json
{
    "success": false,
    "error": "Error message description"
}
```

### 3. Upload Test
**POST** `/upload-test`

Test file upload functionality without processing.

**Request:** Same as `/recommend` but only requires `image` field.

## Frontend Integration Examples

### HTML Form Example
```html
<form id="fashionForm" enctype="multipart/form-data">
    <input type="file" id="imageInput" accept="image/*" required>
    <input type="number" id="heightInput" placeholder="Height (cm)" min="100" max="250" required>
    <input type="number" id="weightInput" placeholder="Weight (kg)" min="30" max="300" required>
    <select id="occasionInput">
        <option value="casual">Casual</option>
        <option value="formal">Formal</option>
        <option value="party">Party</option>
        <option value="business">Business</option>
        <option value="date">Date</option>
    </select>
    <button type="submit">Get Recommendations</button>
</form>

<div id="results"></div>
```

### JavaScript Integration
```javascript
document.getElementById('fashionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const imageFile = document.getElementById('imageInput').files[0];
    const height = document.getElementById('heightInput').value;
    const weight = document.getElementById('weightInput').value;
    const occasion = document.getElementById('occasionInput').value;
    
    if (!imageFile) {
        alert('Please select an image');
        return;
    }
    
    // Show loading state
    document.getElementById('results').innerHTML = '<p>Processing your image...</p>';
    
    const result = await getFashionRecommendations(imageFile, height, weight, occasion);
    
    if (result) {
        displayRecommendations(result);
    } else {
        document.getElementById('results').innerHTML = '<p>Error getting recommendations</p>';
    }
});

function displayRecommendations(data) {
    const resultsDiv = document.getElementById('results');
    
    let html = `
        <h3>Your Fashion Recommendations</h3>
        <div class="user-info">
            <p>BMI: ${data.user_info.bmi}</p>
            <p>Occasion: ${data.user_info.occasion}</p>
        </div>
    `;
    
    if (data.scraped_products && data.scraped_products.length > 0) {
        html += '<div class="products">';
        data.scraped_products.forEach((product, index) => {
            html += `
                <div class="product-card">
                    <h4>${product.brand || 'Unknown Brand'}</h4>
                    <p>${product.description}</p>
                    <p class="price">${product.price}</p>
                    <p class="rating">Rating: ${product.rating}</p>
                    <a href="${product.product_link}" target="_blank">View Product</a>
                </div>
            `;
        });
        html += '</div>';
    }
    
    resultsDiv.innerHTML = html;
}
```

### React Component Example
```jsx
import React, { useState } from 'react';

function FashionRecommendation() {
    const [image, setImage] = useState(null);
    const [height, setHeight] = useState('');
    const [weight, setWeight] = useState('');
    const [occasion, setOccasion] = useState('casual');
    const [recommendations, setRecommendations] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) return;

        setLoading(true);
        const result = await getFashionRecommendations(image, height, weight, occasion);
        setRecommendations(result);
        setLoading(false);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input 
                    type="file" 
                    accept="image/*"
                    onChange={(e) => setImage(e.target.files[0])}
                    required 
                />
                <input 
                    type="number" 
                    placeholder="Height (cm)"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    min="100" max="250"
                    required 
                />
                <input 
                    type="number" 
                    placeholder="Weight (kg)"
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                    min="30" max="300"
                    required 
                />
                <select value={occasion} onChange={(e) => setOccasion(e.target.value)}>
                    <option value="casual">Casual</option>
                    <option value="formal">Formal</option>
                    <option value="party">Party</option>
                </select>
                <button type="submit" disabled={loading}>
                    {loading ? 'Processing...' : 'Get Recommendations'}
                </button>
            </form>

            {recommendations && (
                <div>
                    <h3>Recommendations</h3>
                    {recommendations.scraped_products?.map((product, index) => (
                        <div key={index} className="product-card">
                            <h4>{product.brand}</h4>
                            <p>{product.description}</p>
                            <p>{product.price}</p>
                            <a href={product.product_link} target="_blank" rel="noopener noreferrer">
                                View Product
                            </a>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
```

## Running the API Server

1. **Install Dependencies:**
```bash
pip install flask flask-cors torch torchvision pillow opencv-python timm joblib numpy
```

2. **Start the Server:**
```bash
cd backend
python api.py
```

3. **Server will start on:** `http://localhost:5000`

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (missing/invalid parameters)
- `413`: File too large (>16MB)
- `500`: Internal server error

Always check the `success` field in the JSON response to determine if the request was processed successfully.

## File Upload Limits

- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG, GIF, BMP
- Files are temporarily stored and automatically cleaned up after processing

## CORS Support

The API includes CORS headers, allowing requests from any origin. This enables frontend applications running on different ports/domains to access the API.

## Testing

Use the provided `test_api_client.py` to test the API:

```bash
python test_api_client.py
```

This will test all endpoints and provide example usage patterns.
