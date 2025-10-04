# My Fashion Project

### An AI-Powered Personalized Fashion Stylist

A comprehensive AI-powered fashion recommendation system that analyzes facial features, age, and gender to provide personalized fashion suggestions. The system combines deep learning, computer vision, and web scraping to deliver tailored fashion recommendations.

---

### 📋 Table of Contents

* [🌟 Features](#-features)
* [🏗️ Architecture](#️-architecture)
* [📁 Project Structure](#-project-structure)
* [🚀 Getting Started](#-getting-started)
* [🎯 How It Works](#-how-it-works)
* [🛠️ API Endpoints](#️-api-endpoints)
* [🧪 Testing](#-testing)
* [🔧 Configuration](#-configuration)
* [📊 Technical Details](#-technical-details)
* [🙏 Acknowledgments](#-acknowledgments)

---

### 🌟 Features

* **AI-Powered Analysis**: Deep learning models for age and gender detection using PyTorch and EfficientNet-B4.
* **Facial Feature Extraction**: Analyzes 16+ facial attributes for personalized recommendations.
* **Real-time Camera Capture**: OpenCV integration for live photo capture.
* **LLM Fashion Intelligence**: Advanced language models generate creative, themed fashion recommendations.
* **Web Scraping**: Automatically finds real fashion products from e-commerce platforms.
* **Modern UI**: Beautiful React frontend with Tailwind CSS and shadcn/ui components.
* **Personalized Recommendations**: Tailored fashion suggestions based on facial features, body type, and occasion.
* **Theme-Aware Styling**: Supports themed occasions (anime, sports, cultural events, etc.).

---

### 🏗️ Architecture

* **Backend (Flask API)**
    * Flask REST API with CORS support
    * PyTorch Models for age/gender classification and facial feature extraction
    * LLM Integration for generating fashion recommendations using Hugging Face
    * Web Scraping Engine for finding real fashion products
    * OpenCV for image processing and camera integration
* **Frontend (React + Vite)**
    * React 18 with TypeScript
    * Vite for fast development and building
    * Tailwind CSS for styling
    * shadcn/ui component library
    * React Router for navigation

---

### 📁 Project Structure

Of course. Here is that project structure formatted properly into a readable tree.

```
my-fashion-project/
├── backend/                          # Flask API server
│   ├── api.py                       # Main Flask application entry point
│   ├── predict.py                   # Age/gender prediction logic
│   ├── ml.py                        # Machine learning utility functions
│   ├── llm.py                       # LLM integration for recommendations
│   ├── web_scrapping.py             # Web scraping functionality
│   ├── main.py                      # Script for model training
│   ├── requirements.txt             # Python dependencies
│   ├── .env.example                 # Example environment variables
│   ├── age_gender_model.pth         # Trained age/gender model
│   ├── celeba_imbalance_aware_classifier.pth  # Trained facial feature model
│   ├── utkface_aligned_cropped/     # UTKFace dataset for training
│   └── uploads/                     # Directory for user-uploaded images
│
├── frontend/                         # React frontend application
│   ├── src/
│   │   ├── assets/                  # Static assets like images and fonts
│   │   ├── components/              # Reusable UI components
│   │   ├── pages/                   # Application pages (views)
│   │   ├── App.tsx                  # Main React App component
│   │   └── main.tsx                 # Application entry point
│   ├── public/                      # Public static files
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # Node.js dependencies
│   ├── vite.config.ts               # Vite configuration
│   └── tailwind.config.ts           # Tailwind CSS configuration
│
├── .gitignore                        # Specifies intentionally untracked files
└── README.md                         # This file
```
---

### 🚀 Getting Started

#### Prerequisites

* Python 3.8+ with pip
* Node.js 18+ with npm
* Git
* Webcam (optional, for camera capture feature)

#### Backend Setup

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/Stormy-1/my-fashion-project.git](https://github.com/Stormy-1/my-fashion-project.git)
    cd my-fashion-project
    ```
2.  **Set up Python environment**
    ```sh
    cd backend
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Python dependencies**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Start the Flask server**
    ```sh
    python api.py
    ```
    The backend will be available at `http://localhost:5000`.

#### Frontend Setup

1.  **Navigate to frontend directory** (from the root `my-fashion-project` folder)
    ```sh
    cd frontend
    ```
2.  **Install Node.js dependencies**
    ```sh
    npm install
    ```
3.  **Start the development server**
    ```sh
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173`.

---

### 🎯 How It Works

1.  **Image Upload/Capture**: Users can upload an image or capture a photo using their webcam.
2.  **AI Analysis Pipeline**: Deep learning models analyze the uploaded image for age, gender, and 16+ facial attributes. User input like height, weight, and occasion is also integrated.
3.  **Fashion Recommendation Generation**: An advanced language model processes the analysis to generate creative, themed fashion recommendations tailored to the user.
4.  **Product Discovery**: The system automatically scrapes e-commerce sites for real products that match the generated recommendations, enriching them with prices, ratings, and images.
5.  **Results Display**: A clean, modern UI displays the product recommendations with details and purchase links.

---

### 🛠️ API Endpoints

* `POST /api/upload`: Upload an image and get fashion recommendations.
* `POST /api/camera_capture`: Process a camera-captured image.
* `GET /api/health`: Health check endpoint.

#### Request Format
```json
{
  "image": "base64_encoded_image_data",
  "height": "170",
  "weight": "65",
  "occasion": "casual"
}
````

#### Response Format

```json
{
  "success": true,
  "recommendations": [
    {
      "brand": "Nike",
      "title": "Product Name",
      "price": "₹1,199",
      "rating": "4.2",
      "reviews": "1,234",
      "image_url": "https://...",
      "product_url": "https://..."
    }
  ]
}
```

-----

### 🧪 Testing

  * **Backend Testing**
    ```sh
    cd backend
    python manual_test.py
    ```
  * **Frontend Testing**
    ```sh
    cd frontend
    npm run lint
    ```

-----

### 🔧 Configuration

  * **Environment Variables**: The backend uses a `.env` file for configuration (e.g., `HF_TOKEN`).
  * **Dependencies**: Backend dependencies are in `backend/requirements.txt`. Frontend dependencies are in `frontend/package.json`.
  * **Model Files**: The system requires pre-trained models (`age_gender_model.pth`, etc.) to be present in the `backend` directory.

-----

### 📊 Technical Details

  * **Machine Learning Models**:
      * **Age/Gender Classification**: Custom CNN trained on the UTKFace dataset.
      * **Facial Feature Extraction**: Classifier trained on the CelebA dataset for 16+ attributes.
  * **Data Flow**:
    1.  Image → Preprocessing → Feature Extraction
    2.  Features → LLM → Fashion Recommendations
    3.  Recommendations → Web Scraping → Product Data
    4.  Product Data → Frontend → User Display

-----

### 🙏 Acknowledgments

  * **UTKFace Dataset** for age/gender classification training.
  * **CelebA Dataset** for facial feature extraction.
  * **OpenCV Community** for computer vision tools.
  * **React & Vite Teams** for excellent frontend tooling.
  * **shadcn/ui** for beautiful, accessible components.

