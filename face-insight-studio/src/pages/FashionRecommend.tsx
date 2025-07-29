import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Upload, Camera, ShoppingBag, Star, Eye, RefreshCw, Loader2, StarHalf, ArrowLeft, Sparkles } from 'lucide-react';
import Navbar from '@/components/Navbar';
import { saveRecommendation } from '@/utils/recommendationStorage';

const FashionRecommend = () => {
  const navigate = useNavigate();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [textRecommendations, setTextRecommendations] = useState<any[]>([]);
  const [showRecommendationsButton, setShowRecommendationsButton] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [height, setHeight] = useState<string>('');
  const [weight, setWeight] = useState<string>('');
  const [occasion, setOccasion] = useState<string>('');
  const [formErrors, setFormErrors] = useState<{[key: string]: string}>({
    height: '',
    weight: '',
    occasion: ''
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper function to render star rating
  const renderStarRating = (rating: string) => {
    const numRating = parseFloat(rating);
    if (isNaN(numRating)) {
      return (
        <div className="flex items-center">
          <span className="text-sm text-muted-foreground">No rating</span>
        </div>
      );
    }

    const stars = [];
    const fullStars = Math.floor(numRating);
    const hasHalfStar = numRating % 1 !== 0;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    // Full stars
    for (let i = 0; i < fullStars; i++) {
      stars.push(
        <Star key={`full-${i}`} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
      );
    }

    // Half star
    if (hasHalfStar) {
      stars.push(
        <StarHalf key="half" className="w-4 h-4 fill-yellow-400 text-yellow-400" />
      );
    }

    // Empty stars
    for (let i = 0; i < emptyStars; i++) {
      stars.push(
        <Star key={`empty-${i}`} className="w-4 h-4 text-gray-300" />
      );
    }

    return (
      <div className="flex items-center space-x-1">
        <div className="flex">{stars}</div>
        <span className="text-sm font-medium text-gray-700">{numRating}</span>
      </div>
    );
  };

  // Helper function to format price with rupee symbol
  const formatPrice = (price: string) => {
    if (price === 'Price not available' || !price) {
      return (
        <span className="text-sm text-muted-foreground">Price not available</span>
      );
    }
    
    // Remove all unwanted characters including rupee symbols, commas, and special characters
    // Keep only digits, decimal points, and spaces
    const cleanPrice = price
      .replace(/[₹,ₐ]/g, '') // Remove rupee symbols and the unwanted character
      .replace(/[^0-9.,\s]/g, '') // Keep only numbers, commas, dots, and spaces
      .trim();
    
    return (
      <div className="flex items-center space-x-1">
        <span className="text-lg font-bold text-green-600">₹</span>
        <span className="text-lg font-bold text-primary">{cleanPrice}</span>
      </div>
    );
  };

  const validateForm = () => {
    const errors: {[key: string]: string} = {};
    let isValid = true;
    
    if (!height.trim()) {
      errors.height = 'Height is required';
      isValid = false;
    } else if (isNaN(Number(height)) || Number(height) < 100 || Number(height) > 250) {
      errors.height = 'Please enter a valid height between 100-250cm';
      isValid = false;
    }
    
    if (!weight.trim()) {
      errors.weight = 'Weight is required';
      isValid = false;
    } else if (isNaN(Number(weight)) || Number(weight) < 30 || Number(weight) > 200) {
      errors.weight = 'Please enter a valid weight between 30-200kg';
      isValid = false;
    }
    
    if (!occasion.trim()) {
      errors.occasion = 'Occasion is required';
      isValid = false;
    }
    
    setFormErrors(errors);
    return isValid;
  };

  const handleFileUpload = (file: File) => {
    if (!validateForm()) {
      return;
    }
    
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        sendImageToBackend(file);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const sendImageToBackend = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('height', height);
      formData.append('weight', weight);
      formData.append('occasion', occasion);
      const response = await fetch('http://localhost:5000/api/recommend', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }
      const data = await response.json();
      console.log('Backend response:', data); // Debug log

      // Handle the backend response structure
      let rec = [];
      if (data.success && data.scraped_products && Array.isArray(data.scraped_products)) {
        rec = data.scraped_products.map((item: any, index: number) => ({
          id: index + 1,
          brand: item.brand || 'Unknown Brand',
          description: item.description || item.brand || 'No description available',
          price: item.price || 'Price not available',
          rating: item.rating === 'N/A' ? 'No rating' : item.rating || 'No rating',
          reviews: item.number_of_reviews === 'N/A' ? '0' : item.number_of_reviews || '0',
          image: item.image_link || '',
          link: item.product_link || '#',
        }));
      } else if (data.success) {
        // Handle case where no products were found
        console.log('No scraped products found in response');
        setError('No fashion recommendations found. Please try with a different image.');
      } else {
        // Handle API error response
        const errorMessage = data.error || 'Failed to get recommendations';
        throw new Error(errorMessage);
      }

      console.log('Processed recommendations:', rec);
      setRecommendations(rec);
      
      // Handle LLM text recommendations
      let llmRecs = [];
      if (data.success && data.llm_recommendations) {
        llmRecs = data.llm_recommendations;
        setTextRecommendations(llmRecs);
      }
      
      // Save recommendations to persistent storage
      const userInputs = { height, weight, occasion };
      saveRecommendation(rec, llmRecs, uploadedImage, userInputs);
      
      setShowRecommendationsButton(true);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCameraCapture = async () => {
    try {
      if (!validateForm()) {
        return;
      }
      
      setError(null);
      setIsLoading(true);
      
      console.log('Starting OpenCV camera capture...');
      
      // Prepare form data for backend camera capture
      const formData = new FormData();
      formData.append('height', height);
      formData.append('weight', weight);
      formData.append('occasion', occasion);
      
      // Call backend camera capture endpoint
      const response = await fetch('http://localhost:5000/api/camera-capture', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Camera capture failed');
      }
      
      const data = await response.json();
      console.log('Camera capture response:', data);
      
      if (data.success) {
        // Process the response similar to file upload
        let rec = [];
        if (data.scraped_products && Array.isArray(data.scraped_products)) {
          rec = data.scraped_products.map((item: any, index: number) => ({
            id: index + 1,
            brand: item.brand || 'Unknown Brand',
            description: item.description || item.brand || 'No description available',
            price: item.price || 'Price not available',
            rating: item.rating === 'N/A' ? 'No rating' : item.rating || 'No rating',
            reviews: item.number_of_reviews === 'N/A' ? '0' : item.number_of_reviews || '0',
            image: item.image_link || '',
            link: item.product_link || '#',
          }));
        }
        
        console.log('Processed camera recommendations:', rec);
      
      // Use the captured image from backend if available, otherwise use placeholder
      if (data.captured_image) {
        const capturedImageDataUrl = `data:image/jpeg;base64,${data.captured_image}`;
        setUploadedImage(capturedImageDataUrl);
        console.log('Using captured image from backend');
      } else {
        // Fallback to placeholder if no captured image
        const cameraPlaceholder = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjZjNmNGY2Ii8+CjxwYXRoIGQ9Ik0xMDAgNTBMMTMwIDgwSDExNVY5MEgxMzVWMTUwSDY1VjkwSDg1VjgwSDcwTDEwMCA1MFoiIGZpbGw9IiM5Y2EzYWYiLz4KPHN2Zz4K';
        setUploadedImage(cameraPlaceholder);
        console.log('Using placeholder image (no captured image from backend)');
      }
        
        setRecommendations(rec);
        
        // Handle LLM text recommendations
        let llmRecs = [];
        if (data.success && data.llm_recommendations) {
          llmRecs = data.llm_recommendations;
          setTextRecommendations(llmRecs);
        }
        
        // Save recommendations to persistent storage
        const userInputs = { height, weight, occasion };
        const finalImage = data.captured_image ? `data:image/jpeg;base64,${data.captured_image}` : uploadedImage;
        saveRecommendation(rec, llmRecs, finalImage, userInputs);
        
        setShowRecommendationsButton(true);
        
      } else {
        throw new Error(data.error || 'Camera capture failed');
      }
      
    } catch (err: any) {
      console.error('Camera capture error:', err);
      setError(err.message || 'Camera capture failed');
    } finally {
      setIsLoading(false);
    }
  };

  const tryAgain = () => {
    setUploadedImage(null);
    setIsLoading(false);
    setError(null);
    setRecommendations([]);
    setShowRecommendationsButton(false);
  };

  const viewRecommendations = () => {
    navigate('/recommendations', {
      state: {
        recommendations: recommendations,
        uploadedImage: uploadedImage
      }
    });
  };

  const goToMainPage = () => {
    navigate('/');
  };

  return (
    <>
      <Navbar />
      <section className="py-20 bg-gradient-subtle min-h-screen">
      <div className="container mx-auto px-4">
        {/* Header with Back Button */}
        <div className="flex justify-start mb-8">
          <Button 
            variant="outline" 
            onClick={goToMainPage}
            className="flex items-center space-x-2 hover:bg-primary hover:text-primary-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Main Page</span>
          </Button>
        </div>
        
<div className="text-center mb-10 px-6">
    <h1 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-bold text-gradient leading-[1.3] mb-3 ">
    <span className="text-gradient">Get Your Style Recommendations  </span>
    </h1>
    <p className="text-sm sm:text-base md:text-lg text-muted-foreground max-w-2xl mx-auto leading-normal break-words">
    Upload your photo or use your camera to get personalized fashion suggestions powered by AI.
    </p>
</div>

        {(
          <div className="max-w-4xl mx-auto">
            <Card className="card-glass">
              <CardContent className="p-8">
                {!uploadedImage && !isLoading && (
                  <div className="space-y-8">
                    {/* User Input Section */}
                    <div className="bg-muted/20 rounded-xl p-6">
                      <h3 className="text-xl font-semibold mb-4 text-center">Personal Information</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="height">Height (cm)</Label>
                          <Input
                            id="height"
                            type="number"
                            value={height}
                            onChange={(e) => {
                              setHeight(e.target.value);
                              if (formErrors.height) {
                                setFormErrors({...formErrors, height: ''});
                              }
                            }}
                            placeholder="Enter your height in cm"
                            min="100"
                            max="250"
                            className={formErrors.height ? 'border-red-500' : ''}
                            required
                          />
                          {formErrors.height && (
                            <p className="text-sm text-red-500 mt-1">{formErrors.height}</p>
                          )}
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="weight">Weight (kg)</Label>
                          <Input
                            id="weight"
                            type="number"
                            value={weight}
                            onChange={(e) => {
                              setWeight(e.target.value);
                              if (formErrors.weight) {
                                setFormErrors({...formErrors, weight: ''});
                              }
                            }}
                            placeholder="Enter your weight in kg"
                            min="30"
                            max="200"
                            className={formErrors.weight ? 'border-red-500' : ''}
                            required
                          />
                          {formErrors.weight && (
                            <p className="text-sm text-red-500 mt-1">{formErrors.weight}</p>
                          )}
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="occasion">Occasion</Label>
                          <Input
                            id="occasion"
                            type="text"
                            value={occasion}
                            onChange={(e) => {
                              setOccasion(e.target.value);
                              if (formErrors.occasion) {
                                setFormErrors({...formErrors, occasion: ''});
                              }
                            }}
                            placeholder="e.g., Casual, Party, Wedding"
                            className={formErrors.occasion ? 'border-red-500' : ''}
                            required
                          />
                          {formErrors.occasion && (
                            <p className="text-sm text-red-500 mt-1">{formErrors.occasion}</p>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="border-2 border-dashed rounded-xl p-12 text-center">
                      <div className="w-20 h-20 bg-gradient-primary rounded-xl mx-auto mb-6 flex items-center justify-center">
                        <Upload className="w-10 h-10 text-primary-foreground" />
                      </div>
                      <h3 className="text-2xl font-semibold mb-4">Upload Your Photo</h3>
                      <p className="text-muted-foreground mb-8 text-lg">
                        Drag and drop your image here, or click to browse
                      </p>
                      <Button 
                        size="lg" 
                        className="bg-primary hover:bg-primary/90 px-8 py-3 text-lg"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <Upload className="w-5 h-5 mr-2" />
                        Choose Photo
                      </Button>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handleFileUpload(file);
                        }}
                      />
                    </div>
                    <div className="flex items-center">
                      <div className="flex-1 border-t border-border"></div>
                      <span className="px-4 text-muted-foreground font-medium">OR</span>
                      <div className="flex-1 border-t border-border"></div>
                    </div>
                    <div className="text-center">
                      <div className="w-20 h-20 bg-gradient-primary rounded-xl mx-auto mb-6 flex items-center justify-center">
                        <Camera className="w-10 h-10 text-primary-foreground" />
                      </div>
                      <h3 className="text-2xl font-semibold mb-4">Take Photo from Camera</h3>
                      <p className="text-muted-foreground mb-8 text-lg">
                        Use your device camera to capture a live photo
                      </p>
                      <Button 
                        size="lg" 
                        variant="outline"
                        className="px-8 py-3 text-lg border-primary text-primary hover:bg-primary hover:text-primary-foreground"
                        onClick={handleCameraCapture}
                        disabled={isLoading}
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        {isLoading ? 'Processing...' : 'Open Camera'}
                      </Button>
                    </div>
                    <div className="mt-8 p-6 bg-muted/30 rounded-xl">
                      <h4 className="font-semibold mb-3 flex items-center justify-center">
                        <Eye className="w-4 h-4 mr-2 text-primary" />
                        Privacy First
                      </h4>
                      <p className="text-sm text-muted-foreground text-center">
                        All images are processed securely and never stored. Your privacy is our priority.
                      </p>
                    </div>
                  </div>
                )}

                {isLoading && (
                  <div className="text-center py-12">
                    <div className="w-20 h-20 bg-gradient-primary rounded-xl mx-auto mb-6 flex items-center justify-center animate-pulse">
                      <Loader2 className="w-10 h-10 text-primary-foreground animate-spin" />
                    </div>
                    <h3 className="text-2xl font-semibold mb-4">Generating Recommendations...</h3>
                    <p className="text-muted-foreground text-lg">
                      Our AI is analyzing your photo to find the perfect fashion matches
                    </p>
                    <div className="mt-6 w-full max-w-md mx-auto bg-muted/30 rounded-full h-2">
                      <div className="bg-gradient-primary h-2 rounded-full animate-pulse w-3/4"></div>
                    </div>
                  </div>
                )}
                {error && (
                  <div className="text-center text-red-500 mb-4">{error}</div>
                )}
                
                {/* Show View Recommendations button when recommendations are ready */}
                {showRecommendationsButton && uploadedImage && (
                  <div className="text-center mt-6">
                    <div className="mb-4">
                      <img 
                        src={uploadedImage} 
                        alt="Your uploaded photo"
                        className="w-32 h-32 object-cover rounded-lg mx-auto mb-4"
                      />
                      <p className="text-muted-foreground mb-4">
                        ✅ Your fashion recommendations are ready! ({recommendations.length} items found)
                      </p>
                    </div>
                    
                    {/* LLM Text Recommendations Section */}
                    {textRecommendations && textRecommendations.length > 0 && (
                      <div className="mb-8 max-w-3xl mx-auto">
                        <div className="text-center mb-6">
                          <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-br from-purple-500 to-blue-600 rounded-full mb-3">
                            <Sparkles className="w-6 h-6 text-white" />
                          </div>
                          <h3 className="text-xl font-bold text-gray-800 mb-2">
                            Personalized Style Recommendations
                          </h3>
                          <p className="text-gray-600 text-sm">
                            Curated by AI based on your unique features
                          </p>
                        </div>
                        
                        <div className="grid gap-4">
                          {textRecommendations.slice(0, 3).map((rec: any, index: number) => (
                            <div key={index} className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden hover:shadow-xl transition-shadow duration-300">
                              <div className="p-6">
                                <div className="mb-6">
                                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 mb-4">
                                    <div className="flex-1 min-w-0">
                                      <h4 className="text-lg sm:text-xl font-bold text-gray-900 leading-tight break-words">
                                        <span>{rec['Product Name'] || rec.productName || `Style Recommendation ${index + 1}`}</span>
                                      </h4>
                                    </div>
                                    <div className="flex-shrink-0">
                                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                                        index === 0 
                                          ? 'bg-gradient-to-r from-amber-100 to-yellow-100 text-amber-800 border border-amber-200' 
                                          : 'bg-gray-100 text-gray-700 border border-gray-200'
                                      }`}>
                                        {index === 0 ? '⭐ Top Pick' : `Option ${index + 1}`}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                  {(rec.Fit || rec.fit) && (
                                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-2 border border-blue-100">
                                      <div className="flex items-center">
                                        <div className="w-6 h-6 bg-blue-500 rounded-lg flex items-center justify-center mr-2">
                                          <span className="text-white text-xs font-bold">FIT</span>
                                        </div>
                                        <div>
                                          <p className="text-xs text-blue-600 font-medium uppercase tracking-wide">Size & Fit</p>
                                          <p className="text-blue-800 font-medium text-sm">
                                            {rec.Fit || rec.fit}
                                          </p>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                  
                                  {(rec['Color Palette'] || rec.colorPalette) && (
                                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-2 border border-purple-100">
                                      <div className="flex items-center">
                                        <div className="w-6 h-6 bg-purple-500 rounded-lg flex items-center justify-center mr-2">
                                          <span className="text-white text-xs font-bold">🎨</span>
                                        </div>
                                        <div>
                                          <p className="text-xs text-purple-600 font-medium uppercase tracking-wide">Color Palette</p>
                                          <p className="text-purple-800 font-medium text-sm">
                                            {rec['Color Palette'] || rec.colorPalette}
                                          </p>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        <div className="mt-6 text-center">
                          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-gray-50 to-gray-100 rounded-full border border-gray-200">
                            <span className="text-sm text-gray-600">
                              💡 Search for these styles online or explore our curated collection below
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                    <Button 
                      size="lg" 
                      onClick={viewRecommendations}
                      className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium px-8 py-3 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg"
                    >
                      <Eye className="w-5 h-5 mr-2" />
                      View My Recommendations
                    </Button>
                    <div className="mt-4">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={tryAgain}
                        className="px-6 py-2"
                      >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Try Again
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </section>
    </>
  );
};

export default FashionRecommend; 