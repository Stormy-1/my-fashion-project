import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Upload, Camera, ShoppingBag, Star, Eye, RefreshCw, Loader2, StarHalf } from 'lucide-react';

const FashionRecommend = () => {
  const navigate = useNavigate();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [showRecommendationsButton, setShowRecommendationsButton] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [height, setHeight] = useState<string>('170');
  const [weight, setWeight] = useState<string>('70');
  const [occasion, setOccasion] = useState<string>('casual');
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

  const handleFileUpload = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        sendImageToBackend(file);
      };
      reader.readAsDataURL(file);
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
      setShowRecommendationsButton(true);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCameraCapture = async () => {
    try {
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
        
        // Set a placeholder image to indicate camera was used
        const cameraPlaceholder = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjZjNmNGY2Ii8+CjxwYXRoIGQ9Ik0xMDAgNTBMMTMwIDgwSDExNVY5MEgxMzVWMTUwSDY1VjkwSDg1VjgwSDcwTDEwMCA1MFoiIGZpbGw9IiM5Y2EzYWYiLz4KPHN2Zz4K';
        setUploadedImage(cameraPlaceholder);
        
        setRecommendations(rec);
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

  return (
    <section className="py-20 bg-gradient-subtle min-h-screen">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <Badge variant="secondary" className="mb-4 px-4 py-2 text-sm font-medium border border-accent/20 bg-accent/5">
            <ShoppingBag className="w-4 h-4 mr-2" />
            Fashion AI Recommendation
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gradient">
            Get Your Style Recommendations
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
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
                            onChange={(e) => setHeight(e.target.value)}
                            placeholder="170"
                            min="100"
                            max="250"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="weight">Weight (kg)</Label>
                          <Input
                            id="weight"
                            type="number"
                            value={weight}
                            onChange={(e) => setWeight(e.target.value)}
                            placeholder="70"
                            min="30"
                            max="200"
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="occasion">Occasion</Label>
                          <Select value={occasion} onValueChange={setOccasion}>
                            <SelectTrigger>
                              <SelectValue placeholder="Select occasion" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="casual">Casual</SelectItem>
                              <SelectItem value="formal">Formal</SelectItem>
                              <SelectItem value="party">Party</SelectItem>
                              <SelectItem value="business">Business</SelectItem>
                              <SelectItem value="sports">Sports</SelectItem>
                              <SelectItem value="wedding">Wedding</SelectItem>
                            </SelectContent>
                          </Select>
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
  );
};

export default FashionRecommend; 