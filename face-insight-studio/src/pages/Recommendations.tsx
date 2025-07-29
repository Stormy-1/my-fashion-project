import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ShoppingBag, Star, ArrowLeft, StarHalf, RefreshCw, Sparkles, Shirt, Palette, Info, CheckCircle, Clock, Trash2 } from 'lucide-react';
import Navbar from '@/components/Navbar';
import { Separator } from '@/components/ui/separator';
import { getRecommendationHistory, deleteRecommendation, type StoredRecommendation } from '@/utils/recommendationStorage';

interface Recommendation {
  id?: number;
  brand?: string;
  description?: string;
  price?: string;
  rating?: string;
  reviews?: string;
  image?: string;
  link?: string;
}

interface TextRecommendation {
  productName?: string;
  fit?: string;
  colorPalette?: string;
  gender?: string;
}

const Recommendations = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [allRecommendations, setAllRecommendations] = useState<StoredRecommendation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Load all recommendations from storage on component mount
  useEffect(() => {
    const loadRecommendations = () => {
      try {
        const history = getRecommendationHistory();
        setAllRecommendations(history);
      } catch (error) {
        console.error('Error loading recommendation history:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadRecommendations();
  }, []);
  
  // Handle deletion of recommendations
  const handleDeleteRecommendation = (id: string) => {
    deleteRecommendation(id);
    setAllRecommendations(prev => prev.filter(rec => rec.id !== id));
  };

  // Helper function to convert color names to hex codes
  const getColorCode = (colorName: string): string => {
    const colors: { [key: string]: string } = {
      'black': '#000000',
      'white': '#FFFFFF',
      'red': '#FF0000',
      'blue': '#0000FF',
      'green': '#008000',
      'yellow': '#FFFF00',
      'purple': '#800080',
      'pink': '#FFC0CB',
      'orange': '#FFA500',
      'brown': '#A52A2A',
      'gray': '#808080',
      'navy': '#000080',
      'beige': '#F5F5DC',
      'burgundy': '#800020',
      'gold': '#FFD700',
      'silver': '#C0C0C0',
      'cream': '#FFFDD0',
      'maroon': '#800000',
      'teal': '#008080',
      'olive': '#808000'
    };

    // Convert to lowercase and remove any non-alphabetic characters
    const cleanColor = colorName.toLowerCase().replace(/[^a-z]/g, '');
    return colors[cleanColor] || '#E5E7EB';
  };

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

  // Helper function to clean brand names
  const cleanBrandName = (brand: string): string => {
    if (!brand) return 'Brand';
    
    // Remove common unwanted text patterns
    let cleaned = brand
      .replace(/Amazon\s*Brand\s*-\s*Symbol/gi, 'Amazon')
      .replace(/\s*Brand\s*-\s*Symbol/gi, '')
      .replace(/\s*-\s*Symbol/gi, '')
      .replace(/\s*Symbol/gi, '')
      .replace(/\s*Brand\s*/gi, '')
      .trim();
    
    // If nothing left after cleaning, return a default
    if (!cleaned || cleaned.length === 0) {
      return 'Amazon';
    }
    
    // Capitalize first letter
    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
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

  const goBack = () => {
    navigate('/recommend');
  };

  const goToMainPage = () => {
    navigate('/');
  };

  const tryAgain = () => {
    navigate('/recommend');
  };

  return (
    <>
      <Navbar />
      <section className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 py-20">
        <div className="container mx-auto px-4">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold mb-6 text-gradient">
              Your Fashion History
            </h1>
            <p className="text-muted-foreground text-sm sm:text-base md:text-lg mb-8 max-w-4xl mx-auto leading-relaxed">
              All your personalized fashion recommendations, organized by date
            </p>
            <Badge variant="outline" className="mb-4">
              <Clock className="w-4 h-4 mr-2" />
              {allRecommendations.length} recommendation{allRecommendations.length !== 1 ? 's' : ''} found
            </Badge>
          </div>

          {/* Recommendation History */}
          <div className="space-y-12">
            {allRecommendations.map((storedRec, sessionIndex) => (
              <div key={storedRec.id} className="bg-white/50 backdrop-blur-sm rounded-xl p-6 border border-gray-200 shadow-lg">
                {/* Session Header */}
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200">
                  <div className="flex items-center space-x-4">
                    <Badge className="bg-gradient-to-r from-blue-500 to-purple-600 text-white">
                      Session #{sessionIndex + 1}
                    </Badge>
                    <div className="flex items-center text-sm text-muted-foreground">
                      <Clock className="w-4 h-4 mr-2" />
                      {storedRec.date}
                    </div>
                    {storedRec.userInputs && (
                      <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                        {storedRec.userInputs.height && <span>Height: {storedRec.userInputs.height}</span>}
                        {storedRec.userInputs.weight && <span>Weight: {storedRec.userInputs.weight}</span>}
                        {storedRec.userInputs.occasion && <span>Occasion: {storedRec.userInputs.occasion}</span>}
                      </div>
                    )}
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => handleDeleteRecommendation(storedRec.id)}
                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>

                {/* Uploaded Image */}
                {storedRec.uploadedImage && (
                  <div className="mb-6 text-center">
                    <h4 className="text-sm font-medium mb-2 text-gray-700">Your Photo</h4>
                    <div className="inline-block p-2 bg-white rounded-lg shadow-sm">
                      <img 
                        src={storedRec.uploadedImage} 
                        alt="Uploaded" 
                        className="w-24 h-24 object-cover rounded-lg"
                      />
                    </div>
                  </div>
                )}

                {/* LLM Recommendations */}
                {storedRec.textRecommendations && storedRec.textRecommendations.length > 0 && (
                  <div className="mb-8">
                    <h3 className="text-lg font-bold mb-4 text-gray-800 flex items-center">
                      <Sparkles className="w-5 h-5 mr-2 text-purple-600" />
                      AI Style Recommendations
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {storedRec.textRecommendations.map((rec: any, index: number) => (
                        <Card key={index} className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-blue-50">
                          <CardContent className="p-4">
                            <div className="flex items-center mb-3">
                              <Shirt className="w-5 h-5 text-purple-600 mr-2" />
                              <h4 className="font-semibold text-sm text-gray-800">{rec.productName || rec['Product Name']}</h4>
                            </div>
                            <div className="space-y-2 text-xs">
                              {(rec.fit || rec['Fit']) && (
                                <div className="flex items-center">
                                  <Info className="w-3 h-3 text-blue-600 mr-1 flex-shrink-0" />
                                  <span className="text-gray-600">{rec.fit || rec['Fit']}</span>
                                </div>
                              )}
                              {(rec.colorPalette || rec['Color Palette']) && (
                                <div className="flex items-center">
                                  <Palette className="w-3 h-3 text-green-600 mr-1 flex-shrink-0" />
                                  <span className="text-gray-600">{rec.colorPalette || rec['Color Palette']}</span>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {/* Product Recommendations */}
                {storedRec.recommendations && storedRec.recommendations.length > 0 && (
                  <div>
                    <h3 className="text-lg font-bold mb-4 text-gray-800 flex items-center">
                      <ShoppingBag className="w-5 h-5 mr-2 text-blue-600" />
                      Product Recommendations ({storedRec.recommendations.length} items)
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {storedRec.recommendations.map((product: any, index: number) => (
                        <Card key={product.id || index} className="hover:shadow-lg transition-shadow">
                          <CardContent className="p-0">
                            <div className="relative">
                              <img 
                                src={product.image} 
                                alt={product.description}
                                className="w-full h-40 object-contain rounded-t-lg bg-white"
                              />
                              <div className="absolute top-2 right-2 z-10">
                                <Badge variant="secondary" className="bg-white/95 backdrop-blur-sm text-xs shadow-md border border-gray-200 px-2 py-1 whitespace-nowrap">
                                  {cleanBrandName(product.brand)}
                                </Badge>
                              </div>
                            </div>
                            <div className="p-3">
                              <h4 className="font-medium mb-2 line-clamp-2 text-sm text-gray-800">{product.description}</h4>
                              <div className="flex items-center justify-between mb-2">
                                {renderStarRating(product.rating)}
                                <span className="text-xs text-muted-foreground">({product.reviews})</span>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="text-sm">
                                  {formatPrice(product.price)}
                                </div>
                                <Button 
                                  size="sm"
                                  onClick={() => window.open(product.link, '_blank')}
                                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white text-xs px-3 py-1"
                                >
                                  <ShoppingBag className="w-3 h-3 mr-1" />
                                  Buy
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="text-center mt-12 space-x-4">
            <Button 
              size="lg" 
              variant="outline"
              onClick={() => navigate('/')}
              className="px-8 py-3"
            >
              <ArrowLeft className="w-5 h-5 mr-2" />
              Go Back to Main Page
            </Button>
            <Button 
              size="lg" 
              onClick={() => navigate('/recommend')}
              className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
            >
              <RefreshCw className="w-5 h-5 mr-2" />
              Get New Recommendations
            </Button>
          </div>
        </div>
      </section>
    </>
  );
};

export default Recommendations;