import { useLocation, useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ShoppingBag, Star, ArrowLeft, StarHalf, RefreshCw } from 'lucide-react';

const Recommendations = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { recommendations, uploadedImage } = location.state || { recommendations: [], uploadedImage: null };

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

  const goBack = () => {
    navigate('/recommend');
  };

  const tryAgain = () => {
    navigate('/recommend');
  };

  if (!recommendations || recommendations.length === 0) {
    return (
      <section className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100 py-12">
        <div className="container mx-auto px-4">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-8 text-gradient">No Recommendations Found</h1>
            <p className="text-muted-foreground text-lg mb-8">
              Please go back and upload an image to get recommendations.
            </p>
            <Button onClick={goBack} className="px-8 py-3">
              <ArrowLeft className="w-5 h-5 mr-2" />
              Go Back
            </Button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100 py-12">
      <div className="container mx-auto px-4">
        {/* Header with back button */}
        <div className="relative mb-8">
          <div className="absolute left-0 top-0">
            <Button 
              variant="outline" 
              onClick={goBack}
              className="flex items-center"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Upload
            </Button>
          </div>
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gradient">Your Fashion Recommendations</h1>
          </div>
        </div>

        {/* User's uploaded image */}
        {uploadedImage && (
          <div className="flex justify-center mb-8">
            <Card className="card-glass">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold mb-4 text-center">Your Photo</h3>
                <img 
                  src={uploadedImage} 
                  alt="Your uploaded photo"
                  className="w-32 h-32 object-cover rounded-lg mx-auto"
                />
              </CardContent>
            </Card>
          </div>
        )}
        
        {/* Recommendations Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold mb-4 text-gradient">
            Perfect Matches for You
          </h2>
          <p className="text-muted-foreground text-lg mb-8">
            Based on your facial features, here are our AI-curated fashion recommendations
          </p>
        </div>

        {/* Recommendations Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {recommendations.map((product, index) => (
            <Card key={product.id} className={`card-glass hover-lift transition-all duration-500 delay-${index * 100}`}>
              <CardContent className="p-0">
                <div className="relative">
                  <img 
                    src={product.image} 
                    alt={product.description}
                    className="w-full h-auto object-contain rounded-t-lg bg-white"
                    style={{ minHeight: '200px', maxHeight: '300px' }}
                  />
                  <div className="absolute top-2 right-2">
                    <Badge variant="secondary" className="bg-background/80 backdrop-blur-sm">
                      {product.brand}
                    </Badge>
                  </div>
                </div>
                <div className="p-4">
                  <h3 className="font-semibold mb-3 line-clamp-2 text-gray-800">{product.description}</h3>
                  
                  {/* Rating Section */}
                  <div className="flex items-center justify-between mb-3">
                    {renderStarRating(product.rating)}
                    <span className="text-xs text-muted-foreground">({product.reviews} reviews)</span>
                  </div>
                  
                  {/* Price and Buy Button */}
                  <div className="flex items-center justify-between mt-4">
                    <div className="flex flex-col">
                      {formatPrice(product.price)}
                    </div>
                    <Button 
                      size="sm"
                      onClick={() => window.open(product.link, '_blank')}
                      className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium px-4 py-2 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg"
                    >
                      <ShoppingBag className="w-4 h-4 mr-1" />
                      Buy Now
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="text-center">
          <Button 
            size="lg" 
            variant="outline"
            onClick={tryAgain}
            className="px-8 py-3"
          >
            <RefreshCw className="w-5 h-5 mr-2" />
            Try with Another Photo
          </Button>
        </div>
      </div>
    </section>
  );
};

export default Recommendations;
