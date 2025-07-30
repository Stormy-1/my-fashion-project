import { useEffect, useState, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Upload, Camera, ShoppingBag, Star, Eye, RefreshCw, Loader2 } from 'lucide-react';

const DemoSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    const element = document.querySelector('#demo');
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, []);

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
    setShowRecommendations(false);
    try {
      const formData = new FormData();
      formData.append('image', file);
      const response = await fetch('https://fashion-recommendation-system-2-2pyh.onrender.com/api/recommend', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }
      const data = await response.json();
      // Convert backend response to frontend format
      const rec = [{
        id: 1,
        brand: data.brand,
        description: data.description,
        price: data.price,
        rating: data.rating,
        reviews: data.number_of_reviews,
        image: data.image_link,
        link: data.product_link,
      }];
      setRecommendations(rec);
      setShowRecommendations(true);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCameraCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            setUploadedImage(URL.createObjectURL(blob));
            const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
            sendImageToBackend(file);
          }
        }, 'image/jpeg');
        // Stop camera
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        setIsCameraActive(false);
      }
    }
  };

  const tryAgain = () => {
    setUploadedImage(null);
    setRecommendations([]);
    setShowRecommendations(false);
    setIsLoading(false);
    setIsCameraActive(false);
    setError(null);
  };

  return (
    <section id="demo" className="py-20 bg-gradient-subtle">
      <div className="container mx-auto px-4">
        <div className={`text-center mb-16 px-4 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <h2 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl xl:text-5xl font-bold mb-6 text-gradient break-words leading-tight">
            Get Your Style Recommendations
          </h2>
          <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed break-words">
            Upload your photo and let our AI suggest the perfect fashion items for you
          </p>
        </div>

        {!showRecommendations ? (
          <div className="max-w-4xl mx-auto">
            {/* Upload and Camera Section */}
            <div className={`transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
              <Card className="card-glass">
                <CardContent className="p-8">
                  {!uploadedImage && !isCameraActive && !isLoading && (
                    <div className="space-y-8">
                      {/* Upload Area */}
                      <div 
                        className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
                          isDragOver 
                            ? 'border-primary bg-primary/5 scale-105' 
                            : 'border-border hover:border-primary/50 hover:bg-primary/2'
                        }`}
                        onDragOver={(e) => {
                          e.preventDefault();
                          setIsDragOver(true);
                        }}
                        onDragLeave={() => setIsDragOver(false)}
                        onDrop={(e) => {
                          e.preventDefault();
                          setIsDragOver(false);
                          const files = e.dataTransfer.files;
                          if (files[0]) handleFileUpload(files[0]);
                        }}
                      >
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

                      {/* OR Divider */}
                      <div className="flex items-center">
                        <div className="flex-1 border-t border-border"></div>
                        <span className="px-4 text-muted-foreground font-medium">OR</span>
                        <div className="flex-1 border-t border-border"></div>
                      </div>

                      {/* Camera Section */}
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
                        >
                          <Camera className="w-5 h-5 mr-2" />
                          Open Camera
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

                  {/* Camera View */}
                  {isCameraActive && (
                    <div className="text-center">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        className="w-full max-w-md mx-auto rounded-xl mb-6"
                      />
                      <div className="space-x-4">
                        <Button size="lg" onClick={capturePhoto}>
                          <Camera className="w-5 h-5 mr-2" />
                          Capture Photo
                        </Button>
                        <Button size="lg" variant="outline" onClick={tryAgain}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Loading State */}
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
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          /* Recommendations Display */
          <div className="space-y-8">
            {/* Header with uploaded image */}
            <div className="flex flex-col lg:flex-row gap-8 items-start">
              <div className="lg:w-1/3">
                <Card className="card-glass">
                  <CardContent className="p-6">
                    <h3 className="text-lg font-semibold mb-4">Your Photo</h3>
                    {uploadedImage && (
                      <img 
                        src={uploadedImage} 
                        alt="Your uploaded photo"
                        className="w-full h-64 object-cover rounded-lg"
                      />
                    )}
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="w-full mt-4"
                      onClick={tryAgain}
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Try Again
                    </Button>
                  </CardContent>
                </Card>
              </div>

              <div className="lg:w-2/3">
                <h2 className="text-3xl font-bold mb-4 text-gradient">
                  Perfect Matches for You
                </h2>
                <p className="text-muted-foreground text-lg mb-8">
                  Based on your facial features, here are our AI-curated fashion recommendations
                </p>
              </div>
            </div>

            {/* Product Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {recommendations.map((product, index) => (
                <Card key={product.id} className={`card-glass hover-lift transition-all duration-500 delay-${index * 100}`}>
                  <CardContent className="p-0">
                    <div className="relative">
                      <img 
                        src={product.image} 
                        alt={product.description}
                        className="w-full h-48 object-cover rounded-t-lg"
                      />
                      <div className="absolute top-2 right-2">
                        <Badge variant="secondary" className="bg-background/80 backdrop-blur-sm">
                          {product.brand}
                        </Badge>
                      </div>
                    </div>
                    <div className="p-4">
                      <h3 className="font-semibold mb-2 line-clamp-2">{product.description}</h3>
                      <div className="flex items-center space-x-2 mb-3">
                        <div className="flex items-center">
                          <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                          <span className="text-sm font-medium ml-1">{product.rating}</span>
                        </div>
                        <span className="text-xs text-muted-foreground">({product.reviews} reviews)</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-lg font-bold text-primary">{product.price}</span>
                        <Button 
                          size="sm"
                          onClick={() => window.open(product.link, '_blank')}
                          className="bg-primary hover:bg-primary/90"
                        >
                          Buy Now
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Try Again Button */}
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
        )}
      </div>
    </section>
  );
};

export default DemoSection;