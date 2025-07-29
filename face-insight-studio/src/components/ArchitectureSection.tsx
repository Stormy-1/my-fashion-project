import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Layers, 
  Cpu, 
  Zap, 
  Target, 
  Globe, 
  Brain, 
  Camera, 
  ShoppingBag,
  Server,
  Smartphone,
  Database,
  ArrowRight,
  Upload
} from 'lucide-react';

const ArchitectureSection = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Set visible immediately to ensure content shows
    setIsVisible(true);
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const timer = setTimeout(() => {
      const element = document.querySelector('#architecture');
      if (element) observer.observe(element);
    }, 100);

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, []);

  const frontendStack = [
    {
      icon: Smartphone,
      title: "React 18 + TypeScript",
      description: "Modern React with TypeScript for type safety and better development experience",
      tech: "Frontend Framework"
    },
    {
      icon: Zap,
      title: "Vite + Tailwind CSS",
      description: "Lightning-fast build tool with utility-first CSS framework",
      tech: "Build & Styling"
    },
    {
      icon: Layers,
      title: "shadcn/ui Components",
      description: "Beautiful, accessible UI components built on Radix UI primitives",
      tech: "UI Library"
    }
  ];

  const backendStack = [
    {
      icon: Server,
      title: "Flask API Server",
      description: "Python web framework with CORS support for cross-origin requests",
      tech: "Web Server"
    },
    {
      icon: Brain,
      title: "PyTorch ML Models",
      description: "Deep learning models for age, gender, and facial feature detection",
      tech: "Machine Learning"
    },
    {
      icon: Target,
      title: "Hugging Face LLM",
      description: "Large language model integration for creative fashion recommendations",
      tech: "AI Recommendations"
    },
    {
      icon: Globe,
      title: "Web Scraping Engine",
      description: "Selenium-based scraping for real-time product discovery from Amazon",
      tech: "Data Collection"
    }
  ];

  const pipelineFlow = [
    {
      icon: Camera,
      title: "Image Capture & Preprocessing",
      description: "High-resolution image capture via file upload or real-time OpenCV camera integration with automatic face detection and cropping",
      details: "• File upload support (JPG, PNG, WEBP)\n• OpenCV real-time camera capture\n• Automatic face detection & alignment\n• Image preprocessing & normalization",
      tech: "OpenCV • React • Base64 Encoding",
      step: "1"
    },
    {
      icon: Brain,
      title: "Deep Learning Analysis",
      description: "Advanced PyTorch models analyze facial features, predict age/gender, and extract 16+ facial attributes with high accuracy",
      details: "• Age prediction (±3 years accuracy)\n• Gender classification (95%+ accuracy)\n• 16 facial attributes extraction\n• BMI calculation from facial structure",
      tech: "PyTorch • ResNet Architecture • CelebA Dataset",
      step: "2"
    },
    {
      icon: Zap,
      title: "AI Fashion Recommendation",
      description: "Hugging Face LLM generates personalized, creative fashion recommendations based on facial analysis and user preferences",
      details: "• Context-aware outfit suggestions\n• Occasion-based styling (casual, formal, party)\n• Color palette recommendations\n• Themed fashion concepts",
      tech: "Hugging Face • LangChain • GPT Integration",
      step: "3"
    },
    {
      icon: ShoppingBag,
      title: "Product Discovery & Matching",
      description: "Intelligent web scraping engine discovers and matches real products from e-commerce platforms with price comparison",
      details: "• Amazon product scraping\n• Price & rating aggregation\n• Brand & style matching\n• Real-time availability check",
      tech: "Selenium • Beautiful Soup • Product APIs",
      step: "4"
    }
  ];

  const techStack = [
    { name: "React", category: "Frontend", icon: "⚛️" },
    { name: "TypeScript", category: "Language", icon: "📘" },
    { name: "Flask", category: "Backend", icon: "🐍" },
    { name: "PyTorch", category: "ML", icon: "🔥" },
    { name: "OpenCV", category: "Vision", icon: "👁️" },
    { name: "Selenium", category: "Scraping", icon: "🕷️" },
    { name: "Hugging Face", category: "LLM", icon: "🤗" },
    { name: "Tailwind", category: "Styling", icon: "🎨" }
  ];

  return (
    <section id="architecture" className="py-10">
      <div className="container mx-auto px-2">
        <div className={`text-center mb-8 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Badge variant="secondary" className=" px-4 py-2 text-sm font-medium border border-accent/20 bg-accent/5">
            <Layers className="w-4 h-4 mr-2" />
            System Architecture
          </Badge>
          
          
        </div>

        {/* Pipeline Flow */}
        <div className={`mb-16 transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold mb-4 text-gradient">AI Processing Pipeline</h3>
            <p className="text-muted-foreground max-w-2xl mx-auto">Four intelligent stages that transform your photo into personalized fashion recommendations</p>
          </div>
          
          {/* Desktop Horizontal Flow */}
          <div className="hidden md:block">
            <div className="relative">
              {/* Progress Line */}
              <div className="absolute top-20 left-8 right-8 h-0.5 bg-gradient-to-r from-primary via-accent to-primary opacity-20" />
              
              <div className="grid grid-cols-4 gap-6 relative items-start">
                {pipelineFlow.map((step, index) => (
                  <div key={index} className="relative">
                    {/* Step Card */}
                    <Card className="card-glass hover-lift transition-all duration-300 hover:shadow-lg">
                      <CardContent className="p-6 text-center">
                        {/* Icon */}
                        <div className="w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-full mx-auto mb-4 flex items-center justify-center relative">
                          <step.icon className="w-8 h-8 text-primary-foreground" />
                          <div className="absolute -top-2 -right-2 w-6 h-6 bg-background border-2 border-primary rounded-full flex items-center justify-center">
                            <span className="text-xs font-bold text-primary">{step.step}</span>
                          </div>
                        </div>
                        
                        {/* Title */}
                        <h4 className="font-semibold text-lg mb-3 h-14 flex items-center justify-center">{step.title}</h4>
                        
                        {/* Description */}
                        <p className="text-sm text-muted-foreground mb-4 leading-relaxed h-24 overflow-hidden">{step.description}</p>
                        
                        {/* Expandable Details */}
                        <details className="group">
                          <summary className="cursor-pointer text-primary text-sm font-medium hover:text-primary/80 transition-colors">
                            View Details ▼
                          </summary>
                          <div className="mt-4 space-y-3 text-left h-64 overflow-y-auto">
                            <div className="bg-muted/30 rounded-lg p-3">
                              <h5 className="font-medium text-xs mb-2 text-primary">Features:</h5>
                              <div className="text-xs text-muted-foreground space-y-1">
                                {step.details.split('\n').map((line, i) => (
                                  <div key={i} className="flex items-start gap-2">
                                    <div className="w-1 h-1 bg-primary rounded-full mt-1.5 flex-shrink-0" />
                                    <span>{line.replace('• ', '')}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div className="bg-accent/10 rounded-lg p-3">
                              <h5 className="font-medium text-xs mb-2 text-accent-foreground">Tech:</h5>
                              <p className="text-xs font-mono text-muted-foreground">{step.tech}</p>
                            </div>
                          </div>
                        </details>
                      </CardContent>
                    </Card>
                  </div>
                ))}
                
                {/* Arrows positioned between cards */}
                <div className="absolute top-20 left-1/4 transform -translate-x-1/2 z-10">
                  <div className="w-6 h-6 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center shadow-md">
                    <ArrowRight className="w-3 h-3 text-primary-foreground" />
                  </div>
                </div>
                <div className="absolute top-20 left-2/4 transform -translate-x-1/2 z-10">
                  <div className="w-6 h-6 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center shadow-md">
                    <ArrowRight className="w-3 h-3 text-primary-foreground" />
                  </div>
                </div>
                <div className="absolute top-20 left-3/4 transform -translate-x-1/2 z-10">
                  <div className="w-6 h-6 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center shadow-md">
                    <ArrowRight className="w-3 h-3 text-primary-foreground" />
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Mobile Vertical Flow */}
          <div className="md:hidden space-y-6">
            {pipelineFlow.map((step, index) => (
              <div key={index}>
                <Card className="card-glass">
                  <CardContent className="p-6">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center flex-shrink-0">
                        <step.icon className="w-6 h-6 text-primary-foreground" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline" className="border-primary/30 text-primary text-xs">
                            {step.step}
                          </Badge>
                          <h4 className="font-semibold">{step.title}</h4>
                        </div>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                {index < pipelineFlow.length - 1 && (
                  <div className="flex justify-center py-2">
                    <div className="w-0.5 h-4 bg-gradient-to-b from-primary to-accent rounded-full" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Frontend & Backend Architecture */}
        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Frontend */}
          <div className={`transition-all duration-1000 delay-400 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`}>
            <h3 className="text-2xl font-semibold mb-6 text-center">Frontend Architecture</h3>
            <div className="space-y-4">
              {frontendStack.map((item, index) => (
                <Card key={index} className="card-glass hover-lift">
                  <CardContent className="p-6">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center flex-shrink-0">
                        <item.icon className="w-6 h-6 text-primary-foreground" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold">{item.title}</h4>
                          <Badge variant="outline" className="border-primary/30 text-primary text-xs">
                            {item.tech}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Backend */}
          <div className={`transition-all duration-1000 delay-600 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
            <h3 className="text-2xl font-semibold mb-6 text-center">Backend Architecture</h3>
            <div className="space-y-4">
              {backendStack.map((item, index) => (
                <Card key={index} className="card-glass hover-lift">
                  <CardContent className="p-6">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center flex-shrink-0">
                        <item.icon className="w-6 h-6 text-primary-foreground" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold">{item.title}</h4>
                          <Badge variant="outline" className="border-primary/30 text-primary text-xs">
                            {item.tech}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>

        {/* API Endpoints */}
        <div className={`mt-16 transition-all duration-1000 delay-800 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="text-center mb-8">
            <h3 className="text-3xl font-bold mb-4 text-gradient">API Endpoints</h3>
            <p className="text-muted-foreground max-w-2xl mx-auto">Comprehensive REST API for fashion recommendation services with real-time processing capabilities</p>
          </div>
          
          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* Main Recommendation Endpoint */}
            <Card className="card-glass hover-lift transition-all duration-300 hover:shadow-xl">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                    <Upload className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <Badge variant="secondary" className="mb-1 bg-blue-100 text-blue-800 border-blue-200">POST</Badge>
                    <h4 className="font-bold text-lg">/api/recommend</h4>
                  </div>
                </div>
                
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  Primary endpoint for fashion recommendations. Processes uploaded images through ML pipeline and returns personalized product suggestions.
                </p>
                
                <div className="space-y-4">
                  <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4">
                    <h5 className="font-semibold text-sm mb-3 text-blue-700 dark:text-blue-300 flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Request Parameters:
                    </h5>
                    <div className="text-xs space-y-2 font-mono">
                      <div className="flex justify-between"><span className="text-blue-600">image</span><span className="text-muted-foreground">File (JPG, PNG, WEBP) - Max 16MB</span></div>
                      <div className="flex justify-between"><span className="text-blue-600">height</span><span className="text-muted-foreground">Float (100-250 cm)</span></div>
                      <div className="flex justify-between"><span className="text-blue-600">weight</span><span className="text-muted-foreground">Float (30-300 kg)</span></div>
                      <div className="flex justify-between"><span className="text-blue-600">occasion</span><span className="text-muted-foreground">String (casual, formal, party)</span></div>
                    </div>
                  </div>
                  
                  <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4">
                    <h5 className="font-semibold text-sm mb-3 text-green-700 dark:text-green-300 flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Response Format:
                    </h5>
                    <div className="text-xs font-mono bg-background/50 p-3 rounded border">
                      {`{
  "success": true,
  "scraped_products": [
    {
      "brand": "Nike",
      "description": "Running Shoes",
      "price": "₹2,999",
      "rating": "4.5",
      "image_link": "...",
      "product_link": "..."
    }
  ],
  "user_info": {...},
  "facial_features": {...}
}`}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Camera Capture Endpoint */}
            <Card className="card-glass hover-lift transition-all duration-300 hover:shadow-xl">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg flex items-center justify-center">
                    <Camera className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <Badge variant="secondary" className="mb-1 bg-green-100 text-green-800 border-green-200">POST</Badge>
                    <h4 className="font-bold text-lg">/api/camera-capture</h4>
                  </div>
                </div>
                
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  Real-time camera capture endpoint using OpenCV. Captures photo from webcam and processes it for fashion recommendations.
                </p>
                
                <div className="space-y-4">
                  <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4">
                    <h5 className="font-semibold text-sm mb-3 text-green-700 dark:text-green-300 flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Request Parameters:
                    </h5>
                    <div className="text-xs space-y-2 font-mono">
                      <div className="flex justify-between"><span className="text-green-600">height</span><span className="text-muted-foreground">Float (default: 170 cm)</span></div>
                      <div className="flex justify-between"><span className="text-green-600">weight</span><span className="text-muted-foreground">Float (default: 70 kg)</span></div>
                      <div className="flex justify-between"><span className="text-green-600">occasion</span><span className="text-muted-foreground">String (default: casual)</span></div>
                    </div>
                  </div>
                  
                  <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4">
                    <h5 className="font-semibold text-sm mb-3 text-blue-700 dark:text-blue-300 flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Camera Features:
                    </h5>
                    <div className="text-xs space-y-1 text-muted-foreground">
                      <div className="flex items-center gap-2"><div className="w-1 h-1 bg-blue-500 rounded-full"></div>640x480 resolution capture</div>
                      <div className="flex items-center gap-2"><div className="w-1 h-1 bg-blue-500 rounded-full"></div>30 FPS real-time processing</div>
                      <div className="flex items-center gap-2"><div className="w-1 h-1 bg-blue-500 rounded-full"></div>Automatic face detection</div>
                      <div className="flex items-center gap-2"><div className="w-1 h-1 bg-blue-500 rounded-full"></div>Base64 image encoding</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

        </div>
      </div>
    </section>
  );
};

export default ArchitectureSection;