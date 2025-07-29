import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Play, Github, ShoppingBag, Camera, Upload } from 'lucide-react';
const heroImage = '/lovable-uploads/585f634e-0ff5-4109-a73e-1027534c119e.png';

const HeroSection = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const scrollToDemo = () => {
    const element = document.querySelector('#demo');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="home" className="relative min-h-screen flex items-center justify-center py-8 md:py-10">

      {/* Floating elements */}
      <div className="absolute inset-0 z-1">
        <div className="absolute top-40 right-20 w-3 h-3 bg-accent rounded-full animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute top-80 right-40 w-3 h-3 bg-accent rounded-full animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-40 left-32 w-3 h-3 bg-primary-glow rounded-full animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute bottom-80 left-44 w-2 h-2 bg-primary-glow rounded-full animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-40 right-32 w-2 h-2 bg-primary-glow rounded-full animate-float" style={{ animationDelay: '2s' }} />
      </div>

      <div className="relative z-10 container mx-auto px-4 text-center">
        <div className={`transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>


          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold  leading-tight py-10">
            <span className="bg-gradient-to-r from-purple-800 to-blue-300 bg-clip-text text-transparent">Aesthet</span>
            <span className="bg-gradient-to-r from-purple-800 to-pink-300 bg-clip-text text-transparent">IQ</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
            Upload your photo and let our advanced AI analyze your facial features to recommend 
            the perfect fashion items tailored just for you.
          </p>

          <div className="flex justify-center items-center mb-4">
            <Button 
              size="lg" 
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-3 text-lg font-medium shadow-glow hover:scale-105 transition-all duration-300"
              onClick={() => window.location.href = '/recommend'}
            >
              <ShoppingBag className="w-5 h-5 mr-2" />
              Get My Fashion Recommendations
            </Button>
          </div>
          {/* Feature highlights - replaced with single card */}
          <div className="max-w-lg mx-auto mt-8 px-4">
            <div className="card-glass rounded-xl p-6 md:p-8 flex flex-col items-center justify-center hover-lift w-full">
              <div className="w-16 h-16 bg-gradient-primary rounded-xl mb-4 flex items-center justify-center">
                <ShoppingBag className="w-8 h-8 text-primary-foreground" />
              </div>
              <div className="text-xl md:text-2xl font-bold mb-2 text-center">Get Fashion Recommendations</div>
              <div className="text-sm md:text-md text-muted-foreground mb-6 text-center leading-relaxed">Upload or capture your photo and get personalized fashion suggestions powered by AI.</div>
              <Button 
                size="lg" 
                className="bg-primary hover:bg-primary/90 text-primary-foreground px-6 md:px-8 py-3 text-base md:text-lg font-medium shadow-glow hover:scale-105 transition-all duration-300 w-full max-w-xs"
                onClick={() => window.location.href = '/recommend'}
              >
                Start Now
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;