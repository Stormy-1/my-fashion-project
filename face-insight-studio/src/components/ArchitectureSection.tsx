import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Layers, Cpu, Zap, Target } from 'lucide-react';
import modelImage from '@/assets/model-architecture.jpg';

const ArchitectureSection = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.3 }
    );

    const element = document.querySelector('#architecture');
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const architectureDetails = [
    {
      icon: Layers,
      title: "ResNet Backbone",
      description: "Deep residual network with skip connections for optimal gradient flow",
      specs: "50+ layers"
    },
    {
      icon: Cpu,
      title: "PyTorch Lightning",
      description: "Streamlined training with automatic optimization and distributed computing",
      specs: "GPU accelerated"
    },
    {
      icon: Zap,
      title: "Multi-task Learning",
      description: "Simultaneous prediction of age, gender, and 40 facial attributes",
      specs: "42 outputs"
    },
    {
      icon: Target,
      title: "Transfer Learning",
      description: "Pre-trained on ImageNet with fine-tuning on facial data",
      specs: "90%+ accuracy"
    }
  ];

  return (
    <section id="architecture" className="py-20 bg-gradient-subtle">
      <div className="container mx-auto px-4">
        <div className={`text-center mb-16 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Badge variant="secondary" className="mb-4 px-4 py-2 text-sm font-medium border border-accent/20 bg-accent/5">
            <Layers className="w-4 h-4 mr-2" />
            Model Architecture
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gradient">
            Deep Learning Architecture
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Built on ResNet foundations with PyTorch Lightning for scalable, efficient facial analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Architecture Diagram */}
          <div className={`transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`}>
            <Card className="card-glass overflow-hidden">
              <CardContent className="p-0">
                <img 
                  src={modelImage} 
                  alt="Neural Network Architecture"
                  className="w-full h-80 object-cover"
                />
                <div className="p-6">
                  <h3 className="text-xl font-semibold mb-3">ResNet-50 with Custom Head</h3>
                  <p className="text-muted-foreground">
                    Convolutional backbone followed by fully connected layers for multi-attribute classification
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Architecture Details */}
          <div className={`space-y-6 transition-all duration-1000 delay-400 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
            {architectureDetails.map((detail, index) => (
              <Card key={index} className="card-glass hover-lift">
                <CardContent className="p-6">
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-gradient-primary rounded-lg flex items-center justify-center flex-shrink-0">
                      <detail.icon className="w-6 h-6 text-primary-foreground" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-semibold">{detail.title}</h3>
                        <Badge variant="outline" className="border-primary/30 text-primary">
                          {detail.specs}
                        </Badge>
                      </div>
                      <p className="text-muted-foreground">{detail.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Technical Stack */}
        <div className={`mt-16 transition-all duration-1000 delay-600 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Card className="card-glass">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6 text-center">Technical Stack</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="w-16 h-16 bg-gradient-primary rounded-xl mx-auto mb-3 flex items-center justify-center">
                    <span className="text-primary-foreground font-bold">Py</span>
                  </div>
                  <h4 className="font-medium">PyTorch</h4>
                  <p className="text-sm text-muted-foreground">Deep Learning</p>
                </div>
                <div>
                  <div className="w-16 h-16 bg-gradient-primary rounded-xl mx-auto mb-3 flex items-center justify-center">
                    <span className="text-primary-foreground font-bold">⚡</span>
                  </div>
                  <h4 className="font-medium">Lightning</h4>
                  <p className="text-sm text-muted-foreground">Training Framework</p>
                </div>
                <div>
                  <div className="w-16 h-16 bg-gradient-primary rounded-xl mx-auto mb-3 flex items-center justify-center">
                    <span className="text-primary-foreground font-bold">CV</span>
                  </div>
                  <h4 className="font-medium">OpenCV</h4>
                  <p className="text-sm text-muted-foreground">Image Processing</p>
                </div>
                <div>
                  <div className="w-16 h-16 bg-gradient-primary rounded-xl mx-auto mb-3 flex items-center justify-center">
                    <span className="text-primary-foreground font-bold">🔥</span>
                  </div>
                  <h4 className="font-medium">CUDA</h4>
                  <p className="text-sm text-muted-foreground">GPU Acceleration</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default ArchitectureSection;