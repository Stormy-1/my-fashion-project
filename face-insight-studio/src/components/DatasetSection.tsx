import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Database, Users, Image, BarChart3 } from 'lucide-react';

const DatasetSection = () => {
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

    const element = document.querySelector('#dataset');
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const datasetStats = [
    { label: "Total Images", value: "200K+", icon: Image },
    { label: "Unique Identities", value: "10K+", icon: Users },
    { label: "Facial Attributes", value: "40", icon: BarChart3 },
    { label: "Data Quality", value: "99.5%", icon: Database }
  ];

  const attributes = [
    { name: "Age", accuracy: 92 },
    { name: "Gender", accuracy: 96 },
    { name: "Smile", accuracy: 94 },
    { name: "Glasses", accuracy: 97 },
    { name: "Hair Color", accuracy: 89 },
    { name: "Beard", accuracy: 93 },
    { name: "Mustache", accuracy: 91 },
    { name: "Lipstick", accuracy: 88 }
  ];

  const preprocessingSteps = [
    {
      title: "Face Detection",
      description: "MTCNN-based face detection and alignment"
    },
    {
      title: "Normalization", 
      description: "Image resizing to 224x224 with standard normalization"
    },
    {
      title: "Augmentation",
      description: "Random flips, rotations, and color jittering"
    },
    {
      title: "Validation",
      description: "Quality checks and outlier detection"
    }
  ];

  return (
    <section id="dataset" className="py-20">
      <div className="container mx-auto px-4">
        <div className={`text-center mb-16 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Badge variant="secondary" className="mb-4 px-4 py-2 text-sm font-medium border border-primary/20 bg-primary/5">
            <Database className="w-4 h-4 mr-2" />
            Dataset & Training
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gradient">
            CelebA Dataset
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Large-scale celebrity faces dataset with rich annotations for comprehensive facial attribute analysis
          </p>
        </div>

        {/* Dataset Statistics */}
        <div className={`grid grid-cols-1 md:grid-cols-4 gap-6 mb-16 transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          {datasetStats.map((stat, index) => (
            <Card key={index} className="card-glass hover-lift">
              <CardContent className="p-6 text-center">
                <div className="w-16 h-16 bg-gradient-primary rounded-xl mx-auto mb-4 flex items-center justify-center">
                  <stat.icon className="w-8 h-8 text-primary-foreground" />
                </div>
                <div className="text-2xl font-bold text-gradient mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Attribute Accuracy */}
          <div className={`transition-all duration-1000 delay-400 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`}>
            <Card className="card-glass">
              <CardContent className="p-8">
                <h3 className="text-2xl font-semibold mb-6">Model Accuracy by Attribute</h3>
                <div className="space-y-4">
                  {attributes.map((attr, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>{attr.name}</span>
                        <span className="text-muted-foreground">{attr.accuracy}%</span>
                      </div>
                      <Progress value={attr.accuracy} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Preprocessing Pipeline */}
          <div className={`transition-all duration-1000 delay-600 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
            <Card className="card-glass">
              <CardContent className="p-8">
                <h3 className="text-2xl font-semibold mb-6">Preprocessing Pipeline</h3>
                <div className="space-y-6">
                  {preprocessingSteps.map((step, index) => (
                    <div key={index} className="flex items-start space-x-4">
                      <div className="w-8 h-8 bg-gradient-primary rounded-full flex items-center justify-center flex-shrink-0 text-primary-foreground font-bold text-sm">
                        {index + 1}
                      </div>
                      <div>
                        <h4 className="font-semibold mb-1">{step.title}</h4>
                        <p className="text-sm text-muted-foreground">{step.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Training Details */}
        <div className={`mt-16 transition-all duration-1000 delay-800 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Card className="card-glass">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6 text-center">Training Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="text-center">
                  <div className="text-3xl font-bold text-gradient mb-2">Adam</div>
                  <div className="text-sm text-muted-foreground mb-1">Optimizer</div>
                  <div className="text-xs text-muted-foreground">lr=1e-4, β₁=0.9, β₂=0.999</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-gradient mb-2">32</div>
                  <div className="text-sm text-muted-foreground mb-1">Batch Size</div>
                  <div className="text-xs text-muted-foreground">Distributed across 4 GPUs</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-gradient mb-2">50</div>
                  <div className="text-sm text-muted-foreground mb-1">Epochs</div>
                  <div className="text-xs text-muted-foreground">Early stopping at epoch 42</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default DatasetSection;