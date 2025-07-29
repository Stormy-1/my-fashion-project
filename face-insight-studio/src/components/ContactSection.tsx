import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Github, Mail, Linkedin, ExternalLink, Users, Star } from 'lucide-react';

const ContactSection = () => {
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

    const element = document.querySelector('#contact');
    if (element) observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const teamMembers = [
    {
      name: "Dr. Sarah Chen",
      role: "Lead Researcher",
      avatar: "SC",
      bio: "PhD in Computer Vision, 8+ years in deep learning",
      links: {
        github: "https://github.com",
        linkedin: "https://linkedin.com",
        email: "sarah@faceinsight.ai"
      }
    },
    {
      name: "Alex Rodriguez",
      role: "ML Engineer",
      avatar: "AR",
      bio: "PyTorch expert, specialized in neural network optimization",
      links: {
        github: "https://github.com",
        linkedin: "https://linkedin.com",
        email: "alex@faceinsight.ai"
      }
    },
    {
      name: "Maria Kim",
      role: "Data Scientist",
      avatar: "MK",
      bio: "Dataset curation and preprocessing specialist",
      links: {
        github: "https://github.com",
        linkedin: "https://linkedin.com",
        email: "maria@faceinsight.ai"
      }
    }
  ];

  return (
    <section id="contact" className="py-10 mt:py-6">

      <div className="absolute inset-0 z-1">
        <div className="absolute top-20 left-10 w-2 h-2 bg-primary rounded-full animate-float" />
        <div className="absolute top-40 right-20 w-3 h-3 bg-accent rounded-full animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-32 left-1/4 w-2 h-2 bg-primary-glow rounded-full animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute bottom-20 right-1/3 w-3 h-3 bg-accent rounded-full animate-float" style={{ animationDelay: '0.5s' }} />
      </div>
      <div className="container mx-auto px-4">
        <div className={`text-center mb-16 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Badge variant="secondary" className="mb-4 px-4 py-2 text-sm font-medium border border-primary/20 bg-primary/5">
            <Users className="w-4 h-4 mr-2" />
            Team & Contact
          </Badge>
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gradient">
            Meet the Team
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Passionate researchers and engineers pushing the boundaries of facial analysis technology
          </p>
        </div>

        {/* Team Members */}
        <div className={`grid md:grid-cols-3 gap-8 mb-16 transition-all duration-1000 delay-400 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          {teamMembers.map((member, index) => (
            <Card key={index} className="card-glass hover-lift">
              <CardContent className="p-6 text-center">
                <div className="w-20 h-20 bg-gradient-primary rounded-full mx-auto mb-4 flex items-center justify-center text-primary-foreground font-bold text-xl">
                  {member.avatar}
                </div>
                <h3 className="text-xl font-semibold mb-1">{member.name}</h3>
                <p className="text-sm text-primary mb-3">{member.role}</p>
                <p className="text-sm text-muted-foreground mb-6">{member.bio}</p>
                <div className="flex justify-center space-x-3">
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-10 h-10 p-0 border-primary/20 hover:border-primary hover:bg-primary/10"
                    onClick={() => window.open(member.links.github, '_blank')}
                  >
                    <Github className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-10 h-10 p-0 border-primary/20 hover:border-primary hover:bg-primary/10"
                    onClick={() => window.open(member.links.linkedin, '_blank')}
                  >
                    <Linkedin className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-10 h-10 p-0 border-primary/20 hover:border-primary hover:bg-primary/10"
                    onClick={() => window.open(`mailto:${member.links.email}`, '_blank')}
                  >
                    <Mail className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Contact & Links */}
        <div className={`flex justify-center transition-all duration-1000 delay-600 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Card className="card-glass max-w-4xl w-full relative overflow-hidden group hover:scale-[1.02] transition-all duration-300">
            {/* Gradient background overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-accent/5 to-primary/10 opacity-50 group-hover:opacity-70 transition-opacity duration-300" />
            
            <CardContent className="p-8 md:p-12 relative z-10">
              {/* Header section - more compact */}
              <div className="text-center mb-10">
                <div className="flex items-center justify-center mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-full mr-4 flex items-center justify-center shadow-lg">
                    <Mail className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                    Let's Collaborate
                  </h3>
                </div>
                <div className="w-32 h-1 bg-gradient-to-r from-primary to-accent rounded-full mx-auto mb-6" />
                <p className="text-muted-foreground text-lg leading-relaxed max-w-2xl mx-auto">
                  Interested in collaborating or have questions about our research? 
                  We're always excited to discuss facial analysis applications and methodologies.
                </p>
              </div>
              
              {/* Contact info - horizontal layout */}
              <div className="grid md:grid-cols-2 gap-8 mb-8">
                <div className="group/item hover:bg-primary/5 rounded-xl p-6 transition-all duration-300 hover:shadow-lg">
                  <div className="flex items-center mb-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl flex items-center justify-center mr-4 group-hover/item:from-primary/30 group-hover/item:to-primary/20 transition-all duration-300">
                      <Mail className="w-6 h-6 text-primary" />
                    </div>
                    <h4 className="text-xl font-semibold text-foreground">Email Us</h4>
                  </div>
                  <div className="ml-16">
                    <p className="text-muted-foreground mb-2">Get in touch with our team</p>
                    <div className="space-y-1">
                      <p className="text-sm font-medium text-foreground">neelshroff03@gmail.com</p>
                      <p className="text-sm font-medium text-foreground">mohilmandpe33@gmail.com</p>
                    </div>
                  </div>
                </div>
                
                <div className="group/item hover:bg-primary/5 rounded-xl p-6 transition-all duration-300 hover:shadow-lg">
                  <div className="flex items-center mb-4">
                    <div className="w-12 h-12 bg-gradient-to-br from-accent/20 to-accent/10 rounded-xl flex items-center justify-center mr-4 group-hover/item:from-accent/30 group-hover/item:to-accent/20 transition-all duration-300">
                      <Github className="w-6 h-6 text-accent" />
                    </div>
                    <h4 className="text-xl font-semibold text-foreground">GitHub Repository</h4>
                  </div>
                  <div className="ml-16">
                    <p className="text-muted-foreground mb-3">Explore our open source code</p>
                    <a 
                      href="https://github.com/MOHILMANDAPE15/Fashion-Recommendation-system" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-block"
                    >
                      <Button 
                        variant="outline" 
                        className="bg-gradient-to-r from-primary/10 to-accent/10 hover:from-primary/20 hover:to-accent/20 border-primary/20 hover:border-primary/40 transition-all duration-300 group"
                      >
                        <Github className="w-4 h-4 mr-2 text-foreground/80 group-hover:text-foreground transition-colors" />
                        View on GitHub
                        <ExternalLink className="w-3.5 h-3.5 ml-2 text-foreground/60 group-hover:text-foreground/80 transition-colors" />
                      </Button>
                    </a>
                  </div>
                </div>
              </div>
              
              {/* Call to action - horizontal */}
              <div className="text-center border-t border-primary/10 pt-8">
                <div className="flex items-center justify-center space-x-6">
                  <p className="text-muted-foreground">
                    Ready to start a conversation?
                  </p>
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                    <div className="w-2 h-2 bg-accent rounded-full animate-pulse" style={{animationDelay: '0.5s'}} />
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse" style={{animationDelay: '1s'}} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;