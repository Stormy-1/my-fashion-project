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

  const projectStats = [
    { label: "GitHub Stars", value: "2.1K", icon: Star },
    { label: "Contributors", value: "12", icon: Users },
    { label: "Model Downloads", value: "50K+", icon: ExternalLink }
  ];

  return (
    <section id="contact" className="py-20">
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

        {/* Project Statistics */}
        <div className={`grid grid-cols-1 md:grid-cols-3 gap-6 mb-16 transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          {projectStats.map((stat, index) => (
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
        <div className={`grid md:grid-cols-2 gap-8 transition-all duration-1000 delay-600 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <Card className="card-glass">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6">Get Involved</h3>
              <div className="space-y-4">
                <Button
                  className="w-full justify-start bg-primary hover:bg-primary/90"
                  onClick={() => window.open('https://github.com', '_blank')}
                >
                  <Github className="w-5 h-5 mr-3" />
                  View Source Code
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start border-primary/20 hover:border-primary hover:bg-primary/10"
                  onClick={() => window.open('mailto:team@faceinsight.ai', '_blank')}
                >
                  <Mail className="w-5 h-5 mr-3" />
                  Contact Team
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start border-primary/20 hover:border-primary hover:bg-primary/10"
                  onClick={() => window.open('https://arxiv.org', '_blank')}
                >
                  <ExternalLink className="w-5 h-5 mr-3" />
                  Read Paper
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="card-glass">
            <CardContent className="p-8">
              <h3 className="text-2xl font-semibold mb-6">Collaboration</h3>
              <p className="text-muted-foreground mb-6">
                Interested in collaborating or have questions about our research? 
                We're always excited to discuss facial analysis applications and methodologies.
              </p>
              <div className="space-y-3">
                <div className="flex items-center text-sm">
                  <Mail className="w-4 h-4 mr-3 text-primary" />
                  <span>team@faceinsight.ai</span>
                </div>
                <div className="flex items-center text-sm">
                  <Github className="w-4 h-4 mr-3 text-primary" />
                  <span>github.com/face-insight-studio</span>
                </div>
                <div className="flex items-center text-sm">
                  <ExternalLink className="w-4 h-4 mr-3 text-primary" />
                  <span>faceinsight.ai</span>
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