import { 
  Github, 
  Heart, 
  Mail, 
  Phone, 
  MapPin, 
  Twitter, 
  Instagram, 
  Linkedin, 
  Camera, 
  Sparkles, 
  Shield, 
  Zap,
  Users,
  Star
} from 'lucide-react';
import logo from '@/assets/logo-1.png';

const Footer = () => {
  return (
    <footer id="contact" className="border-t border-border/50">
      {/* Main Footer Content */}
      <div className="container mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          
          {/* Brand Section */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <img 
                src={logo} 
                alt="AesthetIQ Logo" 
                className="h-10 w-auto rounded-lg"
              />
              <h1 className="text-xl md:text-xl lg:text-xl font-bold  leading-tight py-10">
            <span className="bg-gradient-to-r from-purple-800 to-blue-300 bg-clip-text text-transparent">Aesthet</span>
            <span className="bg-gradient-to-r from-purple-800 to-pink-300 bg-clip-text text-transparent">IQ</span>
          </h1>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              Revolutionary AI-powered fashion recommendations using facial analysis and machine learning. 
              Discover your perfect style with cutting-edge technology.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="w-10 h-10 bg-primary/20 hover:bg-primary/30 rounded-full flex items-center justify-center transition-colors">
                <Twitter className="w-5 h-5 text-foreground" />
              </a>
              <a href="#" className="w-10 h-10 bg-primary/20 hover:bg-primary/30 rounded-full flex items-center justify-center transition-colors">
                <Instagram className="w-5 h-5 text-foreground" />
              </a>
              <a href="#" className="w-10 h-10 bg-primary/20 hover:bg-primary/30 rounded-full flex items-center justify-center transition-colors">
                <Linkedin className="w-5 h-5 text-foreground" />
              </a>
              <a 
                href="https://github.com/MOHILMANDAPE15/Fashion-Recommendation-system" 
                target="_blank" 
                rel="noopener noreferrer"
                className="w-10 h-10 bg-primary/20 hover:bg-primary/30 rounded-full flex items-center justify-center transition-colors"
              >
                <Github className="w-5 h-5 text-foreground" />
              </a>
            </div>
          </div>

          {/* Features Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
              <Sparkles className="w-5 h-5 mr-2 text-primary" />
              Features
            </h3>
            <ul className="space-y-3 text-muted-foreground">
              <li className="flex items-center hover:text-primary transition-colors cursor-pointer">
                <Zap className="w-4 h-4 mr-2 text-primary" />
                AI Facial Analysis
              </li>
              <li className="flex items-center hover:text-primary transition-colors cursor-pointer">
                <Camera className="w-4 h-4 mr-2 text-primary" />
                Real-time Camera Capture
              </li>
              <li className="flex items-center hover:text-primary transition-colors cursor-pointer">
                <Star className="w-4 h-4 mr-2 text-primary" />
                Personalized Recommendations
              </li>
              <li className="flex items-center hover:text-primary transition-colors cursor-pointer">
                <Shield className="w-4 h-4 mr-2 text-primary" />
                Privacy Protected
              </li>
              <li className="flex items-center hover:text-primary transition-colors cursor-pointer">
                <Users className="w-4 h-4 mr-2 text-primary" />
                Multi-user Support
              </li>
            </ul>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-foreground mb-4">Quick Links</h3>
            <ul className="space-y-3 text-muted-foreground">
              <li>
                <a href="/" className="hover:text-primary transition-colors">
                  Home
                </a>
              </li>
              <li>
                <a href="/recommend" className="hover:text-primary transition-colors">
                  Get Recommendations
                </a>
              </li>
              <li>
                <a href="/recommendations" className="hover:text-primary transition-colors">
                  View History
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  About Us
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition-colors">
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>

          {/* Contact Info */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-foreground mb-4">Get in Touch</h3>
            <div className="space-y-4 text-muted-foreground">
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2">Email Us</h4>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <Mail className="w-4 h-4 mr-3 text-primary" />
                    <a href="mailto:neelshroff03@gmail.com" className="text-sm hover:text-primary transition-colors">
                      neelshroff03@gmail.com
                    </a>
                  </div>
                  <div className="flex items-center">
                    <Mail className="w-4 h-4 mr-3 text-primary" />
                    <a href="mailto:mohilmandape33@gmail.com" className="text-sm hover:text-primary transition-colors">
                      mohilmandape33@gmail.com
                    </a>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-3">GitHub Repository</h4>
                <a 
                  href="https://github.com/MOHILMANDAPE15/Fashion-Recommendation-system" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-gray-200 to-gray-300 hover:from-gray-100 hover:to-gray-200 text-gray-800 rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg border border-gray-300 hover:border-gray-400 group"
                >
                  <Github className="w-4 h-4 mr-2 group-hover:rotate-12 transition-transform duration-300" />
                  <span className="text-sm font-medium">View on GitHub</span>
                  <svg className="w-3 h-3 ml-2 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
                <p className="text-xs text-muted-foreground mt-2">
                     Explore our open source code
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-border/50 bg-card/30">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center text-sm text-muted-foreground">
              <span className="flex items-center">
                Made with <Heart className="w-4 h-4 mx-1 text-red-500" /> by the AesthetIQ team
              </span>
            </div>
            
            <div className="text-sm text-muted-foreground">
              © 2025 AesthetIQ. All rights reserved.
            </div>
            
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>Built for educational and research purposes</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;