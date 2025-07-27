import { Github, Heart } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="py-12 border-t border-border/50 bg-card/30">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
            <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">FI</span>
            </div>
            <span className="text-lg font-semibold bg-gradient-primary bg-clip-text text-transparent">
              Face Insight Studio
            </span>
          </div>
          
          <div className="flex items-center space-x-6 text-sm text-muted-foreground">
            <span className="flex items-center">
              Made with <Heart className="w-4 h-4 mx-1 text-red-500" /> by the Face Insight team
            </span>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center hover:text-foreground transition-colors"
            >
              <Github className="w-4 h-4 mr-1" />
              Open Source
            </a>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-border/50 text-center text-sm text-muted-foreground">
          <p>© 2024 Face Insight Studio. All rights reserved. Built for educational and research purposes.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;