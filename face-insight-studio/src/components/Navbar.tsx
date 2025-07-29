import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Github, Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import logo from '@/assets/logo-1.png';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { href: '#home', label: 'Home' },
    { href: '#architecture', label: 'Architecture' },
    { href: '#contact', label: 'Contact Us' },
    { href: '/recommendations', label: 'Previous Recommendations' },
  ];

  const handleNavigation = (href: string) => {
    setIsMobileMenuOpen(false);
    
    // Check if it's a route navigation (starts with /)
    if (href.startsWith('/')) {
      navigate(href);
      return;
    }
    
    // Handle hash navigation for sections
    // If we're already on the home page, just scroll to section
    if (location.pathname === '/') {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    } else {
      // Navigate to home page with hash for the section
      navigate(`/${href}`);
      // Small delay to ensure page loads before scrolling
      setTimeout(() => {
        const element = document.querySelector(href);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    }
  };

  return (
    <nav className={cn(
      "fixed top-0 w-full z-50 transition-all duration-300",
      isScrolled ? "bg-background/90 backdrop-blur-md border-b border-border/50" : "bg-transparent"
    )}>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <img 
              src={logo} 
              alt="AesthetIQ Logo" 
              className="h-8 w-auto rounded-lg"
            />
           <h1 className="text-xl md:text-xl lg:text-2xl font-bold  leading-tight py-10">
            <span className="bg-gradient-to-r from-purple-800 to-blue-300 bg-clip-text text-transparent">Aesthet</span>
            <span className="bg-gradient-to-r from-purple-800 to-pink-300 bg-clip-text text-transparent">IQ</span>
          </h1>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <button
                key={item.href}
                onClick={() => handleNavigation(item.href)}
                className="text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium"
              >
                {item.label}
              </button>
            ))}
            <Button
              variant="outline"
              size="sm"
              className="border-primary/20 hover:border-primary hover:bg-primary/10"
              onClick={() => window.open('https://github.com', '_blank')}
            >
              <Github className="w-4 h-4 mr-2" />
              GitHub
            </Button>
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          >
            {isMobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 bg-card/95 backdrop-blur-md border-b border-border/50">
            <div className="flex flex-col space-y-4 p-4">
              {navItems.map((item) => (
                <button
                  key={item.href}
                  onClick={() => handleNavigation(item.href)}
                  className="text-left text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium py-2"
                >
                  {item.label}
                </button>
              ))}
              <Button
                variant="outline"
                size="sm"
                className="w-fit border-primary/20 hover:border-primary hover:bg-primary/10"
                onClick={() => {
                  window.open('https://github.com/MOHILMANDAPE15/Fashion-Recommendation-system', '_blank');
                  setIsMobileMenuOpen(false);
                }}
              >
                <Github className="w-4 h-4 mr-2" />
                GitHub
              </Button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;