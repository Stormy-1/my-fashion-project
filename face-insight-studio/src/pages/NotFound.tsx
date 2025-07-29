import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import Navbar from '@/components/Navbar';

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <>
      <Navbar />
      <div className="min-h-screen flex items-center justify-center bg-background relative">
      {/* Global background */}
      <div className="fixed inset-0 z-0">
        <img 
          src="/lovable-uploads/585f634e-0ff5-4109-a73e-1027534c119e.png" 
          alt="Fashion Items Background"
          className="w-full h-full object-cover opacity-50"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-background/70 via-background/60 to-background/75" />
      </div>
      
      <div className="text-center relative z-10">
        <h1 className="text-4xl font-bold mb-4 text-foreground">404</h1>
        <p className="text-xl text-muted-foreground mb-4">Oops! Page not found</p>
        <a href="/" className="text-primary hover:text-primary/80 underline">
          Return to Home
        </a>
      </div>
    </div>
    </>
  );
};

export default NotFound;
