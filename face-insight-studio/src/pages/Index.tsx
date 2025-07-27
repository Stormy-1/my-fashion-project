import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import ContactSection from '@/components/ContactSection';
import Footer from '@/components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen bg-background relative">
      {/* Global background */}
      <div className="fixed inset-0 z-0">
        <img 
          src="/lovable-uploads/585f634e-0ff5-4109-a73e-1027534c119e.png" 
          alt="Fashion Items Background"
          className="w-full h-full object-cover opacity-50"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-background/70 via-background/60 to-background/75" />
      </div>
      
      <div className="relative z-10">
      <Navbar />
      <HeroSection />
      <ContactSection />
      <Footer />
      </div>
    </div>
  );
};

export default Index;
