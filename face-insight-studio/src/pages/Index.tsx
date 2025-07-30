import React from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, Sparkles, Zap, Heart } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import HeroSection from '@/components/HeroSection';
import ArchitectureSection from '@/components/ArchitectureSection';
import Footer from '@/components/Footer';
import FloatingFashionElements from '@/components/FloatingFashionElements';

import backgroundImage from '@/assets/image.png';

const Index = () => {
  return (
    <div className="min-h-screen bg-background relative">
      {/* Global background */}
      <div 
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: '100%',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          opacity: 0.3
        }}
      />
      <div className="absolute inset-0 z-0 bg-gradient-to-br from-background/70 via-background/60 to-background/75" />
      
      {/* Floating Fashion Elements */}
      <FloatingFashionElements />
      
      <div className="relative z-10">
        <Navbar />
        <HeroSection />
        <ArchitectureSection />
        <Footer />
      </div>
    </div>
  );
};

export default Index;
