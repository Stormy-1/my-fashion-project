import React, { useEffect, useState } from 'react';
import { Shirt, Watch, Glasses } from 'lucide-react';

interface FloatingElement {
  id: number;
  icon: string;
  x: number;
  y: number;
  size: number;
  duration: number;
  delay: number;
  direction: 'up' | 'down';
  color: string;
}

const FloatingFashionElements: React.FC = () => {
  const [elements, setElements] = useState<FloatingElement[]>([]);

  const fashionIcons = ['tshirt', 'watch', 'glasses'];
  const colors = ['#FF69B4', '#87CEEB'];

  const renderIcon = (iconType: string, color: string, size: number) => {
    const props = { size, color, strokeWidth: 2 };
    switch (iconType) {
      case 'tshirt': return <Shirt {...props} />;
      case 'watch': return <Watch {...props} />;
      case 'glasses': return <Glasses {...props} />;
      default: return null;
    }
  };

  useEffect(() => {
    const fixedPositions = [
      { x: 10, y: 25 },  // index 0 - Left Top
      { x: 90, y: 20 },  // index 1 - Right Top
      { x: 20, y: 60 },  // index 2 - Left Middle
      { x: 80, y: 55 },  // index 3 - Right Middle
      { x: 10, y: 90 },  // index 4 - Left Bottom
      { x: 90, y: 85 },  // index 5 - Right Bottom
    ];
  
    const getControlledColor = (index: number) => {
      const isLeft = fixedPositions[index].x < 50;
      const isMiddle = index === 2 || index === 3;
      if (isLeft) {
        return isMiddle ? '#87CEEB' : '#FF69B4'; // blue middle, others pink
      } else {
        return isMiddle ? '#FF69B4' : '#87CEEB'; // pink middle, others blue
      }
    };
  
    const newElements: FloatingElement[] = Array.from({ length: 6 }, (_, i) => ({
      id: i,
      icon: fashionIcons[i % fashionIcons.length],
      x: fixedPositions[i].x,
      y: fixedPositions[i].y,
      size: 28,
      duration: 20,
      delay: i * 1.2,
      direction: i % 2 === 0 ? 'up' : 'down',
      color: getControlledColor(i),
    }));
  
    setElements(newElements);
  }, []);
  
  

  return (
    <>
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        {elements.map((el) => (
          <div
            key={el.id}
            className="absolute pointer-events-none"
            style={{
              left: `${el.x}%`,
              top: `${el.y}%`,
              animation: `${el.direction === 'up' ? 'floatUp' : 'floatDown'}-${el.id} ${el.duration}s infinite ease-in-out ${el.delay}s`,
              transform: 'translate(-50%, -50%)',
              opacity: 0.7,
            }}
          >
            {renderIcon(el.icon, el.color, el.size)}
          </div>
        ))}
      </div>

      <style dangerouslySetInnerHTML={{
        __html: `
          ${elements.map(el => `
            @keyframes floatUp-${el.id} {
              0%, 100% {
                transform: translate(-50%, -50%) translateY(0px);
              }
              50% {
                transform: translate(-50%, -50%) translateY(-20px);
              }
            }

            @keyframes floatDown-${el.id} {
              0%, 100% {
                transform: translate(-50%, -50%) translateY(0px);
              }
              50% {
                transform: translate(-50%, -50%) translateY(20px);
              }
            }
          `).join('\n')}
        `
      }} />
    </>
  );
};

export default FloatingFashionElements;
