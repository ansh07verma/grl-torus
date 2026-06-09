import React, { useEffect, useState, useRef } from 'react';

interface TorusVisualizerProps {
  gridSize: int;
  isRunning: boolean;
  routerType: string;
}

interface Packet {
  id: number;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  progress: number; // 0 to 1 between current and next node
  color: string;
}

const TorusVisualizer: React.FC<TorusVisualizerProps> = ({ gridSize, isRunning, routerType }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [packets, setPackets] = useState<Packet[]>([]);
  
  // Simulation loop
  useEffect(() => {
    if (!isRunning) return;
    
    let animationFrameId: number;
    let lastTime = performance.now();
    
    const animate = (time: number) => {
      const dt = (time - lastTime) / 1000; // seconds
      lastTime = time;
      
      setPackets(prev => {
        // Update existing packets
        let updated = prev.map(p => {
          let newProgress = p.progress + dt * 2; // speed
          if (newProgress >= 1) {
            // Reached next node
            if (p.x === p.targetX && p.y === p.targetY) {
              return null; // Delivered
            }
            
            // Simple XY routing logic for visualization purposes
            let nx = p.x;
            let ny = p.y;
            
            if (routerType === 'xy' || routerType === 'gnn' || routerType === 'grl') {
                if (nx !== p.targetX) {
                    nx = (nx + 1) % gridSize; // move right
                } else if (ny !== p.targetY) {
                    ny = (ny + 1) % gridSize; // move down
                }
            } else {
                // Randomish for valiant/odd_even just for visual flair
                if (Math.random() > 0.5 && nx !== p.targetX) nx = (nx + 1) % gridSize;
                else if (ny !== p.targetY) ny = (ny + 1) % gridSize;
            }
            
            return { ...p, x: nx, y: ny, progress: 0 };
          }
          return { ...p, progress: newProgress };
        }).filter(Boolean) as Packet[];
        
        // Spawn new packets occasionally
        if (Math.random() < 0.1) {
            const srcX = Math.floor(Math.random() * gridSize);
            const srcY = Math.floor(Math.random() * gridSize);
            let dstX = Math.floor(Math.random() * gridSize);
            let dstY = Math.floor(Math.random() * gridSize);
            while (dstX === srcX && dstY === srcY) {
                dstX = Math.floor(Math.random() * gridSize);
                dstY = Math.floor(Math.random() * gridSize);
            }
            
            const color = `hsl(${Math.random() * 360}, 80%, 60%)`;
            
            updated.push({
                id: Math.random(),
                x: srcX,
                y: srcY,
                targetX: dstX,
                targetY: dstY,
                progress: 0,
                color
            });
        }
        
        return updated;
      });
      
      animationFrameId = requestAnimationFrame(animate);
    };
    
    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isRunning, gridSize, routerType]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Handle resizing
    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
      }
    };
    resize();
    window.addEventListener('resize', resize);
    
    // Draw logic
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const padding = 60;
      const usableWidth = canvas.width - padding * 2;
      const usableHeight = canvas.height - padding * 2;
      
      const nodeSpacingX = usableWidth / Math.max(1, gridSize - 1);
      const nodeSpacingY = usableHeight / Math.max(1, gridSize - 1);
      
      // Draw links (wraparound torus)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 2;
      
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const px = padding + x * nodeSpacingX;
          const py = padding + y * nodeSpacingY;
          
          // Right link
          ctx.beginPath();
          ctx.moveTo(px, py);
          if (x === gridSize - 1) {
            // Wraparound
            ctx.lineTo(canvas.width, py);
            ctx.moveTo(0, py);
            ctx.lineTo(padding, py);
          } else {
            ctx.lineTo(padding + (x + 1) * nodeSpacingX, py);
          }
          ctx.stroke();
          
          // Down link
          ctx.beginPath();
          ctx.moveTo(px, py);
          if (y === gridSize - 1) {
            // Wraparound
            ctx.lineTo(px, canvas.height);
            ctx.moveTo(px, 0);
            ctx.lineTo(px, padding);
          } else {
            ctx.lineTo(px, padding + (y + 1) * nodeSpacingY);
          }
          ctx.stroke();
        }
      }
      
      // Draw packets
      packets.forEach(p => {
        const sx = padding + p.x * nodeSpacingX;
        const sy = padding + p.y * nodeSpacingY;
        
        let targetX_node = p.x;
        let targetY_node = p.y;
        
        if (routerType === 'xy' || routerType === 'gnn' || routerType === 'grl') {
            if (p.x !== p.targetX) targetX_node = (p.x + 1) % gridSize;
            else if (p.y !== p.targetY) targetY_node = (p.y + 1) % gridSize;
        } else {
             if (p.x !== p.targetX) targetX_node = (p.x + 1) % gridSize;
             else if (p.y !== p.targetY) targetY_node = (p.y + 1) % gridSize;
        }

        const tx = padding + targetX_node * nodeSpacingX;
        const ty = padding + targetY_node * nodeSpacingY;
        
        let currentX = sx;
        let currentY = sy;
        
        // Handle wraparound interpolation
        if (Math.abs(targetX_node - p.x) > 1) {
           // wraparound horizontally
           currentX = sx + (canvas.width * p.progress); // highly simplified
        } else if (Math.abs(targetY_node - p.y) > 1) {
           currentY = sy + (canvas.height * p.progress);
        } else {
           currentX = sx + (tx - sx) * p.progress;
           currentY = sy + (ty - sy) * p.progress;
        }
        
        // Draw glow
        ctx.beginPath();
        ctx.arc(currentX, currentY, 8, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      });
      
      // Draw nodes
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const px = padding + x * nodeSpacingX;
          const py = padding + y * nodeSpacingY;
          
          ctx.beginPath();
          ctx.arc(px, py, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#1e293b';
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.stroke();
        }
      }
      
      requestAnimationFrame(draw);
    };
    
    const drawFrameId = requestAnimationFrame(draw);
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(drawFrameId);
    };
  }, [gridSize, packets, routerType]);

  return (
    <canvas 
      ref={canvasRef} 
      style={{ 
        width: '100%', 
        height: '100%', 
        minHeight: '500px',
        display: 'block'
      }} 
    />
  );
};

export default TorusVisualizer;
