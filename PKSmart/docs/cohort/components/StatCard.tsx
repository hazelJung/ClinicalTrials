/*
 * Design Philosophy: Pharmaceutical Precision
 * - Stat card with top accent border
 * - Animated number counter
 * - Hover lift effect
 */

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  description?: string;
  trend?: "up" | "down" | "neutral";
  accentColor?: "teal" | "amber" | "navy" | "indigo";
  delay?: number;
}

const accentColors = {
  teal: "from-[oklch(0.65_0.15_180)] to-[oklch(0.75_0.12_180)]",
  amber: "from-[oklch(0.75_0.15_75)] to-[oklch(0.85_0.12_75)]",
  navy: "from-[oklch(0.25_0.05_250)] to-[oklch(0.35_0.05_250)]",
  indigo: "from-[oklch(0.50_0.20_260)] to-[oklch(0.60_0.18_260)]",
};

export default function StatCard({
  label,
  value,
  unit,
  description,
  accentColor = "teal",
  delay = 0,
}: StatCardProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [displayValue, setDisplayValue] = useState<string | number>(
    typeof value === "number" ? 0 : value
  );

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
      
      // Animate number if value is numeric
      if (typeof value === "number") {
        const duration = 1000;
        const steps = 30;
        const increment = value / steps;
        let current = 0;
        
        const counter = setInterval(() => {
          current += increment;
          if (current >= value) {
            setDisplayValue(value);
            clearInterval(counter);
          } else {
            setDisplayValue(Math.round(current * 10) / 10);
          }
        }, duration / steps);
        
        return () => clearInterval(counter);
      }
    }, delay);
    
    return () => clearTimeout(timer);
  }, [value, delay]);

  return (
    <div
      className={cn(
        "bg-card rounded-xl border border-border/50 shadow-sm relative overflow-hidden transition-all duration-300",
        "hover:shadow-lg hover:border-border hover:-translate-y-1",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {/* Top accent bar */}
      <div
        className={cn(
          "absolute top-0 left-0 right-0 h-1 bg-gradient-to-r",
          accentColors[accentColor]
        )}
      />
      
      <div className="p-6 pt-5">
        <p className="text-sm font-medium text-muted-foreground mb-2">
          {label}
        </p>
        <div className="flex items-baseline gap-1.5">
          <span className="text-3xl font-bold text-foreground font-mono tracking-tight">
            {displayValue}
          </span>
          {unit && (
            <span className="text-lg font-medium text-muted-foreground">
              {unit}
            </span>
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground mt-2">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}
