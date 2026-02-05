/*
 * Design Philosophy: Pharmaceutical Precision
 * - Clean chart container with subtle shadow
 * - Title and description area
 * - Hover effect for interactivity
 */

import { cn } from "@/lib/utils";
import { ReactNode, useEffect, useState } from "react";

interface ChartCardProps {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
  delay?: number;
}

export default function ChartCard({
  title,
  description,
  children,
  className,
  delay = 0,
}: ChartCardProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <div
      className={cn(
        "bg-card rounded-xl border border-border/50 shadow-sm transition-all duration-500",
        "hover:shadow-lg hover:border-border",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4",
        className
      )}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className="p-6 border-b border-border/50">
        <h3 className="text-lg font-semibold text-foreground">
          {title}
        </h3>
        {description && (
          <p className="text-sm text-muted-foreground mt-1">
            {description}
          </p>
        )}
      </div>
      <div className="p-6">
        {children}
      </div>
    </div>
  );
}
