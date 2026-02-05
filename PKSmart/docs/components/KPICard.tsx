import { motion } from "framer-motion";
import { LucideIcon, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card } from "@/components/ui/card";

interface KPICardProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  color?: "cyan" | "lime" | "purple" | "amber";
  delay?: number;
}

const colorClasses = {
  cyan: {
    icon: "text-cyan-400",
    glow: "glow-cyan",
    bg: "bg-cyan-500/10",
    border: "border-cyan-500/20",
  },
  lime: {
    icon: "text-lime-400",
    glow: "glow-lime",
    bg: "bg-lime-500/10",
    border: "border-lime-500/20",
  },
  purple: {
    icon: "text-purple-400",
    glow: "glow-purple",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
  },
  amber: {
    icon: "text-amber-400",
    glow: "",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
  },
};

export default function KPICard({
  title,
  value,
  unit,
  change,
  changeLabel,
  icon: Icon,
  color = "cyan",
  delay = 0,
}: KPICardProps) {
  const colors = colorClasses[color];
  
  const TrendIcon = change === undefined 
    ? Minus 
    : change > 0 
      ? TrendingUp 
      : change < 0 
        ? TrendingDown 
        : Minus;

  const trendColor = change === undefined
    ? "text-muted-foreground"
    : change > 0
      ? "text-lime-400"
      : change < 0
        ? "text-red-400"
        : "text-muted-foreground";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <Card className={`
        relative overflow-hidden p-5 card-hover
        bg-card/50 backdrop-blur-sm border ${colors.border}
        hover:${colors.glow}
      `}>
        {/* Background Icon */}
        <div className="absolute -right-4 -top-4 opacity-5">
          <Icon className="h-24 w-24" />
        </div>

        {/* Content */}
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-lg ${colors.bg}`}>
              <Icon className={`h-5 w-5 ${colors.icon}`} />
            </div>
            <span className="text-sm text-muted-foreground font-medium">{title}</span>
          </div>

          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold data-value tracking-tight">
              {value}
            </span>
            {unit && (
              <span className="text-sm text-muted-foreground">{unit}</span>
            )}
          </div>

          {change !== undefined && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trendColor}`}>
              <TrendIcon className="h-4 w-4" />
              <span>{change > 0 ? "+" : ""}{change}%</span>
              {changeLabel && (
                <span className="text-muted-foreground ml-1">{changeLabel}</span>
              )}
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  );
}
