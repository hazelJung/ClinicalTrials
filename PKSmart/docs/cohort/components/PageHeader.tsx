/*
 * Design Philosophy: Pharmaceutical Precision
 * - Clean page header with title and description
 * - Phase indicator badge
 */

import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  description?: string;
  phase?: string;
  phaseColor?: "teal" | "amber" | "indigo";
}

const phaseColors = {
  teal: "bg-[oklch(0.92_0.03_180)] text-[oklch(0.35_0.12_180)]",
  amber: "bg-[oklch(0.92_0.03_75)] text-[oklch(0.45_0.12_75)]",
  indigo: "bg-[oklch(0.92_0.03_260)] text-[oklch(0.40_0.15_260)]",
};

export default function PageHeader({
  title,
  description,
  phase,
  phaseColor = "teal",
}: PageHeaderProps) {
  return (
    <div className="mb-8">
      {phase && (
        <span
          className={cn(
            "inline-block px-3 py-1 text-xs font-semibold rounded-full mb-3",
            phaseColors[phaseColor]
          )}
        >
          {phase}
        </span>
      )}
      <h1 className="text-2xl lg:text-3xl font-bold text-foreground">
        {title}
      </h1>
      {description && (
        <p className="text-muted-foreground mt-2 max-w-2xl">
          {description}
        </p>
      )}
    </div>
  );
}
