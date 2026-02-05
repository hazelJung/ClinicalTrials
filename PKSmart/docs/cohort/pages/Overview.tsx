/*
 * Design Philosophy: Pharmaceutical Precision
 * - Overview page showing summary of each single cohort
 * - Quick navigation to each phase's cohort data
 */

import { Link } from "wouter";
import PageHeader from "@/components/PageHeader";
import StatCard from "@/components/StatCard";
import { 
  Beaker, 
  FlaskConical, 
  Award, 
  ArrowRight,
  Dna
} from "lucide-react";
import { cn } from "@/lib/utils";

const phaseCards = [
  {
    phase: "Phase 1",
    title: "임상 1상",
    cohort: "400mg 코호트 (N=6)",
    subtitle: "안전성 및 약동학(PK) 프로파일",
    description: "개별 환자 PK 프로파일, 이상반응(AE) 발생 현황, CYP450 유전자형별 약동학 비교",
    icon: Beaker,
    href: "/phase1",
    color: "teal",
    stats: { patients: "6명", dlt: "1/6" },
  },
  {
    phase: "Phase 2",
    title: "임상 2상",
    cohort: "300mg 코호트 (N=12)",
    subtitle: "예비 유효성 및 종양 반응",
    description: "개별 환자 종양 크기 변화, 치료 지속 기간, 바이오마커별 반응 분석",
    icon: FlaskConical,
    href: "/phase2",
    color: "amber",
    stats: { patients: "12명", orr: "41.7%" },
  },
  {
    phase: "Phase 3",
    title: "임상 3상",
    cohort: "시험군 코호트 (N=15)",
    subtitle: "생존 분석 및 바이오마커 하위 그룹",
    description: "개별 환자 생존 데이터, 치료 타임라인, 바이오마커 양성/음성 비교",
    icon: Award,
    href: "/phase3",
    color: "indigo",
    stats: { patients: "15명", medianOS: "24.8mo" },
  },
];

const colorClasses = {
  teal: {
    bg: "bg-[oklch(0.92_0.03_180)]",
    text: "text-[oklch(0.35_0.12_180)]",
    border: "border-[oklch(0.65_0.15_180)]",
    iconBg: "bg-[oklch(0.65_0.15_180)]",
  },
  amber: {
    bg: "bg-[oklch(0.92_0.03_75)]",
    text: "text-[oklch(0.45_0.12_75)]",
    border: "border-[oklch(0.75_0.15_75)]",
    iconBg: "bg-[oklch(0.75_0.15_75)]",
  },
  indigo: {
    bg: "bg-[oklch(0.92_0.03_260)]",
    text: "text-[oklch(0.40_0.15_260)]",
    border: "border-[oklch(0.50_0.20_260)]",
    iconBg: "bg-[oklch(0.50_0.20_260)]",
  },
};

export default function Overview() {
  return (
    <div className="space-y-8">
      <PageHeader
        title="임상시험 코호트 결과 대시보드"
        description="각 임상 단계별 단일 코호트의 개별 환자 데이터를 시각화합니다."
      />

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="임상시험 단계"
          value={3}
          description="Phase 1, 2, 3"
          accentColor="navy"
          delay={0}
        />
        <StatCard
          label="총 환자 수"
          value={33}
          unit="명"
          description="3개 코호트 합계"
          accentColor="teal"
          delay={100}
        />
        <StatCard
          label="데이터 유형"
          value="개별"
          description="환자별 상세 데이터"
          accentColor="amber"
          delay={200}
        />
        <StatCard
          label="약물유전체 분석"
          value="포함"
          description="CYP450, KRAS, EGFR"
          accentColor="indigo"
          delay={300}
        />
      </div>

      {/* Phase Cards */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">
          코호트별 결과 보기
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {phaseCards.map((card, index) => {
            const Icon = card.icon;
            const colors = colorClasses[card.color as keyof typeof colorClasses];
            
            return (
              <Link key={card.href} href={card.href}>
                <div
                  className={cn(
                    "group bg-card rounded-xl border border-border/50 shadow-sm overflow-hidden",
                    "transition-all duration-300 hover:shadow-lg hover:border-border hover:-translate-y-1",
                    "cursor-pointer h-full"
                  )}
                  style={{ 
                    animationDelay: `${index * 100 + 400}ms`,
                    animation: "fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards",
                    opacity: 0,
                  }}
                >
                  {/* Header */}
                  <div className={cn("p-6", colors.bg)}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center", colors.iconBg)}>
                          <Icon className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <span className={cn("text-xs font-semibold", colors.text)}>
                            {card.phase}
                          </span>
                          <h3 className="text-lg font-bold text-foreground">
                            {card.title}
                          </h3>
                        </div>
                      </div>
                      <ArrowRight className={cn(
                        "w-5 h-5 transition-transform duration-300",
                        colors.text,
                        "group-hover:translate-x-1"
                      )} />
                    </div>
                    <p className={cn("text-sm font-medium mt-2", colors.text)}>
                      {card.cohort}
                    </p>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    <p className="text-sm font-medium text-foreground mb-2">
                      {card.subtitle}
                    </p>
                    <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                      {card.description}
                    </p>
                    
                    {/* Stats */}
                    <div className="flex gap-4">
                      {Object.entries(card.stats).map(([key, value]) => (
                        <div key={key} className="text-center">
                          <p className="text-lg font-bold font-mono text-foreground">
                            {value}
                          </p>
                          <p className="text-xs text-muted-foreground uppercase">
                            {key}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* System Info */}
      <div className="bg-card rounded-xl border border-border/50 shadow-sm p-6">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-lg bg-[oklch(0.92_0.03_180)] flex items-center justify-center flex-shrink-0">
            <Dna className="w-6 h-6 text-[oklch(0.35_0.12_180)]" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              대시보드 안내
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              본 대시보드는 각 임상 단계별 <strong>단일 코호트</strong>의 개별 환자 데이터를 시각화합니다. 
              1상은 400mg 용량 코호트, 2상은 300mg RP2D 코호트, 3상은 시험군 코호트의 데이터를 포함합니다.
              각 페이지에서 개별 환자의 약동학, 이상반응, 종양 반응, 생존 데이터를 확인할 수 있습니다.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
