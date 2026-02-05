/**
 * Advanced Cohort Settings Component
 * 한미약품 FDA 타겟 임상 데이터 기반 디지털 트윈 코호트 설정
 * 
 * Design: Pharma Command Center - Dark Mode Mission Control
 * Features: Phase 연동 자동 설정, 적응증 기반 프리셋, 동적 PK 영향 계산
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Dna,
  Scale,
  Pill,
  Heart,
  Activity,
  Users,
  Syringe,
  ChevronDown,
  ChevronUp,
  Info,
  Zap,
  AlertTriangle,
  Check,
  Settings2,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { toast } from "sonner";

// Types
interface CohortSettings {
  phase: "phase1" | "phase2" | "phase3";
  population: string;
  sampleSize: number;
  ageRange: [number, number];
  genderRatio: number;
  // Advanced settings
  geneticPolymorphism: {
    enzyme: string;
    distribution: { PM: number; IM: number; EM: number; UM: number };
  };
  bodyMetrics: {
    bmiRange: [number, number];
    distribution: string;
  };
  organFunction: {
    renalDistribution: { normal: number; mild: number; moderate: number; severe: number };
    hepaticGrade: string;
  };
  concomitantMeds: string[];
  diseaseState: {
    indication: string;
    biomarkers: Record<string, { mean: number; sd: number }>;
  };
  dosingRegimen: {
    type: string;
    doses: number[];
    frequency: string;
  };
}

interface PKImpact {
  parameter: string;
  change: string;
  factor: number;
  severity: "low" | "medium" | "high";
}

// Phase-based default settings
const PHASE_DEFAULTS: Record<string, Partial<CohortSettings>> = {
  phase1: {
    ageRange: [19, 55],
    bodyMetrics: { bmiRange: [18.5, 30], distribution: "normal" },
    organFunction: {
      renalDistribution: { normal: 100, mild: 0, moderate: 0, severe: 0 },
      hepaticGrade: "normal",
    },
    dosingRegimen: { type: "escalation", doses: [10, 30, 100, 300], frequency: "QD" },
  },
  phase2: {
    ageRange: [18, 75],
    bodyMetrics: { bmiRange: [25, 45], distribution: "lognormal" },
    organFunction: {
      renalDistribution: { normal: 60, mild: 25, moderate: 15, severe: 0 },
      hepaticGrade: "mild_allowed",
    },
    dosingRegimen: { type: "dose_finding", doses: [100, 150, 200], frequency: "QD" },
  },
  phase3: {
    ageRange: [18, 80],
    bodyMetrics: { bmiRange: [25, 50], distribution: "lognormal" },
    organFunction: {
      renalDistribution: { normal: 50, mild: 30, moderate: 15, severe: 5 },
      hepaticGrade: "moderate_allowed",
    },
    dosingRegimen: { type: "fixed", doses: [150], frequency: "QD" },
  },
};

// Genetic presets
const GENETIC_PRESETS = [
  { id: "cyp2d6_pm", name: "CYP2D6 PM Cohort", enzyme: "CYP2D6", distribution: { PM: 100, IM: 0, EM: 0, UM: 0 } },
  { id: "cyp2c19_asian", name: "CYP2C19 Asian", enzyme: "CYP2C19", distribution: { PM: 20, IM: 35, EM: 40, UM: 5 } },
  { id: "cyp2d6_eur", name: "CYP2D6 European", enzyme: "CYP2D6", distribution: { PM: 7, IM: 25, EM: 63, UM: 5 } },
  { id: "ugt1a1_28", name: "UGT1A1*28 Cohort", enzyme: "UGT1A1", distribution: { PM: 10, IM: 40, EM: 50, UM: 0 } },
];

// DDI Presets by indication
const DDI_PRESETS: Record<string, Array<{ name: string; cyp: string | null; factor: number }>> = {
  diabetes: [
    { name: "Metformin", cyp: null, factor: 1.0 },
    { name: "Glimepiride", cyp: "CYP2C9", factor: 1.2 },
    { name: "Empagliflozin", cyp: null, factor: 1.0 },
    { name: "Atorvastatin", cyp: "CYP3A4", factor: 1.4 },
    { name: "Lisinopril", cyp: null, factor: 1.0 },
  ],
  obesity: [
    { name: "Metformin", cyp: null, factor: 1.0 },
    { name: "Orlistat", cyp: null, factor: 1.0 },
    { name: "Phentermine", cyp: "CYP3A4", factor: 1.3 },
  ],
  oncology: [
    { name: "Ondansetron", cyp: "CYP3A4", factor: 1.2 },
    { name: "Dexamethasone", cyp: "CYP3A4_inducer", factor: 0.5 },
    { name: "Ketoconazole", cyp: "CYP3A4_inhibitor", factor: 5.0 },
    { name: "Aprepitant", cyp: "CYP3A4_inhibitor", factor: 2.0 },
  ],
  nafld: [
    { name: "Metformin", cyp: null, factor: 1.0 },
    { name: "Vitamin E", cyp: null, factor: 1.0 },
    { name: "Atorvastatin", cyp: "CYP3A4", factor: 1.4 },
  ],
};

// Section component for collapsible sections
function SettingsSection({
  title,
  icon: Icon,
  children,
  defaultOpen = false,
  badge,
  warning,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string;
  warning?: string;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <motion.div
      className="border border-border/50 rounded-xl overflow-hidden bg-card/30 backdrop-blur-sm"
      initial={false}
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-secondary/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Icon className="h-4 w-4 text-primary" />
          </div>
          <span className="font-medium">{title}</span>
          {badge && (
            <Badge variant="secondary" className="text-xs">
              {badge}
            </Badge>
          )}
          {warning && (
            <Tooltip>
              <TooltipTrigger>
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">{warning}</p>
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-2 space-y-4 border-t border-border/30">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// PK Impact indicator
function PKImpactIndicator({ impacts }: { impacts: PKImpact[] }) {
  if (impacts.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-4 p-3 rounded-lg bg-primary/5 border border-primary/20"
    >
      <div className="flex items-center gap-2 mb-2">
        <Zap className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium text-primary">PK 영향 예측</span>
      </div>
      <div className="space-y-1">
        {impacts.map((impact, idx) => (
          <div key={idx} className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{impact.parameter}</span>
            <span
              className={`font-mono ${
                impact.severity === "high"
                  ? "text-red-400"
                  : impact.severity === "medium"
                  ? "text-yellow-400"
                  : "text-green-400"
              }`}
            >
              {impact.change} ({impact.factor.toFixed(1)}×)
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export default function CohortSettings() {
  const [settings, setSettings] = useState<CohortSettings>({
    phase: "phase1",
    population: "ASN",
    sampleSize: 1000,
    ageRange: [18, 80],
    genderRatio: 50,
    geneticPolymorphism: {
      enzyme: "CYP2D6",
      distribution: { PM: 5, IM: 25, EM: 65, UM: 5 },
    },
    bodyMetrics: {
      bmiRange: [18.5, 30],
      distribution: "normal",
    },
    organFunction: {
      renalDistribution: { normal: 100, mild: 0, moderate: 0, severe: 0 },
      hepaticGrade: "normal",
    },
    concomitantMeds: [],
    diseaseState: {
      indication: "diabetes",
      biomarkers: {
        HbA1c: { mean: 8.2, sd: 0.8 },
      },
    },
    dosingRegimen: {
      type: "escalation",
      doses: [10, 30, 100, 300],
      frequency: "QD",
    },
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [pkImpacts, setPkImpacts] = useState<PKImpact[]>([]);

  // Phase change handler - auto-update settings
  useEffect(() => {
    const defaults = PHASE_DEFAULTS[settings.phase];
    if (defaults) {
      setSettings((prev) => ({
        ...prev,
        ...defaults,
        geneticPolymorphism: prev.geneticPolymorphism,
        concomitantMeds: prev.concomitantMeds,
        diseaseState: prev.diseaseState,
      }));
    }
  }, [settings.phase]);

  // Calculate PK impacts based on settings
  useEffect(() => {
    const impacts: PKImpact[] = [];

    // Genetic polymorphism impact
    if (settings.geneticPolymorphism.distribution.PM > 20) {
      impacts.push({
        parameter: "AUC",
        change: "↑ 증가",
        factor: 3.5,
        severity: "high",
      });
      impacts.push({
        parameter: "CL",
        change: "↓ 감소",
        factor: 0.3,
        severity: "high",
      });
    }

    // Renal function impact
    const renalImpaired =
      settings.organFunction.renalDistribution.moderate +
      settings.organFunction.renalDistribution.severe;
    if (renalImpaired > 10) {
      impacts.push({
        parameter: "T½",
        change: "↑ 연장",
        factor: 2.0,
        severity: "medium",
      });
    }

    // DDI impact
    const strongInhibitors = settings.concomitantMeds.filter((med) => {
      const preset = Object.values(DDI_PRESETS)
        .flat()
        .find((p) => p.name === med);
      return preset?.cyp?.includes("inhibitor");
    });
    if (strongInhibitors.length > 0) {
      impacts.push({
        parameter: "AUC",
        change: "↑ DDI",
        factor: 5.0,
        severity: "high",
      });
    }

    setPkImpacts(impacts);
  }, [settings]);

  // Apply genetic preset
  const applyGeneticPreset = (presetId: string) => {
    const preset = GENETIC_PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setSettings((prev) => ({
        ...prev,
        geneticPolymorphism: {
          enzyme: preset.enzyme,
          distribution: preset.distribution,
        },
      }));
      toast.success(`${preset.name} 프리셋이 적용되었습니다.`);
    }
  };

  // Toggle concomitant medication
  const toggleMed = (medName: string) => {
    setSettings((prev) => ({
      ...prev,
      concomitantMeds: prev.concomitantMeds.includes(medName)
        ? prev.concomitantMeds.filter((m) => m !== medName)
        : [...prev.concomitantMeds, medName],
    }));
  };

  // Run simulation
  const runSimulation = () => {
    toast.success("시뮬레이션이 시작되었습니다.", {
      description: `${settings.sampleSize}명의 가상 코호트 생성 중...`,
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-primary/20 to-cyan-500/20 border border-primary/30">
            <Settings2 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-bold">Cohort Settings</h2>
            <p className="text-sm text-muted-foreground">
              Configure your virtual clinical trial
            </p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="gap-2"
        >
          <Sparkles className="h-4 w-4" />
          {showAdvanced ? "Basic" : "Advanced"}
        </Button>
      </div>

      {/* Clinical Phase Selection */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-primary uppercase tracking-wider">
          Clinical Phase
        </label>
        <div className="grid grid-cols-3 gap-2">
          {(["phase1", "phase2", "phase3"] as const).map((phase) => (
            <Button
              key={phase}
              variant={settings.phase === phase ? "default" : "outline"}
              className={`relative ${
                settings.phase === phase
                  ? "bg-primary text-primary-foreground"
                  : "bg-card/50"
              }`}
              onClick={() => setSettings((prev) => ({ ...prev, phase }))}
            >
              {phase === "phase1" && "Phase 1"}
              {phase === "phase2" && "Phase 2"}
              {phase === "phase3" && "Phase 3"}
              {settings.phase === phase && (
                <motion.div
                  layoutId="phaseIndicator"
                  className="absolute inset-0 rounded-md border-2 border-primary"
                  transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
              )}
            </Button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {settings.phase === "phase1" && "건강인 대상 안전성 및 PK 평가"}
          {settings.phase === "phase2" && "환자 대상 개념 증명 및 용량 탐색"}
          {settings.phase === "phase3" && "환자 대상 확증 임상시험"}
        </p>
      </div>

      {/* Demographics */}
      <SettingsSection title="Demographics" icon={Users} defaultOpen={true}>
        <div className="space-y-4">
          {/* Population */}
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground">Race / Population</label>
            <Select
              value={settings.population}
              onValueChange={(v) => setSettings((prev) => ({ ...prev, population: v }))}
            >
              <SelectTrigger className="bg-background/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ASN">Asian (ASN)</SelectItem>
                <SelectItem value="EUR">European (EUR)</SelectItem>
                <SelectItem value="AFR">African (AFR)</SelectItem>
                <SelectItem value="MIX">Mixed Population</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Sample Size */}
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground">Sample Size (Subjects)</label>
            <div className="flex items-center gap-4">
              <Slider
                value={[settings.sampleSize]}
                onValueChange={([v]) => setSettings((prev) => ({ ...prev, sampleSize: v }))}
                min={10}
                max={5000}
                step={10}
                className="flex-1"
              />
              <span className="w-16 text-right font-mono text-primary">
                {settings.sampleSize}
              </span>
            </div>
          </div>

          {/* Age Range */}
          <div className="space-y-2">
            <label className="text-sm text-muted-foreground">
              Age Range:{" "}
              <span className="text-primary">
                {settings.ageRange[0]} - {settings.ageRange[1]}
              </span>
            </label>
            <Slider
              value={settings.ageRange}
              onValueChange={(v) =>
                setSettings((prev) => ({ ...prev, ageRange: v as [number, number] }))
              }
              min={18}
              max={100}
              step={1}
            />
          </div>

          {/* Gender Ratio */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-blue-400">Male</span>
              <span className="text-pink-400">Female: {settings.genderRatio}%</span>
            </div>
            <Slider
              value={[settings.genderRatio]}
              onValueChange={([v]) => setSettings((prev) => ({ ...prev, genderRatio: v }))}
              min={0}
              max={100}
              step={1}
            />
          </div>
        </div>
      </SettingsSection>

      {/* Advanced Settings */}
      <AnimatePresence>
        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-4"
          >
            {/* Genetic Polymorphism */}
            <SettingsSection
              title="Genetic Polymorphism"
              icon={Dna}
              badge="Pharmacogenomics"
              warning={
                settings.geneticPolymorphism.distribution.PM > 20
                  ? "PM 비율이 높습니다. AUC 증가에 주의하세요."
                  : undefined
              }
            >
              <div className="space-y-4">
                {/* Enzyme Selection */}
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Target Enzyme</label>
                  <Select
                    value={settings.geneticPolymorphism.enzyme}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        geneticPolymorphism: { ...prev.geneticPolymorphism, enzyme: v },
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="CYP2D6">CYP2D6</SelectItem>
                      <SelectItem value="CYP2C19">CYP2C19</SelectItem>
                      <SelectItem value="CYP3A4">CYP3A4</SelectItem>
                      <SelectItem value="CYP2C9">CYP2C9</SelectItem>
                      <SelectItem value="UGT1A1">UGT1A1</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Phenotype Distribution */}
                <div className="space-y-3">
                  <label className="text-sm text-muted-foreground">Phenotype Distribution</label>
                  {(["PM", "IM", "EM", "UM"] as const).map((phenotype) => (
                    <div key={phenotype} className="flex items-center gap-3">
                      <span className="w-8 text-xs font-mono text-muted-foreground">
                        {phenotype}
                      </span>
                      <Slider
                        value={[settings.geneticPolymorphism.distribution[phenotype]]}
                        onValueChange={([v]) =>
                          setSettings((prev) => ({
                            ...prev,
                            geneticPolymorphism: {
                              ...prev.geneticPolymorphism,
                              distribution: {
                                ...prev.geneticPolymorphism.distribution,
                                [phenotype]: v,
                              },
                            },
                          }))
                        }
                        min={0}
                        max={100}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-10 text-right text-sm font-mono text-primary">
                        {settings.geneticPolymorphism.distribution[phenotype]}%
                      </span>
                    </div>
                  ))}
                </div>

                {/* Presets */}
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Quick Presets</label>
                  <div className="flex flex-wrap gap-2">
                    {GENETIC_PRESETS.map((preset) => (
                      <Button
                        key={preset.id}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={() => applyGeneticPreset(preset.id)}
                      >
                        {preset.name}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            </SettingsSection>

            {/* Body Metrics */}
            <SettingsSection title="Body Metrics (BMI/BSA)" icon={Scale}>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">
                    BMI Range (kg/m²):{" "}
                    <span className="text-primary">
                      {settings.bodyMetrics.bmiRange[0]} - {settings.bodyMetrics.bmiRange[1]}
                    </span>
                  </label>
                  <Slider
                    value={settings.bodyMetrics.bmiRange}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        bodyMetrics: { ...prev.bodyMetrics, bmiRange: v as [number, number] },
                      }))
                    }
                    min={15}
                    max={60}
                    step={0.5}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Weight Distribution</label>
                  <Select
                    value={settings.bodyMetrics.distribution}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        bodyMetrics: { ...prev.bodyMetrics, distribution: v },
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="normal">Normal Distribution</SelectItem>
                      <SelectItem value="lognormal">Log-normal (Realistic)</SelectItem>
                      <SelectItem value="uniform">Uniform</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </SettingsSection>

            {/* Organ Function */}
            <SettingsSection title="Organ Function" icon={Heart}>
              <div className="space-y-4">
                {/* Renal Function */}
                <div className="space-y-3">
                  <label className="text-sm text-muted-foreground">
                    Renal Function (eGFR Distribution)
                  </label>
                  {(["normal", "mild", "moderate", "severe"] as const).map((level) => (
                    <div key={level} className="flex items-center gap-3">
                      <span className="w-20 text-xs capitalize text-muted-foreground">
                        {level}
                      </span>
                      <Slider
                        value={[settings.organFunction.renalDistribution[level]]}
                        onValueChange={([v]) =>
                          setSettings((prev) => ({
                            ...prev,
                            organFunction: {
                              ...prev.organFunction,
                              renalDistribution: {
                                ...prev.organFunction.renalDistribution,
                                [level]: v,
                              },
                            },
                          }))
                        }
                        min={0}
                        max={100}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-10 text-right text-sm font-mono text-primary">
                        {settings.organFunction.renalDistribution[level]}%
                      </span>
                    </div>
                  ))}
                </div>

                {/* Hepatic Function */}
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">
                    Hepatic Function (Child-Pugh)
                  </label>
                  <Select
                    value={settings.organFunction.hepaticGrade}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        organFunction: { ...prev.organFunction, hepaticGrade: v },
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="normal">Normal (A, 5-6점)</SelectItem>
                      <SelectItem value="mild_allowed">Mild Allowed (A-B)</SelectItem>
                      <SelectItem value="moderate_allowed">Moderate Allowed (B)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </SettingsSection>

            {/* Concomitant Medications (DDI) */}
            <SettingsSection
              title="Concomitant Medications (DDI)"
              icon={Pill}
              badge={settings.concomitantMeds.length > 0 ? `${settings.concomitantMeds.length} selected` : undefined}
            >
              <div className="space-y-4">
                {/* Indication Preset */}
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Indication Preset</label>
                  <Select
                    value={settings.diseaseState.indication}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        diseaseState: { ...prev.diseaseState, indication: v },
                        concomitantMeds: [],
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="diabetes">당뇨 (Diabetes)</SelectItem>
                      <SelectItem value="obesity">비만 (Obesity)</SelectItem>
                      <SelectItem value="oncology">항암 (Oncology)</SelectItem>
                      <SelectItem value="nafld">NAFLD/NASH</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Medication List */}
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Select Medications</label>
                  <div className="grid grid-cols-2 gap-2">
                    {DDI_PRESETS[settings.diseaseState.indication]?.map((med) => (
                      <button
                        key={med.name}
                        onClick={() => toggleMed(med.name)}
                        className={`flex items-center gap-2 p-2 rounded-lg border text-left text-sm transition-colors ${
                          settings.concomitantMeds.includes(med.name)
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border/50 bg-background/30 text-muted-foreground hover:border-primary/50"
                        }`}
                      >
                        {settings.concomitantMeds.includes(med.name) ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <div className="h-4 w-4 rounded border border-muted-foreground/50" />
                        )}
                        <div>
                          <span className="block">{med.name}</span>
                          {med.cyp && (
                            <span className="text-xs text-muted-foreground">{med.cyp}</span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </SettingsSection>

            {/* Dosing Regimen */}
            <SettingsSection title="Dosing Regimen" icon={Syringe}>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Regimen Type</label>
                  <Select
                    value={settings.dosingRegimen.type}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        dosingRegimen: { ...prev.dosingRegimen, type: v },
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="escalation">Dose Escalation (3+3)</SelectItem>
                      <SelectItem value="dose_finding">Dose Finding</SelectItem>
                      <SelectItem value="fixed">Fixed Dose</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-sm text-muted-foreground">Frequency</label>
                  <Select
                    value={settings.dosingRegimen.frequency}
                    onValueChange={(v) =>
                      setSettings((prev) => ({
                        ...prev,
                        dosingRegimen: { ...prev.dosingRegimen, frequency: v },
                      }))
                    }
                  >
                    <SelectTrigger className="bg-background/50">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="QD">QD (Once Daily)</SelectItem>
                      <SelectItem value="BID">BID (Twice Daily)</SelectItem>
                      <SelectItem value="QW">QW (Once Weekly)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </SettingsSection>
          </motion.div>
        )}
      </AnimatePresence>

      {/* PK Impact Preview */}
      <PKImpactIndicator impacts={pkImpacts} />

      {/* Run Simulation Button */}
      <Button
        onClick={runSimulation}
        className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-primary to-cyan-500 hover:from-primary/90 hover:to-cyan-500/90"
      >
        <Activity className="mr-2 h-5 w-5" />
        Run Visual Simulation
      </Button>
      <p className="text-center text-xs text-muted-foreground">
        Estimation time: ~1 second (N=100) to ~10 seconds (N=1000)
      </p>
    </div>
  );
}
