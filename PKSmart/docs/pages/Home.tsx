import DashboardLayout from "@/components/DashboardLayout";
import KPICard from "@/components/KPICard";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { Link } from "wouter";
import {
  FlaskConical,
  TestTube2,
  BarChart3,
  Users,
  Shield,
  Target,
  TrendingUp,
  ArrowRight,
  Dna,
  Activity,
  FileText,
  Settings2,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

// Sample data for overview charts
const timelineData = [
  { month: "Jan", phase1: 85, phase2: 0, phase3: 0 },
  { month: "Feb", phase1: 92, phase2: 0, phase3: 0 },
  { month: "Mar", phase1: 88, phase2: 45, phase3: 0 },
  { month: "Apr", phase1: 95, phase2: 62, phase3: 0 },
  { month: "May", phase1: 90, phase2: 78, phase3: 0 },
  { month: "Jun", phase1: 93, phase2: 85, phase3: 35 },
  { month: "Jul", phase1: 91, phase2: 89, phase3: 52 },
  { month: "Aug", phase1: 94, phase2: 92, phase3: 68 },
];

const enrollmentData = [
  { week: "W1", enrolled: 12, target: 15 },
  { week: "W2", enrolled: 28, target: 30 },
  { week: "W3", enrolled: 45, target: 45 },
  { week: "W4", enrolled: 58, target: 60 },
  { week: "W5", enrolled: 72, target: 75 },
  { week: "W6", enrolled: 89, target: 90 },
  { week: "W7", enrolled: 102, target: 105 },
  { week: "W8", enrolled: 118, target: 120 },
];

const phaseCards = [
  {
    phase: "Phase 1",
    title: "Safety & PK/PD",
    description: "안전성 평가 및 약동학/약력학 분석",
    icon: FlaskConical,
    color: "cyan" as const,
    path: "/phase1",
    metrics: [
      { label: "MTD", value: "200mg" },
      { label: "DLT Rate", value: "8.3%" },
      { label: "Subjects", value: "24" },
    ],
    status: "완료",
    statusColor: "text-lime-400",
  },
  {
    phase: "Phase 2",
    title: "PoC & Dose Finding",
    description: "개념 증명 및 최적 용량 탐색",
    icon: TestTube2,
    color: "lime" as const,
    path: "/phase2",
    metrics: [
      { label: "ORR", value: "42%" },
      { label: "DCR", value: "78%" },
      { label: "Subjects", value: "86" },
    ],
    status: "진행중",
    statusColor: "text-cyan-400",
  },
  {
    phase: "Phase 3",
    title: "Confirmatory Trial",
    description: "통계적 유의성 확증 및 하위 그룹 분석",
    icon: BarChart3,
    color: "purple" as const,
    path: "/phase3",
    metrics: [
      { label: "HR", value: "0.68" },
      { label: "p-value", value: "<0.001" },
      { label: "Subjects", value: "420" },
    ],
    status: "시뮬레이션",
    statusColor: "text-amber-400",
  },
];

export default function Home() {
  return (
    <DashboardLayout>
      {/* Hero Section */}
      <section className="relative mb-8 rounded-2xl overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-30"
          style={{ 
            backgroundImage: `url('https://private-us-east-1.manuscdn.com/sessionFile/tWGgdbARcMOKcDdZwhPNw9/sandbox/UbmshEIfDkN7CjVF5bPBw7-img-1_1770034153000_na1fn_aGVyby1iYWNrZ3JvdW5k.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdFdHZ2RiQVJjTU9LY0RkWndoUE53OS9zYW5kYm94L1VibXNoRUlmRGtON0NqVkY1YlBCdzctaW1nLTFfMTc3MDAzNDE1MzAwMF9uYTFmbl9hR1Z5YnkxaVlXTnJaM0p2ZFc1ay5wbmc~eC1vc3MtcHJvY2Vzcz1pbWFnZS9yZXNpemUsd18xOTIwLGhfMTkyMC9mb3JtYXQsd2VicC9xdWFsaXR5LHFfODAiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=EFQhDHFci0Fab-SySI9-71i241UqFUz1XilnD0hfjGRPql5TEUgpnjciQhX-2TYgstNGR9i6NBa94NO~PfLghrm09S~LHzF19Tegz5u5Grbe~A1Z67Ufk7MTu~HghKEPCHrU7ZiNj-Tn9TjC7JFNP5DES0utSgEHQKsNyNBX0PKc9akYEYnhKTMKQPOz5WzBWi7FDqC3Kzj5v5QgJbC0RRlno3Z~fT70I8C-kpAspUdG6rgdTST7BP2j3O35wQlKFS0FvJXhx~D~qdbwogmACYcNmmlaRGx3QhOnx3c7n8x~M4J5PQGAlMCSCwH2SV02gWhZfQeumlIICdpvwtSYTQ__')`
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-r from-background via-background/90 to-background/70" />
        
        <div className="relative z-10 px-8 py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center gap-3 mb-4">
              <Dna className="h-6 w-6 text-primary" />
              <span className="text-sm font-medium text-primary">PBPK & Pharmacogenomics</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">
              FDA 승인용 임상시험
              <br />
              <span className="gradient-text">시뮬레이션 대시보드</span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mb-6">
              약물유전체 데이터와 PBPK 모델링을 기반으로 한 신약개발 임상시험 시뮬레이션 시스템입니다.
              임상 1-3상 단계별 핵심 결과 지표와 시각화를 통해 FDA 승인 준비를 지원합니다.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link href="/phase1">
                <Button className="gap-2">
                  시뮬레이션 시작
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <Link href="/cohort-settings">
                <Button variant="outline" className="gap-2">
                  <Settings2 className="h-4 w-4" />
                  코호트 설정
                </Button>
              </Link>
              <Button variant="outline" className="gap-2">
                <FileText className="h-4 w-4" />
                분석 보고서
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* KPI Overview */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          핵심 지표 개요
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="총 시험 대상자"
            value={530}
            unit="명"
            change={12}
            changeLabel="vs 목표"
            icon={Users}
            color="cyan"
            delay={0}
          />
          <KPICard
            title="안전성 지표"
            value="Grade 3+"
            unit="8.2%"
            change={-2.1}
            changeLabel="vs 예상"
            icon={Shield}
            color="lime"
            delay={0.1}
          />
          <KPICard
            title="유효성 (ORR)"
            value={42}
            unit="%"
            change={8}
            changeLabel="vs 대조군"
            icon={Target}
            color="purple"
            delay={0.2}
          />
          <KPICard
            title="시뮬레이션 정확도"
            value={94.2}
            unit="%"
            change={3.5}
            changeLabel="vs 이전"
            icon={TrendingUp}
            color="amber"
            delay={0.3}
          />
        </div>
      </section>

      {/* Phase Cards */}
      <section className="mb-8">
        <h2 className="text-xl font-semibold mb-4">임상 단계별 현황</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {phaseCards.map((phase, index) => {
            const Icon = phase.icon;
            return (
              <motion.div
                key={phase.phase}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Link href={phase.path}>
                  <Card className={`
                    relative overflow-hidden p-6 h-full card-hover cursor-pointer
                    bg-card/50 backdrop-blur-sm border-border/50
                    hover:border-primary/50 transition-all duration-300
                  `}>
                    {/* Background Pattern */}
                    <div className="absolute inset-0 opacity-10">
                      <div 
                        className="absolute inset-0 bg-cover bg-center"
                        style={{ 
                          backgroundImage: phase.color === "cyan" 
                            ? `url('https://private-us-east-1.manuscdn.com/sessionFile/tWGgdbARcMOKcDdZwhPNw9/sandbox/UbmshEIfDkN7CjVF5bPBw7-img-2_1770034160000_na1fn_cGhhc2UxLWNhcmQtYmc.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdFdHZ2RiQVJjTU9LY0RkWndoUE53OS9zYW5kYm94L1VibXNoRUlmRGtON0NqVkY1YlBCdzctaW1nLTJfMTc3MDAzNDE2MDAwMF9uYTFmbl9jR2hoYzJVeExXTmhjbVF0WW1jLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3Jlc2l6ZSx3XzE5MjAsaF8xOTIwL2Zvcm1hdCx3ZWJwL3F1YWxpdHkscV84MCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=cRBQGMQneZBEdK5QLB4A3i~1wne8AQGjSkNEDB01e2zhT9wqwg0XFm~EINlWYYRFd-L4T1E3lf3MmFMsQ83llmr7-bzdtYLFngFXsZ35PQml8IHP6Yhk~XYPB~2goukJqW-0WtiFizzvJ~OEOYbjFQHYUt8TOV6YWLVQJ2dJHkB8biF5uhFdswPFlJMGo47cMfr3c60U8tqYa92YBsS7xd21nmEIpBJczN4dKq0h~Ad5KcoS7RdOqBCONj6LgqrmFHnR9R-MhctXg82jvaDxEAZ8p-dKlic5nXTberxTK3ZKKAHguQhLl3UPTv7iOu90RMB97aKyXEOGRxalJAsMyg__')`
                            : phase.color === "lime"
                            ? `url('https://private-us-east-1.manuscdn.com/sessionFile/tWGgdbARcMOKcDdZwhPNw9/sandbox/UbmshEIfDkN7CjVF5bPBw7-img-3_1770034154000_na1fn_cGhhc2UyLWNhcmQtYmc.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdFdHZ2RiQVJjTU9LY0RkWndoUE53OS9zYW5kYm94L1VibXNoRUlmRGtON0NqVkY1YlBCdzctaW1nLTNfMTc3MDAzNDE1NDAwMF9uYTFmbl9jR2hoYzJVeUxXTmhjbVF0WW1jLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3Jlc2l6ZSx3XzE5MjAsaF8xOTIwL2Zvcm1hdCx3ZWJwL3F1YWxpdHkscV84MCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=MS4LXRo7IjWseavTGH3HxjnOdkMx5q9oavcLExpdpWv4Rrus6p8UK6x-cLR9U0kfZdNiVjJlu8fr01FUaw5UR7DIk0Gj0MD62OuPL~Dj4fOWbK4eNWaDpdJ4So3FDyF9EswJ3oWOG3GuMFGB-LbCXod5ZcTpt8dtF2XI4QqLKIkGOZf3NYibGDfGAEuhBuSki~Rcmhffdon59Du17cTpyHXMnlIGPwdob-QHSPON6f2haCGgLmH0NHz57Uwdox2TwHyeVMIg5hL~ZvY33K2reSba7l9mxlfynZ~Za9oukAPhClTv7yNGjRdeqY47k9vi3PXRyjsSovbwKyCUhPSjFw__')`
                            : `url('https://private-us-east-1.manuscdn.com/sessionFile/tWGgdbARcMOKcDdZwhPNw9/sandbox/UbmshEIfDkN7CjVF5bPBw7-img-4_1770034153000_na1fn_cGhhc2UzLWNhcmQtYmc.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdFdHZ2RiQVJjTU9LY0RkWndoUE53OS9zYW5kYm94L1VibXNoRUlmRGtON0NqVkY1YlBCdzctaW1nLTRfMTc3MDAzNDE1MzAwMF9uYTFmbl9jR2hoYzJVekxXTmhjbVF0WW1jLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3Jlc2l6ZSx3XzE5MjAsaF8xOTIwL2Zvcm1hdCx3ZWJwL3F1YWxpdHkscV84MCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=bhBmJvrh7fmFVRGH6kb60jbGwhK~9A5C7kPQfTLZcLMtb3kcGilHUjHK6nrOz3JwFeNsB9ALDlwE5vxB9lu2ADQHR85XxOzCSTHtqNBVGQSC1V5opu9h-3Ys2Ux7MYCQCDcJL-js6DHBXAJr-QcbAYaVHUHvPkBPhzTaapFfxZGS0cWdYVTaEjID4muocVE52vdnSKUYkgY8Tu3olWtnDCTpZDUfyxDR7MVTcX4iHPxg3GlEEvVvIP9~EfSPWr4QK8zaOL4g0b5JGN0iI6-dvxAX6NslLivs6qbaLni3iDUNUWeScwbhsndyABghcmE5Me3dzRuJtKQ78WumZM0Q5g__')`
                        }}
                      />
                    </div>

                    {/* Header */}
                    <div className="relative z-10 flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className={`
                          p-2.5 rounded-xl
                          ${phase.color === "cyan" ? "bg-cyan-500/20" : ""}
                          ${phase.color === "lime" ? "bg-lime-500/20" : ""}
                          ${phase.color === "purple" ? "bg-purple-500/20" : ""}
                        `}>
                          <Icon className={`
                            h-6 w-6
                            ${phase.color === "cyan" ? "text-cyan-400" : ""}
                            ${phase.color === "lime" ? "text-lime-400" : ""}
                            ${phase.color === "purple" ? "text-purple-400" : ""}
                          `} />
                        </div>
                        <div>
                          <h3 className="font-bold text-lg">{phase.phase}</h3>
                          <p className="text-sm text-muted-foreground">{phase.title}</p>
                        </div>
                      </div>
                      <span className={`text-xs font-medium px-2 py-1 rounded-full bg-secondary ${phase.statusColor}`}>
                        {phase.status}
                      </span>
                    </div>

                    {/* Description */}
                    <p className="relative z-10 text-sm text-muted-foreground mb-4">
                      {phase.description}
                    </p>

                    {/* Metrics */}
                    <div className="relative z-10 grid grid-cols-3 gap-2">
                      {phase.metrics.map((metric) => (
                        <div key={metric.label} className="text-center p-2 rounded-lg bg-secondary/50">
                          <p className="text-xs text-muted-foreground">{metric.label}</p>
                          <p className="font-bold data-value">{metric.value}</p>
                        </div>
                      ))}
                    </div>

                    {/* Arrow */}
                    <div className="relative z-10 mt-4 flex justify-end">
                      <ArrowRight className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </Card>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Charts Section */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Timeline Progress */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4">단계별 진행률 추이</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="colorPhase1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#06B6D4" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorPhase2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#84CC16" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#84CC16" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorPhase3" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#A855F7" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#A855F7" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="month" stroke="#64748B" fontSize={12} />
                  <YAxis stroke="#64748B" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1E293B', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="phase1" 
                    stroke="#06B6D4" 
                    fillOpacity={1} 
                    fill="url(#colorPhase1)" 
                    name="Phase 1"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="phase2" 
                    stroke="#84CC16" 
                    fillOpacity={1} 
                    fill="url(#colorPhase2)" 
                    name="Phase 2"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="phase3" 
                    stroke="#A855F7" 
                    fillOpacity={1} 
                    fill="url(#colorPhase3)" 
                    name="Phase 3"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </motion.div>

        {/* Enrollment Progress */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card className="p-6 bg-card/50 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4">환자 등록 현황</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={enrollmentData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="week" stroke="#64748B" fontSize={12} />
                  <YAxis stroke="#64748B" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1E293B', 
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="target" 
                    stroke="#64748B" 
                    strokeDasharray="5 5"
                    name="목표"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="enrolled" 
                    stroke="#06B6D4" 
                    strokeWidth={2}
                    name="등록"
                    dot={{ fill: '#06B6D4', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </motion.div>
      </section>
    </DashboardLayout>
  );
}
