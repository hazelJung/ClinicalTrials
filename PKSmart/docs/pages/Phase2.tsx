import DashboardLayout from "@/components/DashboardLayout";
import KPICard from "@/components/KPICard";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import {
  TestTube2,
  Target,
  TrendingDown,
  Users,
  Pill,
  Activity,
  Dna,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
  ComposedChart,
  Area,
} from "recharts";

// Waterfall Plot Data - Tumor Response
const waterfallData = [
  { patient: "P01", change: -72, biomarker: "+" },
  { patient: "P02", change: -65, biomarker: "+" },
  { patient: "P03", change: -58, biomarker: "+" },
  { patient: "P04", change: -52, biomarker: "-" },
  { patient: "P05", change: -48, biomarker: "+" },
  { patient: "P06", change: -45, biomarker: "+" },
  { patient: "P07", change: -42, biomarker: "-" },
  { patient: "P08", change: -38, biomarker: "+" },
  { patient: "P09", change: -35, biomarker: "-" },
  { patient: "P10", change: -32, biomarker: "+" },
  { patient: "P11", change: -28, biomarker: "-" },
  { patient: "P12", change: -25, biomarker: "+" },
  { patient: "P13", change: -22, biomarker: "-" },
  { patient: "P14", change: -18, biomarker: "+" },
  { patient: "P15", change: -15, biomarker: "-" },
  { patient: "P16", change: -12, biomarker: "-" },
  { patient: "P17", change: -8, biomarker: "-" },
  { patient: "P18", change: -5, biomarker: "-" },
  { patient: "P19", change: 2, biomarker: "-" },
  { patient: "P20", change: 8, biomarker: "-" },
  { patient: "P21", change: 15, biomarker: "-" },
  { patient: "P22", change: 22, biomarker: "-" },
  { patient: "P23", change: 28, biomarker: "-" },
  { patient: "P24", change: 35, biomarker: "-" },
].sort((a, b) => a.change - b.change);

// Dose-Response Curve Data
const doseResponseData = [
  { dose: 50, efficacy: 15, safety: 5 },
  { dose: 100, efficacy: 28, safety: 12 },
  { dose: 150, efficacy: 42, safety: 22 },
  { dose: 200, efficacy: 52, safety: 35 },
  { dose: 250, efficacy: 58, safety: 52 },
  { dose: 300, efficacy: 62, safety: 72 },
];

// Swimmer Plot Data
const swimmerData = [
  { patient: "P01", start: 0, duration: 24, response: 4, ongoing: true },
  { patient: "P02", start: 0, duration: 22, response: 6, ongoing: true },
  { patient: "P03", start: 0, duration: 20, response: 8, ongoing: false },
  { patient: "P04", start: 0, duration: 18, response: 4, ongoing: true },
  { patient: "P05", start: 0, duration: 16, response: 6, ongoing: false },
  { patient: "P06", start: 0, duration: 14, response: 8, ongoing: true },
  { patient: "P07", start: 0, duration: 12, response: 4, ongoing: false },
  { patient: "P08", start: 0, duration: 10, response: 6, ongoing: false },
];

// Biomarker Response Comparison
const biomarkerComparisonData = [
  { group: "Biomarker+", orr: 62, dcr: 88, n: 42 },
  { group: "Biomarker-", orr: 24, dcr: 58, n: 44 },
  { group: "Overall", orr: 42, dcr: 72, n: 86 },
];

// Spider Plot Data (Multi-axis)
const spiderData = [
  { dose: "100mg", efficacy: 28, safety: 88, pk: 75, tolerability: 92 },
  { dose: "150mg", efficacy: 42, safety: 78, pk: 82, tolerability: 85 },
  { dose: "200mg", efficacy: 52, safety: 65, pk: 88, tolerability: 72 },
];

export default function Phase2() {
  return (
    <DashboardLayout>
      {/* Header */}
      <section className="mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-lime-500/20">
              <TestTube2 className="h-6 w-6 text-lime-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Phase 2: PoC & Dose Finding</h1>
              <p className="text-muted-foreground">개념 증명 및 최적 용량 탐색</p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* KPIs */}
      <section className="mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="객관적 반응률 (ORR)"
            value={42}
            unit="%"
            change={18}
            changeLabel="vs 대조군"
            icon={Target}
            color="lime"
            delay={0}
          />
          <KPICard
            title="질병 통제율 (DCR)"
            value={78}
            unit="%"
            icon={Activity}
            color="cyan"
            delay={0.1}
          />
          <KPICard
            title="시험 대상자"
            value={86}
            unit="명"
            icon={Users}
            color="purple"
            delay={0.2}
          />
          <KPICard
            title="최적 용량 (RP2D)"
            value={150}
            unit="mg"
            icon={Pill}
            color="amber"
            delay={0.3}
          />
        </div>
      </section>

      {/* Main Content */}
      <Tabs defaultValue="efficacy" className="space-y-6">
        <TabsList className="bg-secondary/50">
          <TabsTrigger value="efficacy" className="gap-2">
            <Target className="h-4 w-4" />
            Efficacy
          </TabsTrigger>
          <TabsTrigger value="dose" className="gap-2">
            <Pill className="h-4 w-4" />
            Dose-Response
          </TabsTrigger>
          <TabsTrigger value="biomarker" className="gap-2">
            <Dna className="h-4 w-4" />
            Biomarker Analysis
          </TabsTrigger>
        </TabsList>

        {/* Efficacy Tab */}
        <TabsContent value="efficacy" className="space-y-6">
          {/* Waterfall Plot */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-2">Waterfall Plot - 종양 크기 변화</h3>
              <p className="text-sm text-muted-foreground mb-4">
                각 환자의 기저치 대비 최대 종양 크기 변화율 (%) | 
                <span className="text-lime-400 ml-2">● Biomarker+</span>
                <span className="text-muted-foreground ml-2">● Biomarker-</span>
              </p>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={waterfallData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      type="number" 
                      stroke="#64748B" 
                      fontSize={12}
                      domain={[-100, 50]}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <YAxis 
                      dataKey="patient" 
                      type="category" 
                      stroke="#64748B" 
                      fontSize={10}
                      width={40}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1E293B', 
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                      formatter={(value: number) => [`${value}%`, '변화율']}
                    />
                    <ReferenceLine x={-30} stroke="#84CC16" strokeDasharray="3 3" label={{ value: 'PR', fill: '#84CC16', fontSize: 10 }} />
                    <ReferenceLine x={20} stroke="#DC2626" strokeDasharray="3 3" label={{ value: 'PD', fill: '#DC2626', fontSize: 10 }} />
                    <Bar dataKey="change" radius={[0, 4, 4, 0]}>
                      {waterfallData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.biomarker === "+" ? "#84CC16" : "#64748B"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-lime-500" />
                  <span>PR (Partial Response): ≤-30%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-cyan-500" />
                  <span>SD (Stable Disease): -30% ~ +20%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-red-500" />
                  <span>PD (Progressive Disease): ≥+20%</span>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Response Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm text-center">
                <div className="text-4xl font-bold text-lime-400 mb-2">42%</div>
                <div className="text-sm text-muted-foreground">객관적 반응률 (ORR)</div>
                <div className="text-xs text-muted-foreground mt-1">CR: 8% + PR: 34%</div>
              </Card>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm text-center">
                <div className="text-4xl font-bold text-cyan-400 mb-2">78%</div>
                <div className="text-sm text-muted-foreground">질병 통제율 (DCR)</div>
                <div className="text-xs text-muted-foreground mt-1">CR + PR + SD</div>
              </Card>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm text-center">
                <div className="text-4xl font-bold text-purple-400 mb-2">8.2</div>
                <div className="text-sm text-muted-foreground">반응 기간 중앙값 (개월)</div>
                <div className="text-xs text-muted-foreground mt-1">95% CI: 6.4-10.8</div>
              </Card>
            </motion.div>
          </div>
        </TabsContent>

        {/* Dose-Response Tab */}
        <TabsContent value="dose" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Dose-Response Curve */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">용량-반응 곡선</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  유효성과 안전성의 균형점 (Therapeutic Window) 탐색
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={doseResponseData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis 
                        dataKey="dose" 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'Dose (mg)', position: 'bottom', fill: '#64748B' }}
                      />
                      <YAxis 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'Rate (%)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="efficacy" 
                        fill="#84CC16" 
                        fillOpacity={0.2}
                        stroke="#84CC16"
                        strokeWidth={2}
                        name="유효성 (ORR)"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="safety" 
                        stroke="#DC2626" 
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        name="독성 (Grade 3+)"
                        dot={{ fill: '#DC2626' }}
                      />
                      <ReferenceLine x={150} stroke="#06B6D4" strokeWidth={2} label={{ value: 'RP2D', fill: '#06B6D4', fontSize: 12 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* Dose Selection Rationale */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">용량별 효능-안전성 프로파일</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  3상 권장 용량 (RP2D) 결정 근거
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spiderData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="dose" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} domain={[0, 100]} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar dataKey="efficacy" fill="#84CC16" name="유효성" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="safety" fill="#06B6D4" name="안전성" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="tolerability" fill="#A855F7" name="내약성" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* RP2D Selection Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-lime-500/30">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-lime-500/20">
                  <Pill className="h-6 w-6 text-lime-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-2">3상 권장 용량 (RP2D) 결정</h3>
                  <p className="text-muted-foreground mb-4">
                    용량-반응 분석 결과, <span className="text-lime-400 font-bold">150mg</span>이 최적의 치료 용량으로 선정되었습니다.
                    이 용량에서 유효성(ORR 42%)과 안전성(Grade 3+ AE 22%)의 균형이 가장 우수하며,
                    200mg 이상에서는 독성 증가 대비 유효성 증가가 미미합니다.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">선정 용량</p>
                      <p className="text-2xl font-bold data-value text-lime-400">150mg</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">ORR at RP2D</p>
                      <p className="text-2xl font-bold data-value">42%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">Grade 3+ AE</p>
                      <p className="text-2xl font-bold data-value">22%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">Therapeutic Index</p>
                      <p className="text-2xl font-bold data-value text-cyan-400">1.91</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>

        {/* Biomarker Analysis Tab */}
        <TabsContent value="biomarker" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Biomarker Comparison */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">바이오마커별 반응률 비교</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Biomarker+ 환자군에서 유의하게 높은 반응률 관찰
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={biomarkerComparisonData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="group" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} domain={[0, 100]} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar dataKey="orr" fill="#84CC16" name="ORR (%)" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="dcr" fill="#06B6D4" name="DCR (%)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* Responder Analysis */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">Responder vs Non-responder 분석</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  바이오마커 상태에 따른 치료 반응 예측
                </p>
                <div className="space-y-4">
                  <div className="p-4 rounded-lg bg-lime-500/10 border border-lime-500/30">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold text-lime-400">Biomarker Positive</span>
                      <span className="text-sm text-muted-foreground">n=42</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">ORR</p>
                        <p className="text-2xl font-bold data-value">62%</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">DCR</p>
                        <p className="text-2xl font-bold data-value">88%</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50 border border-border">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-semibold">Biomarker Negative</span>
                      <span className="text-sm text-muted-foreground">n=44</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">ORR</p>
                        <p className="text-2xl font-bold data-value">24%</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">DCR</p>
                        <p className="text-2xl font-bold data-value">58%</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-cyan-400">ORR 차이 (Δ)</span>
                      <span className="text-2xl font-bold data-value text-cyan-400">+38%</span>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">p-value &lt; 0.001</p>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* Enrichment Strategy */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-purple-500/30">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-purple-500/20">
                  <Dna className="h-6 w-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">3상 환자 선별 전략 (Enrichment Strategy)</h3>
                  <p className="text-muted-foreground mb-4">
                    Biomarker+ 환자군에서 현저히 높은 반응률(ORR 62% vs 24%)이 관찰되어,
                    3상 임상시험에서는 <span className="text-purple-400 font-bold">Biomarker-guided Enrichment</span> 전략을 권장합니다.
                    이를 통해 임상적 이점이 예상되는 환자군에 집중하여 시험의 성공 확률을 높일 수 있습니다.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">예상 ORR (Enriched)</p>
                      <p className="text-2xl font-bold data-value text-purple-400">62%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">필요 표본 수 감소</p>
                      <p className="text-2xl font-bold data-value">-35%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">동반진단 개발</p>
                      <p className="text-2xl font-bold data-value text-lime-400">권장</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
      </Tabs>
    </DashboardLayout>
  );
}
