import DashboardLayout from "@/components/DashboardLayout";
import KPICard from "@/components/KPICard";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import {
  BarChart3,
  Target,
  Users,
  TrendingUp,
  Activity,
  Dna,
  CheckCircle2,
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
  Legend,
  ComposedChart,
  Area,
} from "recharts";

// Kaplan-Meier Data
const kmData = [
  { month: 0, treatment: 100, control: 100 },
  { month: 3, treatment: 95, control: 88 },
  { month: 6, treatment: 88, control: 75 },
  { month: 9, treatment: 82, control: 62 },
  { month: 12, treatment: 75, control: 52 },
  { month: 15, treatment: 70, control: 44 },
  { month: 18, treatment: 65, control: 38 },
  { month: 21, treatment: 62, control: 32 },
  { month: 24, treatment: 58, control: 28 },
  { month: 27, treatment: 55, control: 25 },
  { month: 30, treatment: 52, control: 22 },
  { month: 33, treatment: 50, control: 20 },
  { month: 36, treatment: 48, control: 18 },
];

// Forest Plot Data
const forestPlotData = [
  { subgroup: "전체 인구", hr: 0.68, lower: 0.55, upper: 0.84, n: 420, favors: "treatment" },
  { subgroup: "연령 <65세", hr: 0.62, lower: 0.48, upper: 0.80, n: 245, favors: "treatment" },
  { subgroup: "연령 ≥65세", hr: 0.75, lower: 0.58, upper: 0.97, n: 175, favors: "treatment" },
  { subgroup: "남성", hr: 0.65, lower: 0.50, upper: 0.85, n: 238, favors: "treatment" },
  { subgroup: "여성", hr: 0.72, lower: 0.54, upper: 0.96, n: 182, favors: "treatment" },
  { subgroup: "Biomarker+", hr: 0.52, lower: 0.40, upper: 0.68, n: 210, favors: "treatment" },
  { subgroup: "Biomarker-", hr: 0.88, lower: 0.68, upper: 1.14, n: 210, favors: "neutral" },
  { subgroup: "ECOG 0", hr: 0.58, lower: 0.44, upper: 0.76, n: 280, favors: "treatment" },
  { subgroup: "ECOG 1", hr: 0.82, lower: 0.62, upper: 1.08, n: 140, favors: "neutral" },
];

// PFS Data
const pfsData = [
  { group: "치료군", median: 12.4, lower: 10.8, upper: 14.2 },
  { group: "대조군", median: 7.2, lower: 6.1, upper: 8.5 },
];

// OS Data
const osData = [
  { group: "치료군", median: 24.8, lower: 21.2, upper: 28.6 },
  { group: "대조군", median: 18.2, lower: 15.4, upper: 21.5 },
];

// Genotype Subgroup Analysis
const genotypeSubgroupData = [
  { genotype: "전체", hr: 0.68, pfs: 12.4, os: 24.8, n: 420 },
  { genotype: "CYP2C19 EM", hr: 0.72, pfs: 11.8, os: 23.5, n: 252 },
  { genotype: "CYP2C19 IM", hr: 0.58, pfs: 14.2, os: 27.8, n: 126 },
  { genotype: "CYP2C19 PM", hr: 0.48, pfs: 16.8, os: 32.4, n: 42 },
];

// Response by Genotype
const responseByGenotypeData = [
  { genotype: "EM", orr: 38, dcr: 72, cr: 6, pr: 32 },
  { genotype: "IM", orr: 48, dcr: 82, cr: 10, pr: 38 },
  { genotype: "PM", orr: 62, dcr: 90, cr: 14, pr: 48 },
];

export default function Phase3() {
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
            <div className="p-2 rounded-xl bg-purple-500/20">
              <BarChart3 className="h-6 w-6 text-purple-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Phase 3: Confirmatory Trial</h1>
              <p className="text-muted-foreground">통계적 유의성 확증 및 하위 그룹 분석</p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* KPIs */}
      <section className="mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="위험비 (HR)"
            value={0.68}
            change={-32}
            changeLabel="위험 감소"
            icon={TrendingUp}
            color="purple"
            delay={0}
          />
          <KPICard
            title="p-value"
            value="<0.001"
            icon={CheckCircle2}
            color="lime"
            delay={0.1}
          />
          <KPICard
            title="시험 대상자"
            value={420}
            unit="명"
            icon={Users}
            color="cyan"
            delay={0.2}
          />
          <KPICard
            title="PFS 중앙값"
            value={12.4}
            unit="개월"
            change={72}
            changeLabel="vs 대조군"
            icon={Activity}
            color="amber"
            delay={0.3}
          />
        </div>
      </section>

      {/* Main Content */}
      <Tabs defaultValue="survival" className="space-y-6">
        <TabsList className="bg-secondary/50">
          <TabsTrigger value="survival" className="gap-2">
            <Activity className="h-4 w-4" />
            Survival Analysis
          </TabsTrigger>
          <TabsTrigger value="subgroup" className="gap-2">
            <Target className="h-4 w-4" />
            Subgroup Analysis
          </TabsTrigger>
          <TabsTrigger value="precision" className="gap-2">
            <Dna className="h-4 w-4" />
            Precision Medicine
          </TabsTrigger>
        </TabsList>

        {/* Survival Analysis Tab */}
        <TabsContent value="survival" className="space-y-6">
          {/* Kaplan-Meier Curve */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-2">Kaplan-Meier 생존 곡선 (PFS)</h3>
              <p className="text-sm text-muted-foreground mb-4">
                무진행 생존기간 분석 | HR 0.68 (95% CI: 0.55-0.84), p &lt; 0.001
              </p>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={kmData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="month" 
                      stroke="#64748B" 
                      fontSize={12}
                      label={{ value: 'Time (months)', position: 'bottom', fill: '#64748B' }}
                    />
                    <YAxis 
                      stroke="#64748B" 
                      fontSize={12}
                      domain={[0, 100]}
                      label={{ value: 'Survival Probability (%)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1E293B', 
                        border: '1px solid #334155',
                        borderRadius: '8px'
                      }}
                      formatter={(value: number) => [`${value}%`, '']}
                    />
                    <Legend />
                    <ReferenceLine y={50} stroke="#64748B" strokeDasharray="3 3" />
                    <Line 
                      type="stepAfter" 
                      dataKey="treatment" 
                      stroke="#A855F7" 
                      strokeWidth={3}
                      name="치료군 (Drug X)"
                      dot={false}
                    />
                    <Line 
                      type="stepAfter" 
                      dataKey="control" 
                      stroke="#64748B" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="대조군 (SOC)"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </motion.div>

          {/* Survival Summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-4">무진행 생존기간 (PFS)</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
                    <div>
                      <p className="text-sm text-muted-foreground">치료군</p>
                      <p className="text-3xl font-bold data-value text-purple-400">12.4</p>
                      <p className="text-xs text-muted-foreground">95% CI: 10.8-14.2 개월</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">대조군</p>
                      <p className="text-3xl font-bold data-value">7.2</p>
                      <p className="text-xs text-muted-foreground">95% CI: 6.1-8.5 개월</p>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50 text-center">
                    <p className="text-sm text-muted-foreground">PFS 개선</p>
                    <p className="text-2xl font-bold data-value text-lime-400">+5.2 개월 (+72%)</p>
                  </div>
                </div>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-4">전체 생존기간 (OS)</h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                    <div>
                      <p className="text-sm text-muted-foreground">치료군</p>
                      <p className="text-3xl font-bold data-value text-cyan-400">24.8</p>
                      <p className="text-xs text-muted-foreground">95% CI: 21.2-28.6 개월</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">대조군</p>
                      <p className="text-3xl font-bold data-value">18.2</p>
                      <p className="text-xs text-muted-foreground">95% CI: 15.4-21.5 개월</p>
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50 text-center">
                    <p className="text-sm text-muted-foreground">OS 개선</p>
                    <p className="text-2xl font-bold data-value text-lime-400">+6.6 개월 (+36%)</p>
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </TabsContent>

        {/* Subgroup Analysis Tab */}
        <TabsContent value="subgroup" className="space-y-6">
          {/* Forest Plot */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-2">Forest Plot - 하위 그룹 분석</h3>
              <p className="text-sm text-muted-foreground mb-4">
                다양한 환자 특성에 따른 위험비 (HR) 분석 | HR &lt; 1.0 = 치료군 유리
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-semibold w-32">하위 그룹</th>
                      <th className="text-center py-3 px-4 font-semibold w-20">N</th>
                      <th className="text-center py-3 px-4 font-semibold w-32">HR (95% CI)</th>
                      <th className="text-left py-3 px-4 font-semibold">Forest Plot</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forestPlotData.map((row, index) => (
                      <tr key={index} className="border-b border-border/50">
                        <td className="py-3 px-4 font-medium">{row.subgroup}</td>
                        <td className="text-center py-3 px-4 data-value">{row.n}</td>
                        <td className="text-center py-3 px-4 data-value">
                          {row.hr.toFixed(2)} ({row.lower.toFixed(2)}-{row.upper.toFixed(2)})
                        </td>
                        <td className="py-3 px-4">
                          <div className="relative h-6 flex items-center">
                            {/* Reference line at 1.0 */}
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-muted-foreground" />
                            {/* CI bar */}
                            <div 
                              className="absolute h-1 bg-muted-foreground/50 rounded"
                              style={{
                                left: `${((row.lower - 0.3) / 1.0) * 100}%`,
                                width: `${((row.upper - row.lower) / 1.0) * 100}%`,
                              }}
                            />
                            {/* Point estimate */}
                            <div 
                              className={`absolute w-3 h-3 rounded-full ${
                                row.favors === "treatment" ? "bg-purple-500" : "bg-amber-500"
                              }`}
                              style={{
                                left: `calc(${((row.hr - 0.3) / 1.0) * 100}% - 6px)`,
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex justify-center gap-8 text-sm text-muted-foreground">
                <span>← 치료군 유리</span>
                <span>|</span>
                <span>대조군 유리 →</span>
              </div>
            </Card>
          </motion.div>

          {/* Subgroup Insights */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-purple-500/30">
              <h3 className="text-lg font-semibold mb-4">하위 그룹 분석 주요 발견</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-4 rounded-lg bg-lime-500/10 border border-lime-500/30">
                  <h4 className="font-semibold text-lime-400 mb-2">최대 이점 그룹</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex justify-between">
                      <span>Biomarker+ 환자</span>
                      <span className="data-value text-lime-400">HR 0.52</span>
                    </li>
                    <li className="flex justify-between">
                      <span>ECOG 0 환자</span>
                      <span className="data-value text-lime-400">HR 0.58</span>
                    </li>
                    <li className="flex justify-between">
                      <span>65세 미만</span>
                      <span className="data-value text-lime-400">HR 0.62</span>
                    </li>
                  </ul>
                </div>
                <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/30">
                  <h4 className="font-semibold text-amber-400 mb-2">이점 불확실 그룹</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex justify-between">
                      <span>Biomarker- 환자</span>
                      <span className="data-value text-amber-400">HR 0.88</span>
                    </li>
                    <li className="flex justify-between">
                      <span>ECOG 1 환자</span>
                      <span className="data-value text-amber-400">HR 0.82</span>
                    </li>
                    <li className="text-xs mt-2">
                      * 95% CI가 1.0을 포함하여 통계적 유의성 미달
                    </li>
                  </ul>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>

        {/* Precision Medicine Tab */}
        <TabsContent value="precision" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Genotype Subgroup */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">CYP2C19 유전자형별 생존 분석</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  약물 대사 효소 유전자형에 따른 치료 효과 차이
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={genotypeSubgroupData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#64748B" fontSize={12} domain={[0, 35]} />
                      <YAxis dataKey="genotype" type="category" stroke="#64748B" fontSize={12} width={100} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="pfs" fill="#A855F7" name="PFS (개월)" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="os" fill="#06B6D4" name="OS (개월)" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* Response by Genotype */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">유전자형별 반응률</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  PM 환자에서 가장 높은 반응률 관찰
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={responseByGenotypeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="genotype" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} domain={[0, 100]} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="cr" stackId="a" fill="#84CC16" name="CR (%)" radius={[0, 0, 0, 0]} />
                      <Bar dataKey="pr" stackId="a" fill="#06B6D4" name="PR (%)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* Precision Medicine Summary */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-cyan-500/30">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-cyan-500/20">
                  <Dna className="h-6 w-6 text-cyan-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold mb-2">정밀 의료 (Precision Medicine) 전략</h3>
                  <p className="text-muted-foreground mb-4">
                    약물유전체 분석 결과, CYP2C19 Poor Metabolizer (PM) 환자군에서 가장 우수한 치료 효과가 관찰되었습니다.
                    이는 약물 노출 증가로 인한 효능 향상으로 해석되며, FDA 승인 시 
                    <span className="text-cyan-400 font-bold"> 동반진단(Companion Diagnostics)</span> 개발을 권장합니다.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">PM 환자 HR</p>
                      <p className="text-2xl font-bold data-value text-cyan-400">0.48</p>
                      <p className="text-xs text-muted-foreground">52% 위험 감소</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">PM 환자 ORR</p>
                      <p className="text-2xl font-bold data-value text-lime-400">62%</p>
                      <p className="text-xs text-muted-foreground">vs 전체 42%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">PM 환자 PFS</p>
                      <p className="text-2xl font-bold data-value">16.8</p>
                      <p className="text-xs text-muted-foreground">개월</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">PM 환자 OS</p>
                      <p className="text-2xl font-bold data-value">32.4</p>
                      <p className="text-xs text-muted-foreground">개월</p>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* FDA Submission Strategy */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-4">FDA 승인 전략 요약</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-4 rounded-lg bg-lime-500/10 border border-lime-500/30">
                  <CheckCircle2 className="h-6 w-6 text-lime-400 mb-2" />
                  <h4 className="font-semibold mb-2">Primary Endpoint 달성</h4>
                  <p className="text-sm text-muted-foreground">
                    PFS에서 통계적으로 유의한 개선 (HR 0.68, p&lt;0.001)
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
                  <CheckCircle2 className="h-6 w-6 text-cyan-400 mb-2" />
                  <h4 className="font-semibold mb-2">OS 경향성 확인</h4>
                  <p className="text-sm text-muted-foreground">
                    전체 생존기간에서도 긍정적 경향 (HR 0.72, 중간 분석)
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
                  <CheckCircle2 className="h-6 w-6 text-purple-400 mb-2" />
                  <h4 className="font-semibold mb-2">Biomarker 전략</h4>
                  <p className="text-sm text-muted-foreground">
                    동반진단 개발로 정밀 의료 적응증 확보 가능
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
      </Tabs>
    </DashboardLayout>
  );
}
