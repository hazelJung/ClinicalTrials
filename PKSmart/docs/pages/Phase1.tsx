import DashboardLayout from "@/components/DashboardLayout";
import KPICard from "@/components/KPICard";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import {
  FlaskConical,
  Shield,
  Activity,
  AlertTriangle,
  Users,
  Pill,
  Clock,
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
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
} from "recharts";

// PK Profile Data - Spaghetti Plot
const pkProfileData = [
  { time: 0, p1: 0, p2: 0, p3: 0, p4: 0, p5: 0, mean: 0 },
  { time: 0.5, p1: 45, p2: 52, p3: 38, p4: 48, p5: 55, mean: 47.6 },
  { time: 1, p1: 120, p2: 135, p3: 105, p4: 128, p5: 142, mean: 126 },
  { time: 2, p1: 185, p2: 198, p3: 170, p4: 190, p5: 205, mean: 189.6 },
  { time: 4, p1: 145, p2: 158, p3: 132, p4: 150, p5: 165, mean: 150 },
  { time: 6, p1: 95, p2: 108, p3: 85, p4: 100, p5: 115, mean: 100.6 },
  { time: 8, p1: 62, p2: 72, p3: 55, p4: 65, p5: 78, mean: 66.4 },
  { time: 12, p1: 28, p2: 35, p3: 22, p4: 30, p5: 38, mean: 30.6 },
  { time: 24, p1: 8, p2: 12, p3: 5, p4: 9, p5: 14, mean: 9.6 },
];

// AE Heatmap Data
const aeHeatmapData = [
  { soc: "위장관계", dose50: 2, dose100: 4, dose150: 6, dose200: 8 },
  { soc: "신경계", dose50: 1, dose100: 2, dose150: 3, dose200: 5 },
  { soc: "피부", dose50: 0, dose100: 1, dose150: 2, dose200: 3 },
  { soc: "심혈관계", dose50: 0, dose100: 0, dose150: 1, dose200: 2 },
  { soc: "간담도계", dose50: 0, dose100: 1, dose150: 1, dose200: 2 },
];

// DLT Data
const dltData = [
  { dose: "50mg", subjects: 3, dlt: 0, rate: 0 },
  { dose: "100mg", subjects: 6, dlt: 0, rate: 0 },
  { dose: "150mg", subjects: 6, dlt: 1, rate: 16.7 },
  { dose: "200mg", subjects: 6, dlt: 1, rate: 16.7 },
  { dose: "250mg", subjects: 3, dlt: 2, rate: 66.7 },
];

// CYP450 Genotype Data
const genotypeData = [
  { genotype: "EM", auc: 1850, cmax: 195, subjects: 12 },
  { genotype: "IM", auc: 2450, cmax: 245, subjects: 8 },
  { genotype: "PM", auc: 3200, cmax: 310, subjects: 4 },
];

// PK Parameters Box Plot Data
const pkBoxData = [
  { dose: "50mg", min: 420, q1: 480, median: 520, q3: 580, max: 650 },
  { dose: "100mg", min: 850, q1: 920, median: 1020, q3: 1120, max: 1250 },
  { dose: "150mg", min: 1280, q1: 1380, median: 1520, q3: 1680, max: 1850 },
  { dose: "200mg", min: 1720, q1: 1850, median: 2050, q3: 2280, max: 2500 },
];

const COLORS = ['#06B6D4', '#84CC16', '#A855F7', '#F59E0B', '#EC4899'];

export default function Phase1() {
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
            <div className="p-2 rounded-xl bg-cyan-500/20">
              <FlaskConical className="h-6 w-6 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Phase 1: Safety & PK/PD</h1>
              <p className="text-muted-foreground">안전성 평가 및 약동학/약력학 분석</p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* KPIs */}
      <section className="mb-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="최대 내약 용량 (MTD)"
            value="200"
            unit="mg"
            icon={Pill}
            color="cyan"
            delay={0}
          />
          <KPICard
            title="DLT 발생률"
            value={8.3}
            unit="%"
            change={-5.2}
            changeLabel="vs 예상"
            icon={AlertTriangle}
            color="amber"
            delay={0.1}
          />
          <KPICard
            title="시험 대상자"
            value={24}
            unit="명"
            icon={Users}
            color="lime"
            delay={0.2}
          />
          <KPICard
            title="반감기 (t1/2)"
            value={12.4}
            unit="hr"
            icon={Clock}
            color="purple"
            delay={0.3}
          />
        </div>
      </section>

      {/* Main Content */}
      <Tabs defaultValue="pk" className="space-y-6">
        <TabsList className="bg-secondary/50">
          <TabsTrigger value="pk" className="gap-2">
            <Activity className="h-4 w-4" />
            PK Profile
          </TabsTrigger>
          <TabsTrigger value="safety" className="gap-2">
            <Shield className="h-4 w-4" />
            Safety
          </TabsTrigger>
          <TabsTrigger value="genotype" className="gap-2">
            <Dna className="h-4 w-4" />
            Pharmacogenomics
          </TabsTrigger>
        </TabsList>

        {/* PK Profile Tab */}
        <TabsContent value="pk" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Spaghetti Plot */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">PK Profile (Spaghetti Plot)</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  개별 환자별 약물 혈중 농도 변화 (200mg 코호트)
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={pkProfileData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'Time (hr)', position: 'bottom', fill: '#64748B' }}
                      />
                      <YAxis 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'Concentration (ng/mL)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Line type="monotone" dataKey="p1" stroke="#06B6D4" strokeOpacity={0.5} dot={false} name="Patient 1" />
                      <Line type="monotone" dataKey="p2" stroke="#06B6D4" strokeOpacity={0.5} dot={false} name="Patient 2" />
                      <Line type="monotone" dataKey="p3" stroke="#06B6D4" strokeOpacity={0.5} dot={false} name="Patient 3" />
                      <Line type="monotone" dataKey="p4" stroke="#06B6D4" strokeOpacity={0.5} dot={false} name="Patient 4" />
                      <Line type="monotone" dataKey="p5" stroke="#06B6D4" strokeOpacity={0.5} dot={false} name="Patient 5" />
                      <Line type="monotone" dataKey="mean" stroke="#06B6D4" strokeWidth={3} dot={{ fill: '#06B6D4' }} name="Mean" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* PK Parameters */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">용량별 AUC 분포</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  용량 비례성 평가를 위한 AUC 파라미터 분포
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={pkBoxData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="dose" stroke="#64748B" fontSize={12} />
                      <YAxis 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'AUC (ng·hr/mL)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar dataKey="median" fill="#06B6D4" name="Median AUC" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* PK Parameters Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm overflow-x-auto">
              <h3 className="text-lg font-semibold mb-4">약동학 파라미터 요약</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 font-semibold">Parameter</th>
                    <th className="text-center py-3 px-4 font-semibold">50mg</th>
                    <th className="text-center py-3 px-4 font-semibold">100mg</th>
                    <th className="text-center py-3 px-4 font-semibold">150mg</th>
                    <th className="text-center py-3 px-4 font-semibold">200mg (MTD)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 text-muted-foreground">Cmax (ng/mL)</td>
                    <td className="text-center py-3 px-4 data-value">52.3 ± 8.2</td>
                    <td className="text-center py-3 px-4 data-value">98.5 ± 15.4</td>
                    <td className="text-center py-3 px-4 data-value">148.2 ± 22.1</td>
                    <td className="text-center py-3 px-4 data-value text-cyan-400">195.6 ± 28.5</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 text-muted-foreground">AUC0-∞ (ng·hr/mL)</td>
                    <td className="text-center py-3 px-4 data-value">520 ± 85</td>
                    <td className="text-center py-3 px-4 data-value">1020 ± 165</td>
                    <td className="text-center py-3 px-4 data-value">1520 ± 248</td>
                    <td className="text-center py-3 px-4 data-value text-cyan-400">2050 ± 335</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 text-muted-foreground">Tmax (hr)</td>
                    <td className="text-center py-3 px-4 data-value">2.0 ± 0.5</td>
                    <td className="text-center py-3 px-4 data-value">2.2 ± 0.6</td>
                    <td className="text-center py-3 px-4 data-value">2.1 ± 0.5</td>
                    <td className="text-center py-3 px-4 data-value text-cyan-400">2.3 ± 0.7</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 text-muted-foreground">t1/2 (hr)</td>
                    <td className="text-center py-3 px-4 data-value">11.8 ± 2.1</td>
                    <td className="text-center py-3 px-4 data-value">12.2 ± 2.4</td>
                    <td className="text-center py-3 px-4 data-value">12.1 ± 2.2</td>
                    <td className="text-center py-3 px-4 data-value text-cyan-400">12.4 ± 2.5</td>
                  </tr>
                </tbody>
              </table>
            </Card>
          </motion.div>
        </TabsContent>

        {/* Safety Tab */}
        <TabsContent value="safety" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* DLT Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">용량 제한 독성 (DLT) 발생</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  3+3 용량 증량 디자인에 따른 DLT 현황
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={dltData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="dose" stroke="#64748B" fontSize={12} />
                      <YAxis stroke="#64748B" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="subjects" fill="#06B6D4" name="대상자 수" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="dlt" fill="#DC2626" name="DLT 발생" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* AE Summary */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">이상반응 발생 현황</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  기관계 분류(SOC)별 용량-의존적 AE 발생 빈도
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={aeHeatmapData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#64748B" fontSize={12} />
                      <YAxis dataKey="soc" type="category" stroke="#64748B" fontSize={12} width={80} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="dose50" fill="#06B6D4" name="50mg" stackId="a" />
                      <Bar dataKey="dose100" fill="#84CC16" name="100mg" stackId="a" />
                      <Bar dataKey="dose150" fill="#A855F7" name="150mg" stackId="a" />
                      <Bar dataKey="dose200" fill="#F59E0B" name="200mg" stackId="a" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* Safety Summary Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm overflow-x-auto">
              <h3 className="text-lg font-semibold mb-4">이상반응 요약 (CTCAE Grade)</h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 font-semibold">이상반응</th>
                    <th className="text-center py-3 px-4 font-semibold">Grade 1-2</th>
                    <th className="text-center py-3 px-4 font-semibold">Grade 3</th>
                    <th className="text-center py-3 px-4 font-semibold">Grade 4</th>
                    <th className="text-center py-3 px-4 font-semibold">전체 (%)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">오심 (Nausea)</td>
                    <td className="text-center py-3 px-4 data-value">8</td>
                    <td className="text-center py-3 px-4 data-value">1</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value text-amber-400">37.5%</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">두통 (Headache)</td>
                    <td className="text-center py-3 px-4 data-value">6</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value">25.0%</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">피로 (Fatigue)</td>
                    <td className="text-center py-3 px-4 data-value">5</td>
                    <td className="text-center py-3 px-4 data-value">1</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value">25.0%</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4">설사 (Diarrhea)</td>
                    <td className="text-center py-3 px-4 data-value">4</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value">0</td>
                    <td className="text-center py-3 px-4 data-value">16.7%</td>
                  </tr>
                </tbody>
              </table>
            </Card>
          </motion.div>
        </TabsContent>

        {/* Pharmacogenomics Tab */}
        <TabsContent value="genotype" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Genotype vs PK */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">CYP2C19 유전자형별 PK 비교</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  대사 효소 유전자형에 따른 약물 노출 차이
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis 
                        dataKey="auc" 
                        type="number" 
                        stroke="#64748B" 
                        fontSize={12}
                        name="AUC"
                        label={{ value: 'AUC (ng·hr/mL)', position: 'bottom', fill: '#64748B' }}
                      />
                      <YAxis 
                        dataKey="cmax" 
                        type="number" 
                        stroke="#64748B" 
                        fontSize={12}
                        name="Cmax"
                        label={{ value: 'Cmax (ng/mL)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                      />
                      <ZAxis dataKey="subjects" range={[100, 500]} name="Subjects" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                        formatter={(value, name) => [value, name]}
                      />
                      <Legend />
                      <Scatter 
                        name="EM (Extensive)" 
                        data={[genotypeData[0]]} 
                        fill="#06B6D4"
                      />
                      <Scatter 
                        name="IM (Intermediate)" 
                        data={[genotypeData[1]]} 
                        fill="#84CC16"
                      />
                      <Scatter 
                        name="PM (Poor)" 
                        data={[genotypeData[2]]} 
                        fill="#DC2626"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>

            {/* Genotype Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <Card className="p-6 bg-card/50 backdrop-blur-sm">
                <h3 className="text-lg font-semibold mb-2">유전자형별 AUC 분포</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  PM 그룹에서 유의하게 높은 약물 노출 관찰
                </p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={genotypeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="genotype" stroke="#64748B" fontSize={12} />
                      <YAxis 
                        stroke="#64748B" 
                        fontSize={12}
                        label={{ value: 'AUC (ng·hr/mL)', angle: -90, position: 'insideLeft', fill: '#64748B' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1E293B', 
                          border: '1px solid #334155',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar dataKey="auc" name="AUC" radius={[4, 4, 0, 0]}>
                        {genotypeData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={index === 2 ? '#DC2626' : index === 1 ? '#84CC16' : '#06B6D4'} 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </motion.div>
          </div>

          {/* Pharmacogenomics Insight */}
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
                <div>
                  <h3 className="text-lg font-semibold mb-2">약물유전체 분석 인사이트</h3>
                  <p className="text-muted-foreground mb-4">
                    CYP2C19 Poor Metabolizer (PM) 환자군에서 AUC가 Extensive Metabolizer (EM) 대비 약 73% 증가하였습니다.
                    이는 PM 환자에서 용량 조절이 필요할 수 있음을 시사합니다.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">EM vs PM AUC 비율</p>
                      <p className="text-2xl font-bold data-value text-cyan-400">1.73x</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">PM 환자 비율</p>
                      <p className="text-2xl font-bold data-value">16.7%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50">
                      <p className="text-sm text-muted-foreground">권장 용량 조절</p>
                      <p className="text-2xl font-bold data-value text-amber-400">50% ↓</p>
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
