/*
 * Design Philosophy: Pharmaceutical Precision
 * - Phase 3 clinical trial - SINGLE COHORT (Treatment arm) individual patient data
 * - Individual patient survival data, events, biomarker status
 * - Indigo accent color scheme
 */

import PageHeader from "@/components/PageHeader";
import StatCard from "@/components/StatCard";
import ChartCard from "@/components/ChartCard";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
  Cell,
} from "recharts";

// Individual patient survival data (Treatment arm cohort)
const patientSurvivalData = [
  { patient: "P01", os: 32, pfs: 18, status: "alive", event: "none", biomarker: "+" },
  { patient: "P02", os: 28, pfs: 16, status: "alive", event: "none", biomarker: "+" },
  { patient: "P03", os: 26, pfs: 14, status: "alive", event: "none", biomarker: "+" },
  { patient: "P04", os: 25, pfs: 15, status: "alive", event: "none", biomarker: "-" },
  { patient: "P05", os: 24, pfs: 12, status: "alive", event: "none", biomarker: "+" },
  { patient: "P06", os: 22, pfs: 11, status: "deceased", event: "death", biomarker: "-" },
  { patient: "P07", os: 21, pfs: 10, status: "alive", event: "none", biomarker: "+" },
  { patient: "P08", os: 20, pfs: 9, status: "deceased", event: "death", biomarker: "-" },
  { patient: "P09", os: 18, pfs: 8, status: "alive", event: "none", biomarker: "-" },
  { patient: "P10", os: 16, pfs: 7, status: "deceased", event: "death", biomarker: "-" },
  { patient: "P11", os: 14, pfs: 6, status: "deceased", event: "death", biomarker: "-" },
  { patient: "P12", os: 12, pfs: 5, status: "deceased", event: "death", biomarker: "-" },
  { patient: "P13", os: 28, pfs: 15, status: "alive", event: "none", biomarker: "+" },
  { patient: "P14", os: 24, pfs: 13, status: "alive", event: "none", biomarker: "+" },
  { patient: "P15", os: 22, pfs: 11, status: "alive", event: "none", biomarker: "-" },
];

// Kaplan-Meier style data for cohort
const kmData = [
  { month: 0, survival: 100, atRisk: 15 },
  { month: 3, survival: 100, atRisk: 15 },
  { month: 6, survival: 93, atRisk: 14 },
  { month: 9, survival: 87, atRisk: 13 },
  { month: 12, survival: 80, atRisk: 12 },
  { month: 15, survival: 73, atRisk: 11 },
  { month: 18, survival: 67, atRisk: 10 },
  { month: 21, survival: 67, atRisk: 10 },
  { month: 24, survival: 60, atRisk: 9 },
  { month: 27, survival: 60, atRisk: 9 },
  { month: 30, survival: 60, atRisk: 9 },
];

// Individual patient timeline events
const patientTimelineData = [
  { patient: "P01", enrollment: 0, firstResponse: 3, progression: null, death: null, lastFollowup: 32 },
  { patient: "P02", enrollment: 0, firstResponse: 4, progression: null, death: null, lastFollowup: 28 },
  { patient: "P03", enrollment: 0, firstResponse: 3, progression: null, death: null, lastFollowup: 26 },
  { patient: "P04", enrollment: 0, firstResponse: 5, progression: 20, death: null, lastFollowup: 25 },
  { patient: "P05", enrollment: 0, firstResponse: 4, progression: null, death: null, lastFollowup: 24 },
  { patient: "P06", enrollment: 0, firstResponse: 3, progression: 15, death: 22, lastFollowup: 22 },
  { patient: "P07", enrollment: 0, firstResponse: 4, progression: null, death: null, lastFollowup: 21 },
  { patient: "P08", enrollment: 0, firstResponse: 5, progression: 12, death: 20, lastFollowup: 20 },
  { patient: "P09", enrollment: 0, firstResponse: 6, progression: 14, death: null, lastFollowup: 18 },
  { patient: "P10", enrollment: 0, firstResponse: 4, progression: 10, death: 16, lastFollowup: 16 },
  { patient: "P11", enrollment: 0, firstResponse: null, progression: 8, death: 14, lastFollowup: 14 },
  { patient: "P12", enrollment: 0, firstResponse: null, progression: 6, death: 12, lastFollowup: 12 },
];

// Biomarker subgroup analysis
const biomarkerSubgroupData = [
  { patient: "P01", biomarker: "+", pdl1: 82, os: 32, pfs: 18, response: "PR" },
  { patient: "P02", biomarker: "+", pdl1: 75, os: 28, pfs: 16, response: "PR" },
  { patient: "P03", biomarker: "+", pdl1: 68, os: 26, pfs: 14, response: "PR" },
  { patient: "P05", biomarker: "+", pdl1: 72, os: 24, pfs: 12, response: "PR" },
  { patient: "P07", biomarker: "+", pdl1: 65, os: 21, pfs: 10, response: "SD" },
  { patient: "P13", biomarker: "+", pdl1: 78, os: 28, pfs: 15, response: "PR" },
  { patient: "P14", biomarker: "+", pdl1: 70, os: 24, pfs: 13, response: "PR" },
  { patient: "P04", biomarker: "-", pdl1: 25, os: 25, pfs: 15, response: "SD" },
  { patient: "P06", biomarker: "-", pdl1: 18, os: 22, pfs: 11, response: "SD" },
  { patient: "P08", biomarker: "-", pdl1: 22, os: 20, pfs: 9, response: "PD" },
  { patient: "P09", biomarker: "-", pdl1: 15, os: 18, pfs: 8, response: "SD" },
  { patient: "P10", biomarker: "-", pdl1: 12, os: 16, pfs: 7, response: "PD" },
  { patient: "P11", biomarker: "-", pdl1: 8, os: 14, pfs: 6, response: "PD" },
  { patient: "P12", biomarker: "-", pdl1: 5, os: 12, pfs: 5, response: "PD" },
  { patient: "P15", biomarker: "-", pdl1: 20, os: 22, pfs: 11, response: "SD" },
];

const chartColors = {
  teal: "oklch(0.65 0.15 180)",
  amber: "oklch(0.75 0.15 75)",
  indigo: "oklch(0.55 0.18 220)",
  rose: "oklch(0.60 0.18 25)",
  green: "oklch(0.65 0.18 145)",
  navy: "oklch(0.35 0.05 250)",
};

const responseColors: Record<string, string> = {
  PR: chartColors.teal,
  SD: chartColors.amber,
  PD: chartColors.rose,
};

export default function Phase3() {
  // Calculate cohort summary stats
  const aliveCount = patientSurvivalData.filter(p => p.status === "alive").length;
  const medianOS = 24.8; // Pre-calculated
  const medianPFS = 12.5; // Pre-calculated
  const biomarkerPositive = patientSurvivalData.filter(p => p.biomarker === "+").length;

  return (
    <div className="space-y-8">
      <PageHeader
        title="임상 3상 - 시험군 코호트"
        description="시험군(신약) 코호트의 개별 환자 생존 데이터"
        phase="Phase 3 | Treatment Arm | N=15"
        phaseColor="indigo"
      />

      {/* Cohort Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="중앙 OS"
          value={medianOS}
          unit="mo"
          description="전체 생존기간"
          accentColor="indigo"
          delay={0}
        />
        <StatCard
          label="중앙 PFS"
          value={medianPFS}
          unit="mo"
          description="무진행 생존기간"
          accentColor="teal"
          delay={100}
        />
        <StatCard
          label="생존 환자"
          value={`${aliveCount}/15`}
          description={`${((aliveCount/15)*100).toFixed(1)}% 생존율`}
          accentColor="amber"
          delay={200}
        />
        <StatCard
          label="Biomarker+"
          value={`${biomarkerPositive}/15`}
          description="바이오마커 양성"
          accentColor="indigo"
          delay={300}
        />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="survival" className="space-y-6">
        <TabsList className="bg-muted/50 p-1 rounded-lg">
          <TabsTrigger value="survival" className="rounded-md px-4 py-2 text-sm font-medium">
            생존 분석
          </TabsTrigger>
          <TabsTrigger value="timeline" className="rounded-md px-4 py-2 text-sm font-medium">
            환자 타임라인
          </TabsTrigger>
          <TabsTrigger value="biomarker" className="rounded-md px-4 py-2 text-sm font-medium">
            바이오마커
          </TabsTrigger>
        </TabsList>

        {/* Survival Tab */}
        <TabsContent value="survival" className="space-y-6">
          <ChartCard
            title="코호트 생존 곡선"
            description="시험군 코호트 15명 환자의 전체 생존율 추이"
            delay={400}
          >
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={kmData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                <defs>
                  <linearGradient id="survivalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartColors.indigo} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={chartColors.indigo} stopOpacity={0.05}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.90 0.01 250)" />
                <XAxis 
                  dataKey="month" 
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                  label={{ value: '시간 (개월)', position: 'bottom', offset: 20 }}
                />
                <YAxis 
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                  label={{ value: '생존율 (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid oklch(0.90 0.01 250)',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                  formatter={(value: number, name: string) => {
                    if (name === 'survival') return [`${value}%`, '생존율'];
                    return [value, 'At Risk'];
                  }}
                />
                <ReferenceLine y={50} stroke={chartColors.navy} strokeDasharray="5 5" label={{ value: '중앙값', position: 'right', fontSize: 11 }} />
                <Area 
                  type="stepAfter" 
                  dataKey="survival" 
                  name="survival"
                  stroke={chartColors.indigo}
                  strokeWidth={2.5}
                  fill="url(#survivalGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
            
            {/* At Risk Table */}
            <div className="mt-6 pt-4 border-t border-border/50">
              <p className="text-sm font-medium text-muted-foreground mb-2">Number at Risk</p>
              <div className="flex gap-4 overflow-x-auto">
                {kmData.filter((_, i) => i % 2 === 0).map((point) => (
                  <div key={point.month} className="text-center min-w-[60px]">
                    <p className="text-xs text-muted-foreground">{point.month}mo</p>
                    <p className="text-sm font-mono font-medium">{point.atRisk}</p>
                  </div>
                ))}
              </div>
            </div>
          </ChartCard>

          {/* Individual Patient Survival Table */}
          <div className="bg-card rounded-xl border border-border/50 shadow-sm overflow-hidden">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-lg font-semibold text-foreground">
                개별 환자 생존 데이터
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      환자 ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      OS (개월)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      PFS (개월)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      상태
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Biomarker
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientSurvivalData.map((row) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        {row.patient}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.os}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.pfs}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.status === 'alive' 
                            ? 'bg-[oklch(0.90_0.12_145)] text-[oklch(0.35_0.18_145)]' 
                            : 'bg-[oklch(0.90_0.12_25)] text-[oklch(0.45_0.18_25)]'
                        }`}>
                          {row.status === 'alive' ? '생존' : '사망'}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.biomarker === '+' 
                            ? 'bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]' 
                            : 'bg-muted text-muted-foreground'
                        }`}>
                          {row.biomarker}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        {/* Timeline Tab */}
        <TabsContent value="timeline" className="space-y-6">
          <ChartCard
            title="개별 환자 치료 타임라인"
            description="시험군 코호트 각 환자의 주요 이벤트 타임라인 (개월)"
            delay={400}
          >
            <ResponsiveContainer width="100%" height={500}>
              <BarChart 
                data={patientTimelineData} 
                layout="vertical" 
                margin={{ top: 20, right: 80, left: 50, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.90 0.01 250)" />
                <XAxis 
                  type="number"
                  domain={[0, 35]}
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                  label={{ value: '시간 (개월)', position: 'bottom', offset: 0 }}
                />
                <YAxis 
                  type="category"
                  dataKey="patient"
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid oklch(0.90 0.01 250)',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                  formatter={(value: number, name: string) => [`${value} 개월`, '추적 기간']}
                />
                <Bar dataKey="lastFollowup" radius={[0, 4, 4, 0]}>
                  {patientTimelineData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.death ? chartColors.rose : entry.progression ? chartColors.amber : chartColors.teal}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            
            {/* Timeline Legend */}
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.teal }} />
                <span className="text-sm text-muted-foreground">진행 없음</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.amber }} />
                <span className="text-sm text-muted-foreground">질병 진행</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.rose }} />
                <span className="text-sm text-muted-foreground">사망</span>
              </div>
            </div>
          </ChartCard>
        </TabsContent>

        {/* Biomarker Tab */}
        <TabsContent value="biomarker" className="space-y-6">
          <ChartCard
            title="바이오마커 양성/음성 환자 비교"
            description="바이오마커 상태에 따른 개별 환자 생존 데이터"
            delay={400}
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      환자 ID
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Biomarker
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      PD-L1 (%)
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      OS (개월)
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      PFS (개월)
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      반응
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {biomarkerSubgroupData.map((row) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        {row.patient}
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.biomarker === '+' 
                            ? 'bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]' 
                            : 'bg-muted text-muted-foreground'
                        }`}>
                          {row.biomarker}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-12 h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className="h-full rounded-full"
                              style={{ 
                                width: `${row.pdl1}%`,
                                backgroundColor: row.pdl1 >= 50 ? chartColors.teal : row.pdl1 >= 25 ? chartColors.amber : chartColors.rose
                              }}
                            />
                          </div>
                          <span className="text-sm font-mono text-muted-foreground w-8">
                            {row.pdl1}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-mono text-muted-foreground">
                        {row.os}
                      </td>
                      <td className="px-6 py-4 text-center text-sm font-mono text-muted-foreground">
                        {row.pfs}
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span 
                          className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold"
                          style={{ 
                            backgroundColor: `${responseColors[row.response]}20`,
                            color: responseColors[row.response]
                          }}
                        >
                          {row.response}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Subgroup Summary */}
            <div className="grid grid-cols-2 gap-4 mt-6 pt-4 border-t border-border/50">
              <div className="bg-[oklch(0.95_0.03_180)] rounded-lg p-4">
                <p className="text-sm font-medium text-[oklch(0.35_0.15_180)]">Biomarker+ (n=7)</p>
                <p className="text-lg font-bold font-mono text-foreground mt-1">중앙 OS: 26.3mo</p>
                <p className="text-sm text-muted-foreground">중앙 PFS: 14.6mo</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="text-sm font-medium text-muted-foreground">Biomarker- (n=8)</p>
                <p className="text-lg font-bold font-mono text-foreground mt-1">중앙 OS: 18.5mo</p>
                <p className="text-sm text-muted-foreground">중앙 PFS: 8.3mo</p>
              </div>
            </div>
          </ChartCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}
