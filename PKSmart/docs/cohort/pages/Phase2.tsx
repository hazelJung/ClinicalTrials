/*
 * Design Philosophy: Pharmaceutical Precision
 * - Phase 2 clinical trial - SINGLE COHORT (300mg RP2D) individual patient data
 * - Individual patient tumor response, treatment duration
 * - Amber accent color scheme
 */

import PageHeader from "@/components/PageHeader";
import StatCard from "@/components/StatCard";
import ChartCard from "@/components/ChartCard";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

// Individual patient tumor response data (Waterfall Plot)
const patientResponseData = [
  { patient: "P01", change: -72, bestResponse: "PR", biomarker: "KRAS WT" },
  { patient: "P02", change: -58, bestResponse: "PR", biomarker: "KRAS WT" },
  { patient: "P03", change: -45, bestResponse: "PR", biomarker: "EGFR+" },
  { patient: "P04", change: -38, bestResponse: "PR", biomarker: "KRAS WT" },
  { patient: "P05", change: -32, bestResponse: "PR", biomarker: "EGFR+" },
  { patient: "P06", change: -25, bestResponse: "SD", biomarker: "KRAS Mut" },
  { patient: "P07", change: -18, bestResponse: "SD", biomarker: "KRAS WT" },
  { patient: "P08", change: -8, bestResponse: "SD", biomarker: "KRAS Mut" },
  { patient: "P09", change: 5, bestResponse: "SD", biomarker: "EGFR-" },
  { patient: "P10", change: 12, bestResponse: "SD", biomarker: "KRAS Mut" },
  { patient: "P11", change: 25, bestResponse: "PD", biomarker: "KRAS Mut" },
  { patient: "P12", change: 38, bestResponse: "PD", biomarker: "EGFR-" },
];

// Individual patient treatment duration (Swimmer Plot data)
const patientDurationData = [
  { patient: "P01", duration: 18, response: "PR", ongoing: true, events: [{ week: 8, type: "PR" }] },
  { patient: "P02", duration: 16, response: "PR", ongoing: true, events: [{ week: 6, type: "PR" }] },
  { patient: "P03", duration: 15, response: "PR", ongoing: false, events: [{ week: 8, type: "PR" }, { week: 15, type: "PD" }] },
  { patient: "P04", duration: 14, response: "PR", ongoing: true, events: [{ week: 10, type: "PR" }] },
  { patient: "P05", duration: 13, response: "PR", ongoing: true, events: [{ week: 6, type: "PR" }] },
  { patient: "P06", duration: 12, response: "SD", ongoing: false, events: [{ week: 12, type: "PD" }] },
  { patient: "P07", duration: 11, response: "SD", ongoing: false, events: [{ week: 11, type: "PD" }] },
  { patient: "P08", duration: 9, response: "SD", ongoing: false, events: [{ week: 9, type: "PD" }] },
  { patient: "P09", duration: 8, response: "SD", ongoing: false, events: [{ week: 8, type: "PD" }] },
  { patient: "P10", duration: 6, response: "SD", ongoing: false, events: [{ week: 6, type: "PD" }] },
  { patient: "P11", duration: 4, response: "PD", ongoing: false, events: [{ week: 4, type: "PD" }] },
  { patient: "P12", duration: 3, response: "PD", ongoing: false, events: [{ week: 3, type: "PD" }] },
];

// Individual patient biomarker data
const patientBiomarkerData = [
  { patient: "P01", kras: "WT", egfr: "+", pdl1: 85, response: "PR" },
  { patient: "P02", kras: "WT", egfr: "+", pdl1: 72, response: "PR" },
  { patient: "P03", kras: "WT", egfr: "+", pdl1: 68, response: "PR" },
  { patient: "P04", kras: "WT", egfr: "-", pdl1: 45, response: "PR" },
  { patient: "P05", kras: "WT", egfr: "+", pdl1: 62, response: "PR" },
  { patient: "P06", kras: "Mut", egfr: "-", pdl1: 35, response: "SD" },
  { patient: "P07", kras: "WT", egfr: "-", pdl1: 28, response: "SD" },
  { patient: "P08", kras: "Mut", egfr: "-", pdl1: 22, response: "SD" },
  { patient: "P09", kras: "WT", egfr: "-", pdl1: 15, response: "SD" },
  { patient: "P10", kras: "Mut", egfr: "-", pdl1: 18, response: "SD" },
  { patient: "P11", kras: "Mut", egfr: "-", pdl1: 8, response: "PD" },
  { patient: "P12", kras: "Mut", egfr: "-", pdl1: 5, response: "PD" },
];

const chartColors = {
  teal: "oklch(0.65 0.15 180)",
  amber: "oklch(0.75 0.15 75)",
  indigo: "oklch(0.55 0.18 220)",
  rose: "oklch(0.60 0.18 25)",
  green: "oklch(0.65 0.18 145)",
};

const responseColors: Record<string, string> = {
  CR: chartColors.green,
  PR: chartColors.teal,
  SD: chartColors.amber,
  PD: chartColors.rose,
};

export default function Phase2() {
  // Calculate cohort summary stats
  const responders = patientResponseData.filter(p => p.change <= -30).length;
  const orr = ((responders / patientResponseData.length) * 100).toFixed(1);
  const dcr = ((patientResponseData.filter(p => p.change < 20).length / patientResponseData.length) * 100).toFixed(1);
  const avgDuration = (patientDurationData.reduce((sum, p) => sum + p.duration, 0) / patientDurationData.length).toFixed(1);

  return (
    <div className="space-y-8">
      <PageHeader
        title="임상 2상 - 300mg 코호트"
        description="300mg RP2D 코호트의 개별 환자 유효성 데이터"
        phase="Phase 2 | Cohort 300mg (RP2D) | N=12"
        phaseColor="amber"
      />

      {/* Cohort Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="ORR"
          value={`${orr}%`}
          description={`${responders}/12 환자 반응`}
          accentColor="teal"
          delay={0}
        />
        <StatCard
          label="DCR"
          value={`${dcr}%`}
          description="질병 통제율"
          accentColor="amber"
          delay={100}
        />
        <StatCard
          label="평균 치료 기간"
          value={avgDuration}
          unit="mo"
          description="코호트 평균"
          accentColor="indigo"
          delay={200}
        />
        <StatCard
          label="진행 중"
          value={`${patientDurationData.filter(p => p.ongoing).length}/12`}
          description="치료 지속 환자"
          accentColor="teal"
          delay={300}
        />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="response" className="space-y-6">
        <TabsList className="bg-muted/50 p-1 rounded-lg">
          <TabsTrigger value="response" className="rounded-md px-4 py-2 text-sm font-medium">
            종양 반응
          </TabsTrigger>
          <TabsTrigger value="duration" className="rounded-md px-4 py-2 text-sm font-medium">
            치료 기간
          </TabsTrigger>
          <TabsTrigger value="biomarker" className="rounded-md px-4 py-2 text-sm font-medium">
            바이오마커
          </TabsTrigger>
        </TabsList>

        {/* Response Tab */}
        <TabsContent value="response" className="space-y-6">
          <ChartCard
            title="개별 환자 종양 반응 (Waterfall Plot)"
            description="300mg 코호트 12명 환자의 최대 종양 크기 변화율 (%)"
            delay={400}
          >
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={patientResponseData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.90 0.01 250)" />
                <XAxis 
                  dataKey="patient" 
                  tick={{ fontSize: 11 }}
                  stroke="oklch(0.50 0.02 250)"
                />
                <YAxis 
                  domain={[-80, 50]}
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                  label={{ value: '종양 크기 변화율 (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid oklch(0.90 0.01 250)',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                  formatter={(value: number, name: string, props: any) => [
                    `${value}% (${props.payload.bestResponse})`,
                    '변화율'
                  ]}
                />
                <ReferenceLine y={-30} stroke={chartColors.teal} strokeDasharray="5 5" label={{ value: 'PR 기준 (-30%)', position: 'right', fontSize: 11 }} />
                <ReferenceLine y={20} stroke={chartColors.rose} strokeDasharray="5 5" label={{ value: 'PD 기준 (+20%)', position: 'right', fontSize: 11 }} />
                <Bar dataKey="change" radius={[4, 4, 0, 0]}>
                  {patientResponseData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={responseColors[entry.bestResponse]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            
            {/* Response Legend */}
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.teal }} />
                <span className="text-sm text-muted-foreground">PR (부분 반응)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.amber }} />
                <span className="text-sm text-muted-foreground">SD (안정)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.rose }} />
                <span className="text-sm text-muted-foreground">PD (진행)</span>
              </div>
            </div>
          </ChartCard>

          {/* Individual Response Table */}
          <div className="bg-card rounded-xl border border-border/50 shadow-sm overflow-hidden">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-lg font-semibold text-foreground">
                개별 환자 반응 데이터
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
                      최대 변화율
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      최적 반응
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      바이오마커
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientResponseData.map((row) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        {row.patient}
                      </td>
                      <td className="px-6 py-4 text-sm font-mono">
                        <span className={row.change <= -30 ? "text-[oklch(0.55_0.18_180)]" : row.change >= 20 ? "text-[oklch(0.55_0.18_25)]" : "text-muted-foreground"}>
                          {row.change > 0 ? '+' : ''}{row.change}%
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span 
                          className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold"
                          style={{ 
                            backgroundColor: `${responseColors[row.bestResponse]}20`,
                            color: responseColors[row.bestResponse]
                          }}
                        >
                          {row.bestResponse}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground">
                        {row.biomarker}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        {/* Duration Tab */}
        <TabsContent value="duration" className="space-y-6">
          <ChartCard
            title="개별 환자 치료 기간 (Swimmer Plot)"
            description="300mg 코호트 각 환자의 치료 지속 기간 (개월)"
            delay={400}
          >
            <ResponsiveContainer width="100%" height={450}>
              <BarChart 
                data={patientDurationData} 
                layout="vertical" 
                margin={{ top: 20, right: 80, left: 50, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.90 0.01 250)" />
                <XAxis 
                  type="number"
                  domain={[0, 20]}
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                  label={{ value: '기간 (개월)', position: 'bottom', offset: 0 }}
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
                  formatter={(value: number, name: string, props: any) => [
                    `${value} 개월 ${props.payload.ongoing ? '(진행 중)' : ''}`,
                    '치료 기간'
                  ]}
                />
                <Bar dataKey="duration" radius={[0, 4, 4, 0]}>
                  {patientDurationData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={responseColors[entry.response]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            
            {/* Duration Legend */}
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.teal }} />
                <span className="text-sm text-muted-foreground">PR (부분 반응)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.amber }} />
                <span className="text-sm text-muted-foreground">SD (안정)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded" style={{ backgroundColor: chartColors.rose }} />
                <span className="text-sm text-muted-foreground">PD (진행)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">▶ 진행 중</span>
              </div>
            </div>
          </ChartCard>
        </TabsContent>

        {/* Biomarker Tab */}
        <TabsContent value="biomarker" className="space-y-6">
          <ChartCard
            title="개별 환자 바이오마커 프로파일"
            description="300mg 코호트 각 환자의 바이오마커 상태 및 반응"
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
                      KRAS
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      EGFR
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      PD-L1 (%)
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      반응
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientBiomarkerData.map((row) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        {row.patient}
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.kras === 'WT' 
                            ? 'bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]' 
                            : 'bg-[oklch(0.90_0.12_25)] text-[oklch(0.45_0.18_25)]'
                        }`}>
                          {row.kras}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.egfr === '+' 
                            ? 'bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]' 
                            : 'bg-muted text-muted-foreground'
                        }`}>
                          {row.egfr}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className="h-full rounded-full"
                              style={{ 
                                width: `${row.pdl1}%`,
                                backgroundColor: row.pdl1 >= 50 ? chartColors.teal : row.pdl1 >= 25 ? chartColors.amber : chartColors.rose
                              }}
                            />
                          </div>
                          <span className="text-sm font-mono text-muted-foreground w-10">
                            {row.pdl1}%
                          </span>
                        </div>
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
          </ChartCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}
