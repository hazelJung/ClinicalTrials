/*
 * Design Philosophy: Pharmaceutical Precision
 * - Phase 1 clinical trial - SINGLE COHORT (400mg) individual patient data
 * - Individual patient PK profiles, AE data, PGx data
 * - Teal accent color scheme
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
} from "recharts";

// Individual patient PK data for 400mg cohort (Spaghetti Plot - each line = 1 patient)
const individualPKData = [
  { time: 0, P01: 0, P02: 0, P03: 0, P04: 0, P05: 0, P06: 0 },
  { time: 1, P01: 580, P02: 620, P03: 545, P04: 690, P05: 510, P06: 605 },
  { time: 2, P01: 720, P02: 680, P03: 750, P04: 810, P05: 640, P06: 695 },
  { time: 4, P01: 520, P02: 480, P03: 560, P04: 610, P05: 450, P06: 505 },
  { time: 8, P01: 280, P02: 310, P03: 265, P04: 340, P05: 240, P06: 295 },
  { time: 12, P01: 145, P02: 165, P03: 130, P04: 185, P05: 120, P06: 155 },
  { time: 24, P01: 45, P02: 55, P03: 38, P04: 68, P05: 32, P06: 48 },
  { time: 48, P01: 12, P02: 18, P03: 10, P04: 22, P05: 8, P06: 15 },
];

// Individual patient PK parameters for 400mg cohort
const patientPKParams = [
  { patient: "P01", cmax: 720, tmax: 2, auc: 8450, t12: 17.2, cl: 47.3 },
  { patient: "P02", cmax: 680, tmax: 2, auc: 7820, t12: 18.5, cl: 51.2 },
  { patient: "P03", cmax: 750, tmax: 2, auc: 8120, t12: 16.8, cl: 49.3 },
  { patient: "P04", cmax: 810, tmax: 2, auc: 9280, t12: 19.8, cl: 43.1 },
  { patient: "P05", cmax: 640, tmax: 2, auc: 7150, t12: 15.5, cl: 55.9 },
  { patient: "P06", cmax: 695, tmax: 2, auc: 7680, t12: 17.8, cl: 52.1 },
];

// Individual patient Adverse Events for 400mg cohort
const patientAEData = [
  { patient: "P01", fatigue: 1, nausea: 0, headache: 1, diarrhea: 0, rash: 0, dlt: false },
  { patient: "P02", fatigue: 2, nausea: 1, headache: 0, diarrhea: 1, rash: 0, dlt: false },
  { patient: "P03", fatigue: 1, nausea: 1, headache: 1, diarrhea: 0, rash: 1, dlt: false },
  { patient: "P04", fatigue: 2, nausea: 2, headache: 1, diarrhea: 2, rash: 0, dlt: false },
  { patient: "P05", fatigue: 1, nausea: 0, headache: 0, diarrhea: 0, rash: 0, dlt: false },
  { patient: "P06", fatigue: 3, nausea: 2, headache: 2, diarrhea: 1, rash: 1, dlt: true },
];

// Individual patient PGx data for 400mg cohort
const patientPGxData = [
  { patient: "P01", cyp2d6: "EM", cyp3a4: "*1/*1", auc: 8450, cmax: 720 },
  { patient: "P02", cyp2d6: "IM", cyp3a4: "*1/*1", auc: 7820, cmax: 680 },
  { patient: "P03", cyp2d6: "EM", cyp3a4: "*1/*22", auc: 8120, cmax: 750 },
  { patient: "P04", cyp2d6: "PM", cyp3a4: "*1/*1", auc: 9280, cmax: 810 },
  { patient: "P05", cyp2d6: "EM", cyp3a4: "*1/*1", auc: 7150, cmax: 640 },
  { patient: "P06", cyp2d6: "IM", cyp3a4: "*1/*22", auc: 7680, cmax: 695 },
];

const chartColors = {
  teal: "oklch(0.65 0.15 180)",
  tealLight: "oklch(0.75 0.12 180)",
  indigo: "oklch(0.55 0.18 220)",
  amber: "oklch(0.75 0.15 75)",
  rose: "oklch(0.60 0.18 25)",
};

const patientColors = [
  "oklch(0.65 0.15 180)", // teal
  "oklch(0.55 0.18 220)", // indigo
  "oklch(0.75 0.15 75)",  // amber
  "oklch(0.60 0.18 25)",  // rose
  "oklch(0.55 0.20 145)", // green
  "oklch(0.60 0.15 300)", // purple
];

const gradeColors: Record<number, string> = {
  0: "bg-muted",
  1: "bg-[oklch(0.85_0.12_180)]",
  2: "bg-[oklch(0.75_0.15_75)]",
  3: "bg-[oklch(0.60_0.18_25)]",
};

export default function Phase1() {
  // Calculate cohort summary stats
  const avgCmax = Math.round(patientPKParams.reduce((sum, p) => sum + p.cmax, 0) / patientPKParams.length);
  const avgT12 = (patientPKParams.reduce((sum, p) => sum + p.t12, 0) / patientPKParams.length).toFixed(1);
  const dltCount = patientAEData.filter(p => p.dlt).length;
  
  return (
    <div className="space-y-8">
      <PageHeader
        title="임상 1상 - 400mg 코호트"
        description="400mg 용량 코호트의 개별 환자 안전성 및 약동학 데이터"
        phase="Phase 1 | Cohort 400mg | N=6"
        phaseColor="teal"
      />

      {/* Cohort Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="코호트 용량"
          value={400}
          unit="mg"
          description="1일 1회 경구 투여"
          accentColor="teal"
          delay={0}
        />
        <StatCard
          label="DLT 발생"
          value={`${dltCount}/6`}
          description={`${((dltCount/6)*100).toFixed(1)}% 발생률`}
          accentColor="amber"
          delay={100}
        />
        <StatCard
          label="평균 Cmax"
          value={avgCmax}
          unit="ng/mL"
          description="코호트 평균"
          accentColor="teal"
          delay={200}
        />
        <StatCard
          label="평균 t1/2"
          value={avgT12}
          unit="hr"
          description="코호트 평균 반감기"
          accentColor="indigo"
          delay={300}
        />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="pk" className="space-y-6">
        <TabsList className="bg-muted/50 p-1 rounded-lg">
          <TabsTrigger value="pk" className="rounded-md px-4 py-2 text-sm font-medium">
            약동학(PK)
          </TabsTrigger>
          <TabsTrigger value="safety" className="rounded-md px-4 py-2 text-sm font-medium">
            안전성(AE)
          </TabsTrigger>
          <TabsTrigger value="pgx" className="rounded-md px-4 py-2 text-sm font-medium">
            약물유전체
          </TabsTrigger>
        </TabsList>

        {/* PK Tab */}
        <TabsContent value="pk" className="space-y-6">
          <ChartCard
            title="개별 환자 PK Profile (Spaghetti Plot)"
            description="400mg 코호트 6명 환자의 개별 혈중 농도-시간 곡선"
            delay={400}
          >
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={individualPKData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.90 0.01 250)" />
                <XAxis 
                  dataKey="time" 
                  label={{ value: '시간 (hr)', position: 'bottom', offset: 0 }}
                  tick={{ fontSize: 12 }}
                  stroke="oklch(0.50 0.02 250)"
                />
                <YAxis 
                  label={{ value: '혈중 농도 (ng/mL)', angle: -90, position: 'insideLeft' }}
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
                />
                <Legend />
                {['P01', 'P02', 'P03', 'P04', 'P05', 'P06'].map((patient, index) => (
                  <Line 
                    key={patient}
                    type="monotone" 
                    dataKey={patient} 
                    name={`환자 ${patient}`}
                    stroke={patientColors[index]}
                    strokeWidth={2}
                    dot={{ fill: patientColors[index], strokeWidth: 2, r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Individual PK Parameters Table */}
          <div className="bg-card rounded-xl border border-border/50 shadow-sm overflow-hidden">
            <div className="p-6 border-b border-border/50">
              <h3 className="text-lg font-semibold text-foreground">
                개별 환자 PK 파라미터
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                400mg 코호트 각 환자의 약동학 파라미터
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      환자 ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Cmax (ng/mL)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Tmax (hr)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      AUC (ng·hr/mL)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      t1/2 (hr)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      CL/F (L/hr)
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientPKParams.map((row, index) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        <div className="flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: patientColors[index] }}
                          />
                          {row.patient}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.cmax}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.tmax}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.auc.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.t12}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.cl}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        {/* Safety Tab */}
        <TabsContent value="safety" className="space-y-6">
          <ChartCard
            title="개별 환자 이상반응 프로파일"
            description="400mg 코호트 각 환자의 이상반응 발생 현황 (Grade 0-3)"
            delay={400}
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      환자 ID
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      피로
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      오심
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      두통
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      설사
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      발진
                    </th>
                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      DLT
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientAEData.map((row, index) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-4 py-4 text-sm font-medium text-foreground">
                        <div className="flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: patientColors[index] }}
                          />
                          {row.patient}
                        </div>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-mono font-medium ${gradeColors[row.fatigue]} ${row.fatigue >= 3 ? 'text-white' : 'text-foreground'}`}>
                          {row.fatigue}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-mono font-medium ${gradeColors[row.nausea]} ${row.nausea >= 3 ? 'text-white' : 'text-foreground'}`}>
                          {row.nausea}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-mono font-medium ${gradeColors[row.headache]} ${row.headache >= 3 ? 'text-white' : 'text-foreground'}`}>
                          {row.headache}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-mono font-medium ${gradeColors[row.diarrhea]} ${row.diarrhea >= 3 ? 'text-white' : 'text-foreground'}`}>
                          {row.diarrhea}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={`inline-flex items-center justify-center w-8 h-8 rounded-lg text-sm font-mono font-medium ${gradeColors[row.rash]} ${row.rash >= 3 ? 'text-white' : 'text-foreground'}`}>
                          {row.rash}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        {row.dlt ? (
                          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-[oklch(0.90_0.15_25)] text-[oklch(0.45_0.18_25)]">
                            Yes
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-muted text-muted-foreground">
                            No
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* Grade Legend */}
            <div className="flex justify-center gap-6 mt-6 pt-4 border-t border-border/50">
              <div className="flex items-center gap-2">
                <span className={`w-6 h-6 rounded ${gradeColors[0]}`} />
                <span className="text-sm text-muted-foreground">Grade 0</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-6 h-6 rounded ${gradeColors[1]}`} />
                <span className="text-sm text-muted-foreground">Grade 1</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-6 h-6 rounded ${gradeColors[2]}`} />
                <span className="text-sm text-muted-foreground">Grade 2</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`w-6 h-6 rounded ${gradeColors[3]}`} />
                <span className="text-sm text-muted-foreground">Grade 3 (DLT)</span>
              </div>
            </div>
          </ChartCard>
        </TabsContent>

        {/* PGx Tab */}
        <TabsContent value="pgx" className="space-y-6">
          <ChartCard
            title="개별 환자 약물유전체 프로파일"
            description="400mg 코호트 각 환자의 CYP 유전자형 및 PK 영향"
            delay={400}
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/30">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      환자 ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      CYP2D6 표현형
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      CYP3A4 유전자형
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      AUC (ng·hr/mL)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Cmax (ng/mL)
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {patientPGxData.map((row, index) => (
                    <tr key={row.patient} className="hover:bg-muted/20 transition-colors">
                      <td className="px-6 py-4 text-sm font-medium text-foreground">
                        <div className="flex items-center gap-2">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: patientColors[index] }}
                          />
                          {row.patient}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${
                          row.cyp2d6 === 'EM' ? 'bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]' :
                          row.cyp2d6 === 'IM' ? 'bg-[oklch(0.90_0.12_75)] text-[oklch(0.45_0.15_75)]' :
                          'bg-[oklch(0.90_0.12_25)] text-[oklch(0.45_0.18_25)]'
                        }`}>
                          {row.cyp2d6}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.cyp3a4}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.auc.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground font-mono">
                        {row.cmax}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {/* PGx Legend */}
            <div className="flex justify-center gap-6 mt-6 pt-4 border-t border-border/50">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-[oklch(0.90_0.12_180)] text-[oklch(0.35_0.15_180)]">
                  EM
                </span>
                <span className="text-sm text-muted-foreground">Extensive Metabolizer</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-[oklch(0.90_0.12_75)] text-[oklch(0.45_0.15_75)]">
                  IM
                </span>
                <span className="text-sm text-muted-foreground">Intermediate Metabolizer</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-[oklch(0.90_0.12_25)] text-[oklch(0.45_0.18_25)]">
                  PM
                </span>
                <span className="text-sm text-muted-foreground">Poor Metabolizer</span>
              </div>
            </div>
          </ChartCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}
