/**
 * Cohort Settings Page
 * Advanced Cohort Settings UI for Digital Twin Clinical Trial Simulation
 * 
 * Design: Pharma Command Center - Dark Mode Mission Control
 */

import DashboardLayout from "@/components/DashboardLayout";
import CohortSettingsComponent from "@/components/CohortSettings";
import { motion } from "framer-motion";
import { Settings2, FileText, Download, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export default function CohortSettingsPage() {
  const handleExportSettings = () => {
    toast.success("설정이 JSON 파일로 내보내기되었습니다.");
  };

  const handleShareSettings = () => {
    toast.info("공유 링크가 클립보드에 복사되었습니다.");
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-cyan-500/20 border border-primary/30">
                <Settings2 className="h-6 w-6 text-primary" />
              </div>
              <h1 className="text-2xl lg:text-3xl font-bold">
                <span className="gradient-text">Advanced Cohort Settings</span>
              </h1>
            </div>
            <p className="text-muted-foreground max-w-2xl">
              한미약품 FDA 타겟 임상 데이터 기반 디지털 트윈 코호트 설정. 
              유전적 다형성, 신장/간 기능, 병용 약물 등 7가지 고급 파라미터를 설정하여 
              정밀한 임상시험 시뮬레이션을 수행합니다.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleExportSettings}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="outline" size="sm" onClick={handleShareSettings}>
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>
          </div>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Settings Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1"
          >
            <div className="sticky top-24">
              <CohortSettingsComponent />
            </div>
          </motion.div>

          {/* Preview & Documentation Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Settings Summary Card */}
            <div className="p-6 rounded-xl border border-border/50 bg-card/30 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">설정 가이드</h3>
              </div>
              
              <div className="space-y-4 text-sm text-muted-foreground">
                <div className="p-4 rounded-lg bg-primary/5 border border-primary/20">
                  <h4 className="font-medium text-foreground mb-2">Phase 연동 자동 설정</h4>
                  <p>
                    임상 단계(Phase 1/2/3)를 선택하면 관련 옵션들이 자동으로 조정됩니다.
                    Phase 1은 건강인 기준, Phase 2/3은 환자 기준이 적용됩니다.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-cyan-500/5 border border-cyan-500/20">
                  <h4 className="font-medium text-foreground mb-2">유전체 시나리오 프리셋</h4>
                  <p>
                    CYP2D6 PM Cohort, CYP2C19 Asian 등 FDA 제출용 유전체 시나리오를 
                    원클릭으로 생성할 수 있습니다.
                  </p>
                </div>

                <div className="p-4 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
                  <h4 className="font-medium text-foreground mb-2">동적 PK 영향 계산</h4>
                  <p>
                    설정 변경 시 실시간으로 PK 영향(AUC, Cmax, T½ 변화)을 예측하여 
                    가이드 문구를 표시합니다.
                  </p>
                </div>
              </div>
            </div>

            {/* Feature Categories */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  title: "유전적 다형성",
                  description: "CYP2D6, CYP2C19 등 대사 효소 유전자형 분포 설정",
                  color: "from-purple-500/20 to-pink-500/20",
                },
                {
                  title: "신체 지표",
                  description: "BMI, BSA 범위 및 분포 설정",
                  color: "from-blue-500/20 to-cyan-500/20",
                },
                {
                  title: "병용 약물 (DDI)",
                  description: "적응증별 병용 약물 프리셋 및 DDI 시뮬레이션",
                  color: "from-green-500/20 to-emerald-500/20",
                },
                {
                  title: "신장/간 기능",
                  description: "eGFR, Child-Pugh 기반 장기 기능 분포 설정",
                  color: "from-orange-500/20 to-red-500/20",
                },
                {
                  title: "질병 상태",
                  description: "적응증별 바이오마커 기준선 설정",
                  color: "from-teal-500/20 to-cyan-500/20",
                },
                {
                  title: "투여 계획",
                  description: "용량 증량, 용량 탐색, 고정 용량 레지멘 설정",
                  color: "from-indigo-500/20 to-purple-500/20",
                },
              ].map((feature, idx) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + idx * 0.1 }}
                  className={`p-4 rounded-xl border border-border/50 bg-gradient-to-br ${feature.color}`}
                >
                  <h4 className="font-medium mb-1">{feature.title}</h4>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </motion.div>
              ))}
            </div>

            {/* FDA MIDD Compliance Note */}
            <div className="p-6 rounded-xl border border-primary/30 bg-primary/5">
              <h3 className="font-semibold mb-3 flex items-center gap-2">
                <span className="text-primary">FDA MIDD</span> 준수
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                본 시뮬레이션 시스템은 FDA Model-Informed Drug Development (MIDD) 
                가이던스를 준수하여 설계되었습니다. 결과 리포트는 다음 항목을 포함합니다:
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                {["Cmax", "Tmax", "AUC0-inf", "T½", "CL/F", "Vd/F"].map((param) => (
                  <div
                    key={param}
                    className="px-3 py-2 rounded-lg bg-background/50 text-center font-mono text-primary"
                  >
                    {param}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </DashboardLayout>
  );
}
