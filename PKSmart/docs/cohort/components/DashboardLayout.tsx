/*
 * Design Philosophy: Pharmaceutical Precision
 * - Narrow icon-based sidebar with navy theme
 * - Clean content area with molecular grid pattern
 * - Smooth transitions and hover effects
 */

import { Link, useLocation } from "wouter";
import { 
  LayoutDashboard, 
  Beaker, 
  FlaskConical, 
  Award,
  Dna,
  Menu,
  X
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: "/", label: "개요", icon: LayoutDashboard },
  { path: "/phase1", label: "임상 1상", icon: Beaker },
  { path: "/phase2", label: "임상 2상", icon: FlaskConical },
  { path: "/phase3", label: "임상 3상", icon: Award },
];

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex bg-background">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex flex-col w-64 bg-sidebar border-r border-sidebar-border">
        {/* Logo */}
        <div className="p-6 border-b border-sidebar-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-sidebar-primary flex items-center justify-center">
              <Dna className="w-6 h-6 text-sidebar-primary-foreground" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-sidebar-foreground leading-tight">
                임상시험 코호트
              </h1>
              <p className="text-xs text-sidebar-foreground/60">
                결과 대시보드
              </p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location === item.path;
            const Icon = item.icon;
            return (
              <Link key={item.path} href={item.path}>
                <div
                  className={cn(
                    "nav-item",
                    isActive && "active"
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-sidebar-border">
          <p className="text-xs text-sidebar-foreground/50 text-center">
            약물유전체 기반 시뮬레이션
          </p>
          <p className="text-xs text-sidebar-foreground/40 text-center mt-1">
            © 2026 Clinical Trial Dashboard
          </p>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-sidebar border-b border-sidebar-border">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-sidebar-primary flex items-center justify-center">
              <Dna className="w-5 h-5 text-sidebar-primary-foreground" />
            </div>
            <span className="text-sm font-bold text-sidebar-foreground">
              임상시험 대시보드
            </span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-sidebar-foreground"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <nav className="p-4 space-y-1 border-t border-sidebar-border">
            {navItems.map((item) => {
              const isActive = location === item.path;
              const Icon = item.icon;
              return (
                <Link key={item.path} href={item.path}>
                  <div
                    className={cn(
                      "nav-item",
                      isActive && "active"
                    )}
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                  </div>
                </Link>
              );
            })}
          </nav>
        )}
      </div>

      {/* Main Content */}
      <main className="flex-1 lg:ml-0 mt-16 lg:mt-0 molecular-bg min-h-screen">
        <div className="p-6 lg:p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
