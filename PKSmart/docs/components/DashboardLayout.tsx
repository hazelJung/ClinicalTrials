import { Link, useLocation } from "wouter";
import { motion } from "framer-motion";
import { 
  FlaskConical, 
  TestTube2, 
  BarChart3, 
  Activity,
  Dna,
  Menu,
  X,
  Settings2
} from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const navItems = [
  { path: "/", label: "개요", icon: Activity },
  { path: "/phase1", label: "Phase 1", icon: FlaskConical, subtitle: "Safety & PK" },
  { path: "/phase2", label: "Phase 2", icon: TestTube2, subtitle: "PoC & Dose" },
  { path: "/phase3", label: "Phase 3", icon: BarChart3, subtitle: "Confirmatory" },
  { path: "/cohort-settings", label: "Cohort", icon: Settings2, subtitle: "Advanced" },
];

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background grid-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="container flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3">
            <div className="relative">
              <Dna className="h-8 w-8 text-primary" />
              <div className="absolute inset-0 blur-lg bg-primary/30" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold tracking-tight">
                <span className="gradient-text">Clinical Trial</span>
              </h1>
              <p className="text-xs text-muted-foreground -mt-1">Simulation Dashboard</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = location === item.path;
              const Icon = item.icon;
              return (
                <Link key={item.path} href={item.path}>
                  <motion.div
                    className={`
                      relative px-4 py-2 rounded-lg flex items-center gap-2 transition-colors
                      ${isActive 
                        ? "bg-primary/10 text-primary" 
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                      }
                    `}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="font-medium">{item.label}</span>
                    {item.subtitle && (
                      <span className="hidden lg:inline text-xs text-muted-foreground">
                        {item.subtitle}
                      </span>
                    )}
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute bottom-0 left-2 right-2 h-0.5 bg-primary rounded-full"
                        initial={false}
                        transition={{ type: "spring", stiffness: 500, damping: 30 }}
                      />
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </nav>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <motion.nav
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="md:hidden border-t border-border/50 bg-background/95 backdrop-blur-xl"
          >
            <div className="container py-4 space-y-2">
              {navItems.map((item) => {
                const isActive = location === item.path;
                const Icon = item.icon;
                return (
                  <Link 
                    key={item.path} 
                    href={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    <div
                      className={`
                        flex items-center gap-3 px-4 py-3 rounded-lg transition-colors
                        ${isActive 
                          ? "bg-primary/10 text-primary" 
                          : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                        }
                      `}
                    >
                      <Icon className="h-5 w-5" />
                      <div>
                        <span className="font-medium">{item.label}</span>
                        {item.subtitle && (
                          <span className="block text-xs text-muted-foreground">
                            {item.subtitle}
                          </span>
                        )}
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </motion.nav>
        )}
      </header>

      {/* Main Content */}
      <main className="container py-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {children}
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 py-6 mt-auto">
        <div className="container flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>FDA 임상시험 시뮬레이션 대시보드 | PBPK & Pharmacogenomics</p>
          <p>© 2026 Clinical Trial Simulation System</p>
        </div>
      </footer>
    </div>
  );
}
