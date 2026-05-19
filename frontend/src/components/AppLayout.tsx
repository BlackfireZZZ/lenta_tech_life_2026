import { Link, NavLink } from "react-router-dom";
import { ScanBarcode } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Загрузить", end: true },
  { to: "/jobs", label: "Задачи", end: false },
  { to: "/pipeline", label: "Пайплайн", end: false },
  { to: "/experiments", label: "Эксперименты", end: false },
  { to: "/shelf", label: "Аудит полки", end: false },
];

// DESIGN "Layout": the nav bar is full-bleed across the top; the main
// content is max-width contained and centred over the Canvas Fog page.
export function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <header className="w-full border-b border-stone-border bg-cloud-white/80 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="grid size-7 place-items-center rounded-input bg-chartwell-blue text-cloud-white">
              <ScanBarcode className="size-4" />
            </span>
            <span className="font-display text-[15px] font-medium text-slate-text">
              Lenta · Распознавание ценников
            </span>
          </Link>
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "rounded-pill px-3 py-1.5 text-caption font-medium transition-colors",
                    isActive
                      ? "bg-sky-tint/40 text-slate-text"
                      : "text-ash-gray hover:text-slate-text",
                  )
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-12">{children}</main>
    </div>
  );
}
