import { Link } from "react-router-dom";
import { ScanBarcode } from "lucide-react";

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
          <span className="hidden text-caption text-ash-gray sm:block">
            Полка под контролем · Tech Life 2026
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-12">{children}</main>
    </div>
  );
}
