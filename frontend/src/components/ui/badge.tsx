import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Pill chip (radius 9999px). The design stays monochromatic — only Chartwell
// Blue is saturated; the tag-colour and status hues are kept light/tinted so
// they read as data, not decoration (DESIGN "Don't: extra saturated colors").
const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-pill border px-2.5 py-0.5 text-[12px] font-medium leading-none",
  {
    variants: {
      variant: {
        neutral: "border-stone-border bg-canvas-fog text-ash-gray",
        info: "border-sky-tint bg-sky-tint/40 text-slate-text",
        accent: "border-chartwell-blue/30 bg-chartwell-blue/10 text-slate-text",
        success: "border-emerald-200 bg-emerald-50 text-emerald-700",
        warning: "border-amber-200 bg-amber-50 text-amber-700",
        danger: "border-red-200 bg-red-50 text-red-700",
      },
    },
    defaultVariants: { variant: "neutral" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
