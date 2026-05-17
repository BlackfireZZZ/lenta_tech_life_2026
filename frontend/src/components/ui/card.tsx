import { cn } from "@/lib/utils";

// Dashboard Card: Cloud White, 10px radius, soft 4/16 shadow, 24px padding.
// `feature` lifts it to the 16px radius + deep-diffused shadow for the one
// or two blocks that should read as prominent (DESIGN "Elevation").
export function Card({
  className,
  feature,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { feature?: boolean }) {
  return (
    <div
      className={cn(
        "bg-card border border-stone-border",
        feature
          ? "rounded-feature shadow-feature"
          : "rounded-card shadow-card",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex flex-col gap-1 p-6 pb-3", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("font-display text-heading-sm font-medium text-slate-text", className)}
      {...props}
    />
  );
}

export function CardDescription({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-caption text-ash-gray", className)} {...props} />;
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-6 pt-3", className)} {...props} />;
}
