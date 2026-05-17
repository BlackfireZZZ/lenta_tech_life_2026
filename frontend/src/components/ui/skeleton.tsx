import { cn } from "@/lib/utils";

/** Loading placeholder block — calm pulse on a stone tint. */
export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-input bg-stone-border/70", className)}
      {...props}
    />
  );
}
