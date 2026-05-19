import { forwardRef } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Design: primary = Chartwell-Blue filled pill; ghost = transparent w/ stone
// border; subtle = ash-tinted; outline = dark text, input radius. Compact
// vertical padding, pill radius for the main shapes (DESIGN "Do's").
const buttonVariants = cva(
  "inline-flex cursor-pointer items-center justify-center gap-2 whitespace-nowrap font-medium transition-colors disabled:cursor-not-allowed disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-chartwell-blue/40 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary:
          "rounded-pill bg-chartwell-blue text-primary-foreground shadow-subtle hover:bg-chartwell-blue/90",
        ghost:
          "rounded-pill border border-stone-border bg-transparent text-ash-gray hover:text-slate-text hover:border-platinum-outline",
        subtle:
          "rounded-input border border-stone-border bg-ash-gray/10 text-slate-text hover:bg-ash-gray/15",
        outline:
          "rounded-input border border-stone-border bg-transparent text-slate-text hover:bg-canvas-fog",
        link: "text-chartwell-blue underline-offset-4 hover:underline",
      },
      size: {
        sm: "h-8 px-3 text-[13px]",
        md: "h-10 px-5 text-[14px]",
        lg: "h-12 px-7 text-[15px]",
        icon: "size-9 rounded-pill",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";
