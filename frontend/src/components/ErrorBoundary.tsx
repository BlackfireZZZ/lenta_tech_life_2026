import { Component, type ReactNode } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ERROR_BOUNDARY_COPY } from "@/content/appContent";

// Root guard so a render error shows a calm card instead of a blank page
// (architecture.md §4.6 — minus the auth-only pieces).
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="grid min-h-screen place-items-center bg-background p-6">
          <Card className="max-w-md p-6 text-center">
            <h1 className="font-display text-heading text-slate-text">
              {ERROR_BOUNDARY_COPY.title}
            </h1>
            <p className="mt-2 text-[14px] text-ash-gray">
              {this.state.error.message || ERROR_BOUNDARY_COPY.fallbackMessage}
            </p>
            <Button className="mt-5" onClick={() => window.location.assign("/")}>
              {ERROR_BOUNDARY_COPY.backHome}
            </Button>
          </Card>
        </div>
      );
    }
    return this.props.children;
  }
}
