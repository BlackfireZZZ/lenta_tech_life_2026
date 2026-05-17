import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "@/components/AppLayout";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/toaster";
import { Spinner } from "@/components/ui/spinner";

// Pages by feature, lazily loaded (architecture.md §4.6). No auth layer —
// this flow is anonymous (architecture.md §3.8).
const UploadPage = lazy(() => import("@/pages/UploadPage"));
const JobPage = lazy(() => import("@/pages/JobPage"));

function PageFallback() {
  return (
    <div className="grid place-items-center py-32 text-ash-gray">
      <Spinner className="size-6" />
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AppLayout>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/jobs/:id" element={<JobPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </AppLayout>
        <Toaster />
      </BrowserRouter>
    </ErrorBoundary>
  );
}
