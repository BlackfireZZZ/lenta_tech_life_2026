import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Upload } from "lucide-react";
import { jobsApi, type Job } from "@/api/jobs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { JOBS_PAGE_COPY } from "@/content/jobsPageContent";

const POLL_MS = 5000;

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState(false);
  const navigate = useNavigate();

  // Poll so a job that is still processing updates in place. The list is
  // capped server-side, so this stays cheap.
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const data = await jobsApi.list();
        if (!alive) return;
        setJobs(data);
        setError(false);
      } catch {
        if (alive) setError(true);
      } finally {
        if (alive) timer = setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="font-display text-heading-lg font-medium text-slate-text">
            {JOBS_PAGE_COPY.title}
          </h1>
          <p className="mt-2 text-caption text-ash-gray">
            {JOBS_PAGE_COPY.subtitle}
          </p>
        </div>
        <Link to="/">
          <Button size="lg">
            <Upload />
            {JOBS_PAGE_COPY.uploadButton}
          </Button>
        </Link>
      </div>

      {jobs === null && !error ? (
        <Card className="p-2">
          <Skeleton className="h-64 w-full" />
        </Card>
      ) : error && jobs === null ? (
        <Card className="p-10 text-center">
          <p className="text-[14px] text-ash-gray">
            {JOBS_PAGE_COPY.loadFailed}
          </p>
        </Card>
      ) : jobs && jobs.length === 0 ? (
        <Card className="grid place-items-center gap-4 p-12 text-center">
          <p className="text-[15px] text-slate-text">{JOBS_PAGE_COPY.emptyTitle}</p>
          <p className="max-w-sm text-caption text-ash-gray">
            {JOBS_PAGE_COPY.emptyHint}
          </p>
          <Link to="/">
            <Button>
              <Upload />
              {JOBS_PAGE_COPY.uploadButton}
            </Button>
          </Link>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>
              {JOBS_PAGE_COPY.totalPrefix} {jobs?.length ?? 0}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <THead>
                <tr>
                  <TH>{JOBS_PAGE_COPY.columns.video}</TH>
                  <TH className="w-40">{JOBS_PAGE_COPY.columns.status}</TH>
                  <TH className="w-28">{JOBS_PAGE_COPY.columns.tags}</TH>
                  <TH className="w-36">{JOBS_PAGE_COPY.columns.uploaded}</TH>
                </tr>
              </THead>
              <TBody>
                {jobs?.map((j) => {
                  const s = JOBS_PAGE_COPY.status[j.status];
                  return (
                    <TR
                      key={j.id}
                      onClick={() => navigate(`/jobs/${j.id}`)}
                      className="cursor-pointer hover:bg-canvas-fog"
                    >
                      <TD className="max-w-0">
                        <span className="block truncate font-medium text-slate-text">
                          {j.filename || "—"}
                        </span>
                      </TD>
                      <TD>
                        <Badge variant={s.variant}>
                          {s.label}
                          {j.status === "running" &&
                            ` · ${Math.round((j.progress ?? 0) * 100)}%`}
                          {j.status === "queued" &&
                            j.queue_position != null &&
                            (j.queue_position > 0
                              ? ` · перед вами ${j.queue_position}`
                              : " · вы следующий")}
                        </Badge>
                      </TD>
                      <TD className="tabular-nums">
                        {j.rows ?? <span className="text-steel-gray">—</span>}
                      </TD>
                      <TD className="tabular-nums text-ash-gray">
                        {formatWhen(j.created_at)}
                      </TD>
                    </TR>
                  );
                })}
              </TBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
