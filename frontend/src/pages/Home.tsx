// PLACEHOLDER page. Target UX: upload a shelf video → poll job status →
// download the 29-column CSV. Full flow & API wiring: docs/architecture.md §4.
export default function Home() {
  return (
    <main className="screen">
      <section className="card">
        <p className="kicker">Lenta Tech Life 2026</p>
        <h1>Price-Tag Recognition</h1>
        <p className="lede">
          Upload a shelf video, get one CSV row per price tag.
        </p>
        <div className="stub" role="status">
          🚧 Frontend placeholder — UI is built separately.
          <br />
          The upload&nbsp;→&nbsp;process&nbsp;→&nbsp;download flow lands here.
        </div>
        <p className="hint">
          API: <code>POST /api/v1/jobs</code> ·{" "}
          <code>GET /api/v1/jobs/&#123;id&#125;</code>
        </p>
      </section>
    </main>
  );
}
