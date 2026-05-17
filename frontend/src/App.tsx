// PLACEHOLDER app shell. The real router (ThemeProvider → AuthProvider →
// layout routes, lazy pages) is specified in docs/architecture.md §4.6.
// For now we render a single placeholder page so the stack boots.
import Home from "./pages/Home";

export default function App() {
  return <Home />;
}
