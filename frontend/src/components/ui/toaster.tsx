import { Toaster as Sonner } from "sonner";

/** App-wide toast surface, themed to the light "Crisp Data Canvas" palette. */
export function Toaster() {
  return (
    <Sonner
      position="top-center"
      toastOptions={{
        style: {
          background: "var(--color-cloud-white)",
          color: "var(--color-slate-text)",
          border: "1px solid var(--color-stone-border)",
          borderRadius: "10px",
          boxShadow: "var(--shadow-card)",
          fontFamily: "var(--font-sans)",
        },
      }}
    />
  );
}
