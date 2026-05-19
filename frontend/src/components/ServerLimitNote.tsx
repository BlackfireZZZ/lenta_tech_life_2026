import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Honest, calm explanation of *why* videos wait in line. Shown before
 * upload (sets expectations) and on the queued screen (explains the wait).
 * It is a real product limitation, not a bug — say so plainly.
 */
export function ServerLimitNote({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-card border border-stone-border bg-canvas-fog px-4 py-3.5",
        className,
      )}
    >
      <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-input bg-chartwell-blue/10 text-chartwell-blue">
        <Cpu className="size-4" />
      </span>
      <p className="text-[13px] leading-[1.6] text-ash-gray">
        <span className="font-medium text-slate-text">
          Видео обрабатываются по одному.
        </span>{" "}
        Мы намеренно арендовали недорогой сервер с одной небольшой
        видеокартой — на нём распознавание идёт стабильно, но запускать
        несколько видео сразу нельзя: они начали бы делить одну видеокарту и
        каждое считалось бы заметно дольше. Поэтому новое видео ждёт, пока
        освободится сервер, и только потом запускается.
      </p>
    </div>
  );
}
