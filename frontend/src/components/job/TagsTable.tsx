import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import type { JobPredictions, TagPrediction } from "@/api/jobs";
import {
  colorMeta,
  completeness,
  fieldState,
  PASS_THRESHOLD,
  tagLabel,
} from "@/lib/tags";
import { cn, formatTimestamp } from "@/lib/utils";

// One row per unique tag (the CSV granularity). Clicking a row selects it,
// seeks the video and redraws the crop.
export function TagsTable({
  data,
  selected,
  onSelect,
}: {
  data: JobPredictions;
  selected: number;
  onSelect: (index: number) => void;
}) {
  return (
    <Table>
      <THead>
        <tr>
          <TH className="w-10">#</TH>
          <TH>Ценник</TH>
          <TH className="w-28">Без карты</TH>
          <TH className="w-44">Штрихкод</TH>
          <TH className="w-24">Таймкод</TH>
          <TH className="w-28">Полнота</TH>
        </tr>
      </THead>
      <TBody>
        {data.tags.map((tag) => (
          <Row
            key={tag.index}
            tag={tag}
            substantive={data.substantive_fields}
            selected={tag.index === selected}
            onSelect={onSelect}
          />
        ))}
      </TBody>
    </Table>
  );
}

function Row({
  tag,
  substantive,
  selected,
  onSelect,
}: {
  tag: TagPrediction;
  substantive: string[];
  selected: boolean;
  onSelect: (index: number) => void;
}) {
  const m = colorMeta(tag.color);
  const score = completeness(tag, substantive);
  const bc = tag.fields.barcode ?? "";
  const partial = fieldState(tag.fields.barcode) === "value" && bc.length < 13;

  return (
    <TR
      selected={selected}
      onClick={() => onSelect(tag.index)}
      className="cursor-pointer hover:bg-canvas-fog"
    >
      <TD className="text-ash-gray tabular-nums">{tag.index + 1}</TD>
      <TD>
        <div className="flex items-center gap-2.5">
          <span
            className="size-3 shrink-0 rounded-full"
            style={{ background: m.swatch, boxShadow: `0 0 0 1px ${m.ring}` }}
            title={m.label}
          />
          <span
            className={cn(
              "truncate",
              fieldState(tag.fields.product_name) === "value"
                ? "text-slate-text"
                : "italic text-steel-gray",
            )}
          >
            {tagLabel(tag)}
          </span>
        </div>
      </TD>
      <TD className="tabular-nums">
        {fieldState(tag.fields.price_default) === "value" ? (
          `${tag.fields.price_default} ₽`
        ) : (
          <span className="text-steel-gray">—</span>
        )}
      </TD>
      <TD className="font-mono text-[12px] tabular-nums">
        {fieldState(tag.fields.barcode) === "value" ? (
          <span className={partial ? "text-amber-600" : "text-slate-text"}>
            {bc}
            {partial && <span className="ml-1 text-[11px]">(частичный)</span>}
          </span>
        ) : (
          <span className="text-steel-gray">не распознан</span>
        )}
      </TD>
      <TD className="tabular-nums text-ash-gray">{formatTimestamp(tag.frame_timestamp)}</TD>
      <TD>
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-12 overflow-hidden rounded-pill bg-stone-border">
            <div
              className={cn(
                "h-full rounded-pill",
                score >= PASS_THRESHOLD ? "bg-chartwell-blue" : "bg-amber-400",
              )}
              style={{ width: `${Math.round(score * 100)}%` }}
            />
          </div>
          <span className="tabular-nums text-[12px] text-ash-gray">
            {Math.round(score * 100)}%
          </span>
        </div>
      </TD>
    </TR>
  );
}
