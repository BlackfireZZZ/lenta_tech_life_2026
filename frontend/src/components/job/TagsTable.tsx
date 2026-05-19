import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import type { JobPredictions, TagPrediction } from "@/api/jobs";
import { colorMeta, fieldState, tagLabel } from "@/lib/tags";
import { cn } from "@/lib/utils";
import { TAG_SCHEMA_COPY } from "@/content/tagSchemaContent";

// A quick scannable list of every recognized price tag: name, both prices,
// barcode. Click a row to open it in the reviewer above. No internal
// metrics or timecodes — those mean nothing to a store reviewer.
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
          <TH>{TAG_SCHEMA_COPY.tagsTable.name}</TH>
          <TH className="w-28">{TAG_SCHEMA_COPY.tagsTable.noCard}</TH>
          <TH className="w-28">{TAG_SCHEMA_COPY.tagsTable.byCard}</TH>
          <TH className="w-44">{TAG_SCHEMA_COPY.tagsTable.barcode}</TH>
        </tr>
      </THead>
      <TBody>
        {data.tags.map((tag, i) => (
          <Row
            key={tag.index}
            tag={tag}
            ordinal={i + 1}
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
  ordinal,
  selected,
  onSelect,
}: {
  tag: TagPrediction;
  ordinal: number;
  selected: boolean;
  onSelect: (index: number) => void;
}) {
  const m = colorMeta(tag.color);
  const bc = tag.fields.barcode ?? "";
  const partial = fieldState(tag.fields.barcode) === "value" && bc.length < 13;

  return (
    <TR
      selected={selected}
      onClick={() => onSelect(tag.index)}
      className="cursor-pointer hover:bg-canvas-fog"
    >
      <TD className="text-ash-gray tabular-nums">{ordinal}</TD>
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
      <TD className="tabular-nums">
        {fieldState(tag.fields.price_card) === "value" ? (
          `${tag.fields.price_card} ₽`
        ) : (
          <span className="text-steel-gray">—</span>
        )}
      </TD>
      <TD className="font-mono text-[12px] tabular-nums">
        {fieldState(tag.fields.barcode) === "value" ? (
          <span className={partial ? "text-amber-600" : "text-slate-text"}>
            {bc}
            {partial && (
              <span className="ml-1 text-[11px]">{TAG_SCHEMA_COPY.tagsTable.partial}</span>
            )}
          </span>
        ) : (
          <span className="text-steel-gray">{TAG_SCHEMA_COPY.tagsTable.notRecognized}</span>
        )}
      </TD>
    </TR>
  );
}
