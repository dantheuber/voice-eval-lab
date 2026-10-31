import { useLayoutEffect, useRef, useState } from "react";
import type { TimelineEvent, Utterance } from "@/api/types";
import { fmtClock } from "@/lib/format";
import { Icon } from "@/components/ui";
import s from "./review.module.css";

type LaneEvent = Exclude<TimelineEvent, Utterance>;
const CHAR_PX = 6; // 10 px mono
const LABEL_EXTRA_PX = 30; // icon, padding, border
const GAP_PX = 4;
const MAX_ROWS = 3;
const ROW_PX = 19;

/* Labels are wider than the time between events, and a reviewer needs to read each one, so they stack. Past MAX_ROWS the
   row that frees up first takes it, and the tooltip still has the detail. */
export function layoutEvents(items: { atS: number; label: string }[], durationS: number, width: number) {
  const rowEnds: number[] = [];
  return items.map(({ atS, label }) => {
    const w = label.length * CHAR_PX + LABEL_EXTRA_PX;
    const x = Math.max(0, Math.min((atS / durationS) * width - w / 2, width - w));
    let row = rowEnds.findIndex((end) => x >= end + GAP_PX);
    if (row < 0 && rowEnds.length < MAX_ROWS) { row = rowEnds.length; rowEnds.push(0); }
    if (row < 0) row = rowEnds.indexOf(Math.min(...rowEnds));
    rowEnds[row] = x + w;
    return { x, row };
  });
}

function describe(e: LaneEvent) {
  const kind = e.type === "tool_call" ? "tool" : e.type === "barge_in" ? "barge" : "db";
  const label = e.type === "tool_call" ? e.name : e.type === "barge_in" ? "barge-in" : e.detail.split(" ").slice(0, 2).join(" ");
  const title = e.type === "tool_call" ? `${e.name}(${JSON.stringify(e.arguments)}) -> ${JSON.stringify(e.result)}` : e.type === "barge_in" ? `Caller started speaking at ${fmtClock(e.atS)}${e.botStoppedAtS != null ? `; bot stopped after ${(e.botStoppedAtS - e.atS).toFixed(1)} s` : ""}` : e.detail;
  return { kind, label, title };
}

export function EventsLane({ timeline, durationS, onSeek }: { timeline: TimelineEvent[]; durationS: number; onSeek: (t: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current; if (!el) return;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const events = timeline.filter((e): e is LaneEvent => e.type !== "utterance" && !(e.type === "system" && e.kind !== "db_change")).map((e) => ({ e, ...describe(e), atS: e.atS }));
  const places = layoutEvents(events, durationS, width);
  const rows = Math.max(1, ...places.map((p) => p.row + 1));
  return (
    <div className={s.events} ref={ref} style={{ height: rows * ROW_PX + 3 }}>
      {width > 0 && events.map(({ e, kind, label, title }, i) => (
        <button key={e.id} type="button" className={`${s.ev} ${s[kind]}`} style={{ left: places[i].x, top: 3 + places[i].row * ROW_PX }} title={title} onClick={() => onSeek(Math.max(0, e.atS - 0.5))}>
          <Icon name={kind === "tool" ? "tool" : kind === "db" ? "db" : "warn"} size={10} />{label}
        </button>
      ))}
    </div>
  );
}
