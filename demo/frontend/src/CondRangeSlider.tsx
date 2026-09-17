import Slider from "rc-slider";
import "rc-slider/assets/index.css";
import type { Range } from "./constants";
import { formatCondLabel, formatCondValue, sliderBounds } from "./constants";

type Props = {
  name: string;
  value: Range;
  selected?: number;
  onChange: (next: Range) => void;
};

export function CondRangeSlider({ name, value, selected, onChange }: Props) {
  const { min, max, step } = sliderBounds(name);
  const tickPct =
    selected == null ? null : ((Math.min(max, Math.max(min, selected)) - min) / (max - min)) * 100;

  return (
    <div className="cond-row">
      <div className="cond-label">{formatCondLabel(name)}</div>
      <div className="cond-slider">
        <Slider
          range
          min={min}
          max={max}
          step={step}
          value={[value.min, value.max]}
          onChange={(v) => {
            const [lo, hi] = v as [number, number];
            onChange({ min: lo, max: hi });
          }}
        />
        {tickPct != null && selected != null && (
          <span className="cond-selected-tick" style={{ left: `${tickPct}%` }}>
            {formatCondValue(name, selected)}
          </span>
        )}
      </div>
      <div className="cond-values">
        [{formatCondValue(name, value.min)}, {formatCondValue(name, value.max)}]
      </div>
    </div>
  );
}
