"""
Device-agnostic epoch alignment for the NeSy ingestion pipeline.

Every EEG device starts recording at an arbitrary clock time, so the
epoch grid is anchored to that moment rather than the :00/:30 wall-clock
boundary the framework uses internally. A Dreem recording starting at
21:09:55 produces epochs at :25 and :55. Another participant starting at
21:07:12 gets :12 and :42. The offset is arbitrary per recording.

This module maps any device's epochs onto a canonical 30-second grid
(:00 and :30 of each minute) using majority-overlap assignment: each
grid slot gets the label of the epoch that covers the most of it,
provided that coverage exceeds min_overlap_sec (default 15s = 50%).

Slots at recording boundaries where no epoch meets the threshold are
left as -1 (unscored) rather than imputed. Forward-fill handles gaps
up to fill_limit consecutive slots.

Fitbit epochs are already on the :00/:30 grid (pre-aligned internally
by Fitbit's servers before export). For those, alignment is skipped —
controlled by the pre_aligned flag in devices.yaml.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

GRID_RESOLUTION_SEC = 30


def needs_alignment(device_config):
    """Return True if this device's epochs need grid alignment."""
    return not device_config.get("pre_aligned", False)


@dataclass
class AlignmentReport:
    participant_id: str
    night_date: str
    device: str
    n_source_epochs: int
    n_grid_slots: int
    n_assigned: int
    n_forward_filled: int
    n_unscored: int
    coverage_pct: float

    def __repr__(self):
        return (f"AlignmentReport({self.participant_id}, {self.night_date}, "
                f"device={self.device}, coverage={self.coverage_pct:.1f}%)")


class EpochAligner:

    def __init__(self, grid_resolution_sec=30, min_overlap_sec=15, fill_limit=4):
        self.grid_resolution_sec = grid_resolution_sec
        self.min_overlap_sec = min_overlap_sec
        self.fill_limit = fill_limit

    def align(self, epochs, participant_id, night_date, device="unknown"):
        if epochs.empty:
            empty = pd.DataFrame(columns=["grid_start", "grid_end", "sleep_stage", "assignment_type"])
            report = AlignmentReport(participant_id, night_date, device, 0, 0, 0, 0, 0, 0.0)
            return empty, report

        epochs = epochs.sort_values("epoch_start").reset_index(drop=True)
        epoch_starts = pd.to_datetime(epochs["epoch_start"], utc=True)
        epoch_stages = epochs["sleep_stage"].values
        epoch_dur = epochs["duration_sec"].values

        grid_start, grid_end = self._grid_bounds(epoch_starts, epoch_dur)
        grid_slots = self._build_grid(grid_start, grid_end)

        assigned = np.full(len(grid_slots), -1, dtype=int)
        assignment_type = np.full(len(grid_slots), "unscored", dtype=object)

        # vectorised overlap — convert everything to integer seconds
        # then use numpy broadcasting instead of a Python double loop
        grid_t = np.array(grid_slots.astype(np.int64)) // 10 ** 9  # seconds
        ep_t = np.array(epoch_starts.astype(np.int64)) // 10 ** 9
        ep_end_t = ep_t + epoch_dur.astype(np.int64)
        res = self.grid_resolution_sec
        min_ov = self.min_overlap_sec

        # shape: (n_slots, n_epochs)
        ov_start = np.maximum(grid_t[:, None], ep_t[None, :])
        ov_end = np.minimum(grid_t[:, None] + res, ep_end_t[None, :])
        overlaps = np.maximum(0, ov_end - ov_start)  # seconds

        # mask epochs below threshold
        overlaps[overlaps < min_ov] = 0

        best_ep = np.argmax(overlaps, axis=1)  # index of best epoch per slot
        best_ov = overlaps[np.arange(len(grid_slots)), best_ep]

        has_match = best_ov >= min_ov
        assigned[has_match] = epoch_stages[best_ep[has_match]].astype(int)
        assignment_type[has_match] = "overlap"

        assigned, assignment_type = self._forward_fill(assigned, assignment_type)

        n_assigned = int((assignment_type == "overlap").sum())
        n_forward_filled = int((assignment_type == "forward_fill").sum())
        n_unscored = int((assignment_type == "unscored").sum())
        coverage = (n_assigned + n_forward_filled) / len(grid_slots) * 100

        result = pd.DataFrame({
            "grid_start": grid_slots,
            "grid_end": grid_slots + pd.Timedelta(seconds=self.grid_resolution_sec),
            "sleep_stage": assigned,
            "assignment_type": assignment_type,
        })

        report = AlignmentReport(
            participant_id, night_date, device,
            len(epochs), len(grid_slots),
            n_assigned, n_forward_filled, n_unscored,
            round(coverage, 2),
        )
        return result, report
    #
    # def align_to_embraceplus(self, aligned_epochs, ep_ts):
    #     result = ep_ts.copy()
    #     result["eeg_stage"] = -1
    #     if aligned_epochs.empty:
    #         return result
    #     grid = aligned_epochs.copy()
    #     grid["grid_start"] = pd.to_datetime(grid["grid_start"], utc=True)
    #     for ts, _ in result.iterrows():
    #         ts_utc = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    #         min_end = ts_utc + pd.Timedelta(minutes=1)
    #         mask = (grid["grid_start"] >= ts_utc) & (grid["grid_start"] < min_end)
    #         stages = grid.loc[mask, "sleep_stage"].values
    #         valid = stages[stages >= 0]
    #         if len(valid) > 0:
    #             result.at[ts, "eeg_stage"] = int(pd.Series(valid).mode().iloc[0])
    #     return result

    # def align_to_embraceplus(self, aligned_epochs, ep_ts):
    #     result = ep_ts.copy()
    #     result["eeg_stage"] = -1
    #     if aligned_epochs.empty:
    #         return result
    #
    #     # FIX: Reset index if grid_start is currently the index
    #     grid = aligned_epochs.copy()
    #     if "grid_start" not in grid.columns and grid.index.name == "grid_start":
    #         grid = grid.reset_index()
    #     elif "grid_start" not in grid.columns:
    #         grid = grid.reset_index().rename(columns={"index": "grid_start"})
    #
    #     grid["grid_start"] = pd.to_datetime(grid["grid_start"], utc=True)
    #
    #     for ts, _ in result.iterrows():
    #         ts_utc = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    #         min_end = ts_utc + pd.Timedelta(minutes=1)
    #         mask = (grid["grid_start"] >= ts_utc) & (grid["grid_start"] < min_end)
    #         stages = grid.loc[mask, "sleep_stage"].values
    #         valid = stages[stages >= 0]
    #         if len(valid) > 0:
    #             result.at[ts, "eeg_stage"] = int(pd.Series(valid).mode().iloc[0])
    #     return result

    def align_to_embraceplus(self, aligned_epochs, ep_ts):
        result = ep_ts.copy()
        result["eeg_stage"] = -1
        if aligned_epochs.empty:
            return result

        # Ensure grid_start is a column
        grid = aligned_epochs.copy()
        if "grid_start" not in grid.columns:
            grid = grid.reset_index()
            if "grid_start" not in grid.columns:
                grid = grid.rename(columns={grid.columns[0]: "grid_start"})

        # Floor EEG 30s grid_start timestamps to 1-minute intervals
        grid["grid_start"] = pd.to_datetime(grid["grid_start"], utc=True)
        grid["minute_dt"] = grid["grid_start"].dt.floor("1min")

        # Keep valid sleep stages (>= 0) and find the mode stage per (participant_id, minute)
        valid_grid = grid[grid["sleep_stage"] >= 0]
        if valid_grid.empty:
            return result

        # Compute mode stage per participant per minute
        mode_stages = (
            valid_grid.groupby(["participant_id", "minute_dt"])["sleep_stage"]
            .agg(lambda x: pd.Series.mode(x).iloc[0] if not x.empty else -1)
            .rename("eeg_stage")
        )

        # Merge onto EmbracePlus timestamp index
        ep_df = result.reset_index()
        ep_dt_col = ep_df.columns[0]  # datetime index column name
        ep_df["minute_dt"] = pd.to_datetime(ep_df[ep_dt_col], utc=True).dt.floor("1min")

        merged = ep_df.merge(
            mode_stages,
            on=["participant_id", "minute_dt"],
            how="left",
            suffixes=("", "_new"),
        )

        if "eeg_stage_new" in merged.columns:
            result["eeg_stage"] = merged["eeg_stage_new"].fillna(-1).values.astype(int)
        elif "eeg_stage" in merged.columns:
            result["eeg_stage"] = merged["eeg_stage"].fillna(-1).values.astype(int)

        return result

    def _grid_bounds(self, epoch_starts, epoch_dur):
        first = epoch_starts.iloc[0]
        last = epoch_starts.iloc[-1] + pd.Timedelta(seconds=int(epoch_dur[-1]))
        res = pd.Timedelta(seconds=self.grid_resolution_sec)
        origin = first.replace(hour=0, minute=0, second=0, microsecond=0)
        first_grid = origin + ((first - origin) // res) * res
        last_secs = (last - origin).total_seconds()
        last_grid = origin + pd.Timedelta(
            seconds=int(-(-last_secs // self.grid_resolution_sec)) * self.grid_resolution_sec)
        if last_grid.tzinfo is None and first.tzinfo is not None:
            last_grid = last_grid.tz_localize(first.tzinfo)
        return first_grid, last_grid

    def _build_grid(self, start, end):
        return pd.date_range(start=start, end=end,
                             freq=f"{self.grid_resolution_sec}s", inclusive="left")

    @staticmethod
    def _overlap(slot_start, slot_end, ep_start, ep_end):
        overlap_start = max(slot_start, ep_start)
        overlap_end = min(slot_end, ep_end)
        if overlap_end <= overlap_start:
            return 0.0
        return (overlap_end - overlap_start).total_seconds()

    def _forward_fill(self, assigned, assignment_type):
        assigned = assigned.copy()
        assignment_type = assignment_type.copy()
        last_valid = -1
        fill_count = 0
        for i in range(len(assigned)):
            if assignment_type[i] == "overlap":
                last_valid = assigned[i]
                fill_count = 0
            elif last_valid >= 0 and fill_count < self.fill_limit:
                assigned[i] = last_valid
                assignment_type[i] = "forward_fill"
                fill_count += 1
            else:
                fill_count += 1
        return assigned, assignment_type
