"""
Python port of Carrot's fingerprint solver (AHK v1.5), now cross-platform.

Hotkeys (global):
- Ctrl+E: run the solver once.
- Ctrl+R: reset cursor to grid top-left.
- Right Shift (hold): show the scan area overlay.
- Ctrl+T: exit.

Dependencies (install in the same environment you run the game):
    pip install mss opencv-python numpy keyboard

CLI options:
- --monitor N            Select which monitor to capture (mss index, default 1).
- --resolution WxH       Override resolution for scan area + template scaling (e.g., 2560x1440).
- --match-threshold F    Matching cutoff (default 0.58; higher = stricter).
- --scan-box x1,y1,x2,y2 Manual scan area override (screen coords).
- --reset-state          Force cursor to top-left before/after solving to avoid drift.

Notes:
- Runs best with the game in windowed borderless mode so screen captures work and key events land.
- If no template folder for your resolution exists, the script scales the 1920x1080
  templates up/down (covers 2560x1440/2K automatically).
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import mss
import numpy as np


BASE_RESOLUTION = (1920, 1080)
TEMPLATE_COUNT = 16
MATCH_THRESHOLD = 0.58  # Increase if you get false positives, decrease if matches are missed.


def parse_resolution(value: str) -> Tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must look like 2560x1440")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution numbers must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")
    return w, h


def parse_scan_box(value: str) -> Tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Scan box must look like x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Scan box values must be integers") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("Scan box must have x2>x1 and y2>y1")
    return x1, y1, x2, y2


@dataclass(frozen=True)
class DisplayInfo:
    width: int
    height: int
    left: int
    top: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class ScanArea:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def mss_region(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }


def get_display_info(monitor_index: int) -> DisplayInfo:
    """Cross-platform monitor info; monitor_index follows mss (1-based)."""
    with mss.mss() as sct:
        try:
            mon = sct.monitors[monitor_index]
        except IndexError as exc:  # pragma: no cover - depends on hardware
            raise ValueError(f"Monitor {monitor_index} not found. Available: 1..{len(sct.monitors) - 1}") from exc

        return DisplayInfo(
            width=int(mon["width"]),
            height=int(mon["height"]),
            left=int(mon["left"]),
            top=int(mon["top"]),
        )


def compute_scan_area(display: DisplayInfo, logical_size: Tuple[int, int]) -> ScanArea:
    """Recreates the original AHK math anchored to the selected monitor."""
    screen_width, screen_height = logical_size

    x1 = display.left + (screen_width / 4.2)
    y1 = display.top + screen_height / 4.25
    x2 = display.left + (screen_width / 2.55)
    y2 = display.top + screen_height / 1.29

    left = int(min(x1, x2))
    right = int(max(x1, x2))
    top = int(min(y1, y2))
    bottom = int(max(y1, y2))

    if right - left <= 0 or bottom - top <= 0:
        raise ValueError("Scan area has invalid dimensions; verify monitor and resolution settings.")

    return ScanArea(left=left, top=top, right=right, bottom=bottom)


def load_templates(root: Path, screen_size: Tuple[int, int]) -> Tuple[List[np.ndarray], Tuple[float, float], Path]:
    screen_w, screen_h = screen_size
    exact_dir = root / f"{screen_w}x{screen_h}"
    fallback_dir = root / f"{BASE_RESOLUTION[0]}x{BASE_RESOLUTION[1]}"

    if exact_dir.exists():
        source_dir = exact_dir
        scale = (1.0, 1.0)
    elif fallback_dir.exists():
        source_dir = fallback_dir
        scale = (screen_w / BASE_RESOLUTION[0], screen_h / BASE_RESOLUTION[1])
    else:
        raise FileNotFoundError("No template folder found. Expected a folder like 1920x1080 with 1.bmp..16.bmp.")

    templates: List[np.ndarray] = []
    for idx in range(1, TEMPLATE_COUNT + 1):
        path = source_dir / f"{idx}.bmp"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Template missing or unreadable: {path}")

        if scale != (1.0, 1.0):
            new_w = max(1, int(round(img.shape[1] * scale[0])))
            new_h = max(1, int(round(img.shape[0] * scale[1])))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        templates.append(img)

    return templates, scale, source_dir


class GenericKeyboard:
    """Cross-platform keyboard events via the keyboard library."""

    def tap(self, key: str, repeat: int = 1, delay: float = 0.02) -> None:  # pragma: no cover - hardware dependent
        for _ in range(repeat):
            keyboard.press_and_release(key)
            if delay:
                time.sleep(delay)


class Overlay:
    """Red translucent overlay that mirrors the scan area while Right Shift is held."""

    def __init__(self, area: ScanArea) -> None:
        self.area = area
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def show(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._queue.put("show")

    def hide(self) -> None:
        self._queue.put("hide")

    def _run(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.35)
        root.configure(bg="red")
        # geometry wants signed offsets; "+-10" is interpreted as "-10".
        root.geometry(f"{self.area.width}x{self.area.height}{self.area.left:+}{self.area.top:+}")
        root.withdraw()

        def poll() -> None:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                root.after(50, poll)
                return

            if cmd == "show":
                root.deiconify()
            elif cmd == "hide":
                root.withdraw()

            root.after(50, poll)

        poll()
        root.mainloop()


class FingerprintSolver:
    def __init__(
        self,
        template_root: Path,
        monitor_index: int,
        logical_resolution: Optional[Tuple[int, int]],
        match_threshold: float,
        scan_override: Optional[Tuple[int, int, int, int]],
        reset_state: bool,
    ) -> None:
        self.template_root = template_root
        self.monitor = get_display_info(monitor_index)
        self.screen_size = logical_resolution or (self.monitor.width, self.monitor.height)
        if scan_override:
            self.area = ScanArea(*scan_override)
        else:
            self.area = compute_scan_area(self.monitor, self.screen_size)
        self.templates, self.scale, self.template_dir = load_templates(template_root, self.screen_size)

        # Pre-compute grayscale templates for faster matching.
        self.templates_gray = [cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) for tpl in self.templates]

        self.match_threshold = match_threshold
        self.reset_state = reset_state
        self.keyboard = GenericKeyboard()
        self.current_pos = (1, 1)
        self._lock = threading.Lock()

    def _grab_area(self) -> np.ndarray:
        with mss.mss() as sct:
            shot = sct.grab(self.area.mss_region)
            frame = np.array(shot)
            # mss returns BGRA; drop alpha and flip to BGR for OpenCV.
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame

    def _match_template(self, frame_gray: np.ndarray, template_gray: np.ndarray) -> Optional[Tuple[int, int]]:
        res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.match_threshold:
            return None

        hit_x = max_loc[0] + self.area.left
        hit_y = max_loc[1] + self.area.top
        return hit_x, hit_y

    def _map_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        col_width = self.area.width / 2
        row_height = self.area.height / 4

        gx = int((x - self.area.left) // col_width) + 1
        gy = int((y - self.area.top) // row_height) + 1

        gx = max(1, min(2, gx))
        gy = max(1, min(4, gy))
        return gx, gy

    def _tap(self, key: str, repeat: int, delay: float) -> None:
        self.keyboard.tap(key, repeat=repeat, delay=delay)

    def _hard_reset_cursor(self) -> None:
        """Force cursor to top-left of the 2x4 grid to avoid drift."""
        # Two columns -> 2 moves left is enough; add one extra for safety.
        self._tap("a", 3, 0.02)
        # Four rows -> 4 moves up is enough; add one extra for safety.
        self._tap("w", 5, 0.02)
        self.current_pos = (1, 1)

    def reset_cursor(self) -> None:
        """Public reset trigger (hotkey)."""
        if not self._lock.acquire(blocking=False):
            print("Reset skipped; solver is running.")
            return
        try:
            self._hard_reset_cursor()
            print("Cursor reset to top-left.")
        finally:
            self._lock.release()

    def _move_cursor(self, target: Tuple[int, int], click: bool = True) -> None:
        dx = target[0] - self.current_pos[0]
        dy = target[1] - self.current_pos[1]

        if dx > 0:
            self._tap("d", dx, 0.025)
        elif dx < 0:
            self._tap("a", -dx, 0.025)

        if dy > 0:
            self._tap("s", dy, 0.02)
        elif dy < 0:
            self._tap("w", -dy, 0.02)

        if click:
            self.keyboard.tap("enter", repeat=1, delay=0.0)

        self.current_pos = target

    def match_fingerprint(self) -> None:
        if not self._lock.acquire(blocking=False):
            print("Match already running, ignoring request.")
            return

        start = time.time()
        try:
            if self.reset_state:
                self._hard_reset_cursor()

            self.current_pos = (1, 1)
            frame = self._grab_area()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            matches = 0
            for template_gray in self.templates_gray:
                hit = self._match_template(frame_gray, template_gray)
                if not hit:
                    continue

                grid_pos = self._map_to_grid(*hit)
                self._move_cursor(grid_pos, click=True)
                matches += 1

            elapsed = (time.time() - start) * 1000
            print(f"Matches found: {matches} | Capture + solve: {elapsed:.0f}ms")

            if matches == 0:
                self.keyboard.tap("tab", repeat=1, delay=0.0)
            elif matches < 4:
                self._move_cursor((1, 1), click=False)
            else:
                time.sleep(0.01)
                self.keyboard.tap("tab", repeat=1, delay=0.0)
            if self.reset_state:
                self._hard_reset_cursor()
        finally:
            self._lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Casino fingerprint solver")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (mss numbering, default 1)")
    parser.add_argument("--resolution", type=parse_resolution, help="Override resolution for scan area/templates, e.g., 2560x1440")
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD, help="Template match threshold (0..1, default 0.58)")
    parser.add_argument(
        "--scan-box",
        type=parse_scan_box,
        help="Override scan box as x1,y1,x2,y2 (screen coords). Skips auto area calculation.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Force cursor to the top-left of the grid before and after solving (helps if movement drifted).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    try:
        solver = FingerprintSolver(
            template_root=script_dir,
            monitor_index=args.monitor,
            logical_resolution=args.resolution,
            match_threshold=args.match_threshold,
            scan_override=args.scan_box,
            reset_state=args.reset_state,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Startup failed: {exc}")
        return

    overlay = Overlay(solver.area)
    shutdown = threading.Event()

    keyboard.add_hotkey("ctrl+e", solver.match_fingerprint, trigger_on_release=False)
    keyboard.add_hotkey("ctrl+r", solver.reset_cursor, trigger_on_release=False)
    keyboard.on_press_key("right shift", lambda _: overlay.show())
    keyboard.on_release_key("right shift", lambda _: overlay.hide())
    keyboard.add_hotkey("ctrl+t", shutdown.set)

    print("Casino fingerprint solver (Python)")
    print(
        f"Monitor {args.monitor}: {solver.monitor.width}x{solver.monitor.height} "
        f"@ ({solver.monitor.left},{solver.monitor.top}) | Using resolution {solver.screen_size[0]}x{solver.screen_size[1]}"
    )
    print(f"Templates: {solver.template_dir.name} | Match threshold: {solver.match_threshold}")
    print(f"Scan area: {solver.area.left},{solver.area.top} -> {solver.area.right},{solver.area.bottom}")
    print("Hotkeys: Ctrl+E to solve, Ctrl+R to reset cursor, Right Shift to preview area, Ctrl+T to exit.")

    try:
        while not shutdown.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
