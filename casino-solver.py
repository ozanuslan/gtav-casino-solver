"""
Python port of Carrot's fingerprint solver (AHK v1.5), now cross-platform.

Hotkeys (global):
- Ctrl+E: run the solver once.
- Ctrl+R: reset cursor to grid top-left.
- Right Shift (hold): show the scan area overlay.
- Ctrl+T: exit.

Dependencies (install in the same environment you run the game):
    pip install mss opencv-python numpy pynput

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
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mss
import numpy as np
from pynput import keyboard as pynput_keyboard
from pynput.keyboard import Key

BASE_RESOLUTION = (1920, 1080)
TEMPLATE_COUNT = 16
MATCH_THRESHOLD = (
    0.58  # Increase if you get false positives, decrease if matches are missed.
)

# Game control keys
KEY_MOVE_LEFT = "a"
KEY_MOVE_RIGHT = "d"
KEY_MOVE_UP = "w"
KEY_MOVE_DOWN = "s"
KEY_CONFIRM = Key.enter
KEY_SKIP = Key.tab


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
            raise ValueError(
                f"Monitor {monitor_index} not found. Available: 1..{len(sct.monitors) - 1}"
            ) from exc

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
        raise ValueError(
            "Scan area has invalid dimensions; verify monitor and resolution settings."
        )

    return ScanArea(left=left, top=top, right=right, bottom=bottom)


def load_templates(
    root: Path, screen_size: Tuple[int, int]
) -> Tuple[List[np.ndarray], Tuple[float, float], Path]:
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
        raise FileNotFoundError(
            "No template folder found. Expected a folder like 1920x1080 with 1.bmp..16.bmp."
        )

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


class RustInputter:
    """Keyboard input using the Rust enigo-based binary (works with games like GTA V)."""

    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        if not self.binary_path.exists():
            raise FileNotFoundError(f"Rust inputter not found at {binary_path}")

    def send_grid_solution(self, grid: List[List[int]]) -> None:
        """Send a 2x4 grid solution to the Rust binary."""
        json_data = json.dumps(grid)
        print(f"[DEBUG] Calling Rust inputter with grid: {grid}")

        try:
            result = subprocess.run(
                [str(self.binary_path), json_data],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                print(f"[WARNING] Rust inputter failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("[WARNING] Rust inputter timed out")
        except Exception as e:
            print(f"[WARNING] Failed to call Rust inputter: {e}")


class WindowsDirectInput:
    """Direct keyboard input using Windows SendInput API (works with games like GTA V)."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        # Store ctypes module for later use
        self.ctypes = ctypes
        self.wintypes = wintypes

        self.SendInput = ctypes.windll.user32.SendInput
        self.MapVirtualKeyW = ctypes.windll.user32.MapVirtualKeyW
        self.PUL = ctypes.POINTER(ctypes.c_ulong)

        # Virtual key codes
        self.VK_CODES = {
            "a": 0x41,
            "d": 0x44,
            "w": 0x57,
            "s": 0x53,
            "enter": 0x0D,
            "tab": 0x09,
            "shift": 0x10,
            "ctrl": 0x11,
        }

        # Scan codes for hardware-level input
        self.SCAN_CODES = {
            "a": 0x1E,
            "d": 0x20,
            "w": 0x11,
            "s": 0x1F,
            "enter": 0x1C,
            "tab": 0x0F,
        }

        # Define structures for SendInput
        class KeyBdInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class HardwareInput(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class InputUnion(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyBdInput), ("hi", HardwareInput)]

        class Input(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]

        self.KeyBdInput = KeyBdInput
        self.Input = Input
        self.INPUT_KEYBOARD = 1
        self.KEYEVENTF_KEYUP = 0x0002
        self.KEYEVENTF_SCANCODE = 0x0008

    def _get_vk_code(self, key: str | Key) -> int:
        """Get virtual key code for a key."""
        if isinstance(key, Key):
            key_map = {
                Key.enter: "enter",
                Key.tab: "tab",
                Key.shift: "shift",
                Key.ctrl: "ctrl",
            }
            key = key_map.get(key, "")

        key_lower = key.lower() if isinstance(key, str) else ""
        return self.VK_CODES.get(key_lower, 0)

    def _get_scan_code(self, key: str | Key) -> int:
        """Get hardware scan code for a key."""
        if isinstance(key, Key):
            key_map = {
                Key.enter: "enter",
                Key.tab: "tab",
            }
            key = key_map.get(key, "")

        key_lower = key.lower() if isinstance(key, str) else ""
        return self.SCAN_CODES.get(key_lower, 0)

    def tap(self, key: str | Key, repeat: int = 1, delay: float = 0.02) -> None:
        """Send a key press using Windows SendInput API with hardware scan codes."""
        scan_code = self._get_scan_code(key)

        if scan_code == 0:
            print(f"[WARNING] Unknown key: {key}")
            return

        print(
            f"[DEBUG] Tapping key: {key} (SC:{scan_code:#04x}) x{repeat} times with {delay}s delay"
        )

        for _ in range(repeat):
            # Key down - using scan code for hardware-level input
            ki_down = self.KeyBdInput(0, scan_code, self.KEYEVENTF_SCANCODE, 0, None)
            input_down = self.Input(self.INPUT_KEYBOARD)
            input_down.union.ki = ki_down
            self.SendInput(
                1, self.ctypes.byref(input_down), self.ctypes.sizeof(input_down)
            )

            time.sleep(0.015)  # 15ms delay between press and release

            # Key up - using scan code for hardware-level input
            ki_up = self.KeyBdInput(
                0, scan_code, self.KEYEVENTF_SCANCODE | self.KEYEVENTF_KEYUP, 0, None
            )
            input_up = self.Input(self.INPUT_KEYBOARD)
            input_up.union.ki = ki_up
            self.SendInput(1, self.ctypes.byref(input_up), self.ctypes.sizeof(input_up))

            if delay:
                time.sleep(delay)


class GenericKeyboard:
    """Cross-platform keyboard events via pynput."""

    def __init__(self) -> None:
        self.controller = pynput_keyboard.Controller()

    @staticmethod
    def _normalize_key(key: str | Key) -> Union[pynput_keyboard.Key, str]:
        if isinstance(key, Key):
            return key
        lookup = {
            "enter": pynput_keyboard.Key.enter,
            "tab": pynput_keyboard.Key.tab,
            "shift": pynput_keyboard.Key.shift,
            "ctrl": pynput_keyboard.Key.ctrl,
        }
        return lookup.get(key.lower(), key)

    def tap(
        self, key: str | Key, repeat: int = 1, delay: float = 0.02
    ) -> None:  # pragma: no cover - hardware dependent
        key_obj = self._normalize_key(key)
        print(
            f"[DEBUG] Tapping key: {key} ({key_obj}) x{repeat} times with {delay}s delay"
        )
        for _ in range(repeat):
            self.controller.press(key_obj)
            self.controller.release(key_obj)
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
        root.geometry(
            f"{self.area.width}x{self.area.height}{self.area.left:+}{self.area.top:+}"
        )
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
        self.screen_size = logical_resolution or (
            self.monitor.width,
            self.monitor.height,
        )
        if scan_override:
            self.area = ScanArea(*scan_override)
        else:
            self.area = compute_scan_area(self.monitor, self.screen_size)
        self.templates, self.scale, self.template_dir = load_templates(
            template_root, self.screen_size
        )

        # Pre-compute grayscale templates for faster matching.
        self.templates_gray = [
            cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) for tpl in self.templates
        ]

        self.match_threshold = match_threshold
        self.reset_state = reset_state

        # Use WindowsDirectInput - Rust binary has compatibility issues with old enigo version
        print(f"[INFO] Using WindowsDirectInput for key sending")
        self.use_rust = False
        self.keyboard = WindowsDirectInput()
        self.rust_inputter = None

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

    def _match_template(
        self, frame_gray: np.ndarray, template_gray: np.ndarray
    ) -> Optional[Tuple[int, int]]:
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

    def _tap(self, key: str | Key, repeat: int, delay: float) -> None:
        self.keyboard.tap(key, repeat=repeat, delay=delay)

    def _hard_reset_cursor(self) -> None:
        """Force cursor to top-left of the 2x4 grid to avoid drift."""
        # Two columns -> 2 moves left is enough; add one extra for safety.
        self._tap(KEY_MOVE_LEFT, 3, 0.015)
        # Four rows -> 4 moves up is enough; add one extra for safety.
        self._tap(KEY_MOVE_UP, 5, 0.015)
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
            self._tap(KEY_MOVE_RIGHT, dx, 0.015)
        elif dx < 0:
            self._tap(KEY_MOVE_LEFT, -dx, 0.015)

        if dy > 0:
            self._tap(KEY_MOVE_DOWN, dy, 0.015)
        elif dy < 0:
            self._tap(KEY_MOVE_UP, -dy, 0.015)

        if click:
            self.keyboard.tap(KEY_CONFIRM, repeat=1, delay=0.0)

        self.current_pos = target

    def _optimize_path(self, positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimize the order of positions using dynamic programming (Traveling Salesman Problem approximation)."""
        if len(positions) <= 1:
            return positions

        # Calculate distance between two grid positions (Manhattan distance)
        def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Start from current position (1, 1)
        start = (1, 1)
        n = len(positions)

        # For small number of positions (typical case: 4 matches), use complete search with memoization
        if n <= 4:
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def min_path(
                current: Tuple[int, int], remaining: frozenset
            ) -> Tuple[int, List[Tuple[int, int]]]:
                """Find minimum cost path through remaining positions."""
                if not remaining:
                    return (0, [])

                best_cost = float("inf")
                best_path = []

                for next_pos in remaining:
                    cost = distance(current, next_pos)
                    new_remaining = remaining - {next_pos}
                    rest_cost, rest_path = min_path(next_pos, frozenset(new_remaining))
                    total_cost = cost + rest_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = [next_pos] + rest_path

                return (best_cost, best_path)

            _, optimal_path = min_path(start, frozenset(positions))
            return optimal_path

        # For larger sets, fall back to greedy nearest neighbor
        optimized = []
        remaining = positions.copy()
        current = start

        while remaining:
            nearest = min(remaining, key=lambda pos: distance(current, pos))
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest

        return optimized

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

            # Collect all matches first
            match_positions = []
            for template_gray in self.templates_gray:
                hit = self._match_template(frame_gray, template_gray)
                if not hit:
                    continue

                grid_pos = self._map_to_grid(*hit)
                print(
                    f"[DEBUG] Match detected at screen pos {hit} -> grid pos {grid_pos}"
                )
                match_positions.append(grid_pos)

            # Optimize the traversal order
            if match_positions:
                optimized_positions = self._optimize_path(match_positions)
                print(f"[DEBUG] Optimized path: {optimized_positions}")

                # Execute the optimized path
                for grid_pos in optimized_positions:
                    self._move_cursor(grid_pos, click=True)

            matches = len(match_positions)
            elapsed = (time.time() - start) * 1000
            print(f"Matches found: {matches} | Capture + solve: {elapsed:.0f}ms")

            if matches == 0:
                self.keyboard.tap(KEY_SKIP, repeat=1, delay=0.0)
            elif matches < 4:
                self._move_cursor((1, 1), click=False)
            else:
                time.sleep(0.01)
                self.keyboard.tap(KEY_SKIP, repeat=1, delay=0.0)
            if self.reset_state:
                self._hard_reset_cursor()
        finally:
            self._lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Casino fingerprint solver")
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index (mss numbering, default 1)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        help="Override resolution for scan area/templates, e.g., 2560x1440",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help="Template match threshold (0..1, default 0.58)",
    )
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

    hotkeys = pynput_keyboard.GlobalHotKeys(
        {
            "<ctrl>+e": solver.match_fingerprint,
            "<ctrl>+r": solver.reset_cursor,
            "<ctrl>+t": shutdown.set,
        }
    )
    hotkeys.start()

    def on_press(key: Union[pynput_keyboard.Key, pynput_keyboard.KeyCode]) -> None:
        if key == pynput_keyboard.Key.shift_r:
            overlay.show()

    def on_release(key: Union[pynput_keyboard.Key, pynput_keyboard.KeyCode]) -> None:
        if key == pynput_keyboard.Key.shift_r:
            overlay.hide()

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Casino fingerprint solver (Python)")
    print(
        f"Monitor {args.monitor}: {solver.monitor.width}x{solver.monitor.height} "
        f"@ ({solver.monitor.left},{solver.monitor.top}) | Using resolution {solver.screen_size[0]}x{solver.screen_size[1]}"
    )
    print(
        f"Templates: {solver.template_dir.name} | Match threshold: {solver.match_threshold}"
    )
    print(
        f"Scan area: {solver.area.left},{solver.area.top} -> {solver.area.right},{solver.area.bottom}"
    )
    print(
        "Hotkeys: Ctrl+E to solve, Ctrl+R to reset cursor, Right Shift to preview area, Ctrl+T to exit."
    )

    try:
        while not shutdown.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        hotkeys.stop()
        listener.stop()


if __name__ == "__main__":
    main()
