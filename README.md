# GTA V Casino Fingerprint Solver

Python port of Carrot’s AHK fingerprint solver with multi‑monitor support and template scaling.

## Quick start
- Prereq: Python 3.9+; game in windowed/borderless so capture + key events land.
- From the repo root:
  ```bash
  cd gtav-casino-solver
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Run (example: game on monitor 2 at 1920x1080):
  ```bash
  python casino-solver.py --monitor 2 --resolution 1920x1080
  ```

## Controls
- Ctrl+E: solve once.
- Ctrl+Shift+R: reset cursor.
- Right Shift (hold): show scan-area overlay.
- Ctrl+T: exit.

## CLI options
- `--monitor N` — which display to capture (mss numbering, 1-based).
- `--resolution WxH` — your in-game resolution; auto scales templates if a native folder is absent.
- `--match-threshold F` — similarity cutoff (default 0.58; higher = stricter).
- `--scan-box x1,y1,x2,y2` — manual scan rectangle in screen coords if auto placement is off.

## Assets
- Templates live in `1920x1080/1.bmp..16.bmp`. If your exact resolution folder is missing, the solver scales the 1080p set (works for 2560x1440/2K). For best accuracy at other resolutions, capture a native set named `WxH/1.bmp..16.bmp`.

## Tips & troubleshooting
- Verify the printed display info and “Scan area” at startup match your game monitor. Use Right Shift overlay to confirm coverage. If misaligned, tune `--scan-box`.
- Keep the game window focused; some titles may ignore synthetic key events otherwise.
- If false positives/misses: tweak `--match-threshold` slightly (e.g., 0.52–0.65). 
- Multi-monitor: mss indices are 1..N; try `--monitor 1`, then 2, etc., until the printed bounds match your target display.

## Linux Support
Since the release of GTAV Enhanced, BattleEye anticheat does not play nicely with Linux anymore. You will get kicked almost immediately or within couple of minutes of entering a lobby. Thus, I have abandoned the Linux support. Should the BattleEye situation get resolved in the future, I will happily add support again.
