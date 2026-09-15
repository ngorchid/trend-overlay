@echo off
REM Trend Overlay daily run (Windows Task Scheduler, weekdays 20:30). Paper vs LIVE is set by
REM IB_PORT in .env (4003 paper / 4001 live) -- this wrapper is identical for both checkouts.
REM Runs from the repo root regardless of where it's invoked; appends output to a log.
cd /d "%~dp0.."
if not exist "results\paper" mkdir "results\paper"
call .venv\Scripts\activate.bat
python scripts\run_trend_paper.py --live %* 1>> "results\paper\run.log" 2>&1
