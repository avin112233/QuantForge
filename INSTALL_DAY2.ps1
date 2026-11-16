# Run from C:\QuantForge\QuantForge after extracting Day 2 files into that directory.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.\main.py')) { throw 'Run this script from the existing QuantForge project root.' }
& .\.venv\Scripts\python.exe -m pip install 'yfinance>=0.2.60,<2'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pytest -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
Write-Host 'Tests passed. Next run: .\.venv\Scripts\python.exe day2.py'
