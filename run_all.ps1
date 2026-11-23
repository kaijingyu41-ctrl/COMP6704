# Reproduce the whole study from scratch.
#
# Usage (from the repository root):   ./run_all.ps1
#
# The script installs the pinned dependencies, runs the unit tests, runs all seven
# experiments, regenerates the figures and the LaTeX tables, and finally builds
# the report. Every solver parameter and random seed is fixed in
# src/otopt/config.py.
#
# Every step is checked: if a command returns a non-zero exit code the script
# stops immediately, so a failed experiment can never be silently followed by a
# rebuild of a stale PDF.

$ErrorActionPreference = "Stop"

$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "step failed: $Label (exit code $LASTEXITCODE)"
    }
}

Invoke-Checked "install dependencies" { & $Python -m pip install -q -r requirements.txt }

Invoke-Checked "unit tests" { & $Python -m pytest tests -q }

Invoke-Checked "experiment A (correctness certificates + convergence traces)" {
    & $Python experiments/run_experiment_a.py
}

Invoke-Checked "experiment B (scaling and marginal shape)" {
    & $Python experiments/run_experiment_b.py
}

Invoke-Checked "experiment C (parameter sweeps and ablations)" {
    & $Python experiments/run_experiment_c.py
}

Invoke-Checked "experiment D (handwritten-digit transport problems)" {
    & $Python experiments/run_experiment_d.py
}

Invoke-Checked "experiment E (long-budget checkpoints; about 10 minutes)" {
    & $Python experiments/run_experiment_e.py
}

Invoke-Checked "experiment F (timing repeatability; about 3 minutes)" {
    & $Python experiments/run_experiment_f.py
}

Invoke-Checked "experiment G (entropic/Sinkhorn baseline + full-grid continuation; the longest step)" {
    & $Python experiments/run_experiment_g.py
}

Invoke-Checked "figures" { & $Python experiments/make_figures.py }
Invoke-Checked "LaTeX tables" { & $Python experiments/make_tables.py }

# pdflatex is usually not on PATH; fall back to the default MiKTeX location.
$pdflatex = (Get-Command pdflatex -ErrorAction SilentlyContinue).Source
if (-not $pdflatex) {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if (Test-Path $candidate) { $pdflatex = $candidate }
}
if (-not $pdflatex) {
    Write-Warning "pdflatex not found; skipping the report build."
    Write-Warning "results/ figures/ and report/tables/ are up to date."
    exit 0
}

Push-Location report
try {
    Invoke-Checked "report (pdflatex pass 1)" { & $pdflatex -interaction=nonstopmode main.tex | Out-Null }
    Invoke-Checked "report (pdflatex pass 2)" { & $pdflatex -interaction=nonstopmode main.tex | Out-Null }

    if (-not (Test-Path "main.pdf")) { throw "report/main.pdf was not produced" }
    # The assignment asks for "name+studentID", so that spelling is the submitted
    # one; the space-separated name is refreshed alongside it so that the two can
    # never drift apart.
    # ... in report/, in the repository root (where the README links it) and in the
    # parent directory, under both spellings of the file name.
    Copy-Item main.pdf "YU Kaijing+25145003r.pdf" -Force
    Copy-Item main.pdf "YU Kaijing 25145003r.pdf" -Force
    Copy-Item main.pdf "../YU Kaijing+25145003r.pdf" -Force
    Copy-Item main.pdf "../YU Kaijing 25145003r.pdf" -Force
    Copy-Item main.pdf "../../YU Kaijing+25145003r.pdf" -Force
    Copy-Item main.pdf "../../YU Kaijing 25145003r.pdf" -Force
}
finally {
    Pop-Location
}

Write-Host "done: report/main.pdf copied to report/, the repository root and the parent directory, under both file-name spellings" -ForegroundColor Green
