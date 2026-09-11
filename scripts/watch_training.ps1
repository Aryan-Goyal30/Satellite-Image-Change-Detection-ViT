<#
.SYNOPSIS
    Read-only live monitor for an in-progress Earth Guardian training run.

.DESCRIPTION
    Touches nothing the trainer owns. It only:
      * reads the training log file (read-only share)
      * reads checkpoint metadata from a TEMP COPY (never the live file)
      * queries nvidia-smi
      * inspects the python process start time

    Two data sources are combined because the log is block-buffered by the
    shell pipeline that launched training and can lag by several epochs:

      LOG        authoritative per-epoch history, but delayed
      CHECKPOINT updates as soon as val F1 improves, so it is fresher

.PARAMETER RefreshSeconds
    Redraw interval. Default 20.

.PARAMETER Once
    Print a single snapshot and exit (useful for piping/logging).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\watch_training.ps1
    powershell -ExecutionPolicy Bypass -File scripts\watch_training.ps1 -Once
#>
[CmdletBinding()]
param(
    [int]$RefreshSeconds = 20,
    [string]$LogPath = "",
    [string]$Checkpoint = "",
    [int]$TotalEpochs = 50,
    [switch]$Once
)

$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Checkpoint) { $Checkpoint = Join-Path $Root "checkpoints\siamese_unet_r34_best.pt" }
if (-not $LogPath)    { $LogPath    = Join-Path $Root "logs\train_r34.log" }

function Find-Log {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path $Explicit)) { return $Explicit }
    # newest task log that mentions the trainer
    $base = Join-Path $env:LOCALAPPDATA "Temp\claude"
    $cand = Get-ChildItem $base -Recurse -Filter "*.output" -File |
            Sort-Object LastWriteTime -Descending |
            Where-Object { (Select-String -Path $_.FullName -Pattern 'src\.train\.train|LAUNCHING TRAINING' -Quiet) } |
            Select-Object -First 1
    if ($cand) { return $cand.FullName }
    return $null
}

function Get-TrainProcess {
    # A venv launcher and the real interpreter both carry "src.train.train" in
    # their command line; the real trainer is the one accumulating CPU time.
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'src\.train\.train' } |
        Sort-Object UserModeTime -Descending |
        Select-Object -First 1
}

function Read-EpochRows {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path $Path)) { return @() }
    # open read-only + share write, so we never block the writer
    $fs = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
    try {
        $sr = New-Object System.IO.StreamReader($fs)
        $text = $sr.ReadToEnd()
    } finally { $fs.Dispose() }

    $rx = [regex]'(?m)^epoch\s+(\d+)/(\d+) \| loss ([\d.]+) \| val F1 ([\d.]+) IoU ([\d.]+) P ([\d.]+) R ([\d.]+) \| best-thr F1 ([\d.]+)@([\d.]+) \| (\d+)s'
    $rows = @()
    foreach ($m in $rx.Matches($text)) {
        $rows += [pscustomobject]@{
            Epoch = [int]$m.Groups[1].Value
            Total = [int]$m.Groups[2].Value
            Loss  = [double]$m.Groups[3].Value
            F1    = [double]$m.Groups[4].Value
            IoU   = [double]$m.Groups[5].Value
            P     = [double]$m.Groups[6].Value
            R     = [double]$m.Groups[7].Value
            BestThrF1 = [double]$m.Groups[8].Value
            Thr   = [double]$m.Groups[9].Value
            Sec   = [int]$m.Groups[10].Value
        }
    }
    return $rows
}

# Cache checkpoint reads: only re-open python when the file actually changes.
$script:CkCache = @{ Stamp = $null; Data = $null }
function Read-Checkpoint {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $fi = Get-Item $Path
    $stamp = "$($fi.LastWriteTimeUtc.Ticks)-$($fi.Length)"
    if ($script:CkCache.Stamp -eq $stamp) { return $script:CkCache.Data }

    # Copy first: the trainer may be mid-write, and we must never lock it.
    $tmp = Join-Path $env:TEMP ("eg_ckpt_probe_" + [guid]::NewGuid().ToString('N') + ".pt")
    try { Copy-Item $Path $tmp -ErrorAction Stop } catch { return $script:CkCache.Data }

    $py = Join-Path $Root ".venv312\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    $code = @'
import json, sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    vm = ck.get("val_metrics", {}) or {}
    bt = ck.get("val_best_threshold", {}) or {}
    print(json.dumps({
        "epoch": ck.get("epoch"),
        "encoder": ck.get("encoder"),
        "model_name": ck.get("model_name"),
        "f1": vm.get("f1"), "iou": vm.get("iou"),
        "precision": vm.get("precision"), "recall": vm.get("recall"),
        "best_thr_f1": bt.get("f1"), "best_thr": bt.get("threshold"),
    }))
except Exception as e:
    print(json.dumps({"error": str(e)}))
'@
    $tmpPy = Join-Path $env:TEMP ("eg_probe_" + [guid]::NewGuid().ToString('N') + ".py")
    Set-Content -Path $tmpPy -Value $code -Encoding utf8
    $out = & $py $tmpPy $tmp 2>$null
    Remove-Item $tmp, $tmpPy -Force -ErrorAction SilentlyContinue
    if (-not $out) { return $script:CkCache.Data }
    try { $data = $out | ConvertFrom-Json } catch { return $script:CkCache.Data }
    $script:CkCache.Stamp = $stamp
    $script:CkCache.Data = $data
    return $data
}

function Format-Span { param([TimeSpan]$t)
    if ($null -eq $t) { return "n/a" }
    "{0:d2}h {1:d2}m {2:d2}s" -f [int]$t.TotalHours, $t.Minutes, $t.Seconds
}

function Show-Snapshot {
    $log  = Find-Log -Explicit $LogPath
    $proc = Get-TrainProcess
    $rows = Read-EpochRows -Path $log
    $ck   = Read-Checkpoint -Path $Checkpoint

    if (-not $Once) { Clear-Host }
    Write-Host ""
    Write-Host "  EARTH GUARDIAN - training monitor" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
    Write-Host "  ---------------------------------------------------------------"

    # ---- process / elapsed ----
    $elapsed = $null
    if ($proc) {
        $start = [Management.ManagementDateTimeConverter]::ToDateTime($proc.CreationDate)
        $elapsed = (Get-Date) - $start
        Write-Host ("  status      : RUNNING (PID {0}, started {1:HH:mm:ss})" -f $proc.ProcessId, $start) -ForegroundColor Green
    } else {
        Write-Host "  status      : NO TRAINING PROCESS FOUND (finished or stopped)" -ForegroundColor Yellow
    }

    # ---- epoch accounting ----
    $total = $TotalEpochs
    if ($rows.Count -gt 0) { $total = $rows[-1].Total }

    $doneLog = if ($rows.Count) { $rows[-1].Epoch } else { 0 }
    $doneCk  = if ($ck -and $ck.epoch) { [int]$ck.epoch } else { 0 }
    $done    = [Math]::Max($doneLog, $doneCk)

    if ($rows.Count -gt 0) {
        $last = $rows[-1]
        $bestRow = $rows | Sort-Object F1 -Descending | Select-Object -First 1
        $avgSec = ($rows | Measure-Object Sec -Average).Average
        Write-Host ("  epoch       : {0} / {1}   ({2:N1}%)" -f $last.Epoch, $total, (100.0*$last.Epoch/$total))
        Write-Host ("  train loss  : {0:N4}" -f $last.Loss)
        Write-Host ("  val F1      : {0:N4}    IoU {1:N4}   P {2:N4}   R {3:N4}" -f $last.F1, $last.IoU, $last.P, $last.R)
        Write-Host ("  best val F1 : {0:N4}  (epoch {1})" -f $bestRow.F1, $bestRow.Epoch) -ForegroundColor Green
        Write-Host ("  time/epoch  : {0:N0}s  (mean of {1} logged epochs)" -f $avgSec, $rows.Count)
    } else {
        Write-Host "  epoch       : no epoch lines flushed to the log yet" -ForegroundColor DarkYellow
        Write-Host "                (the launcher pipeline buffers ~4KB; checkpoint below is fresher)"
    }

    # ---- checkpoint (fresher than the log) ----
    Write-Host "  ---------------------------------------------------------------"
    if ($ck -and -not $ck.error) {
        $fi = Get-Item $Checkpoint
        Write-Host "  BEST CHECKPOINT (updates the moment val F1 improves)" -ForegroundColor Cyan
        Write-Host ("    epoch     : {0}" -f $ck.epoch)
        Write-Host ("    val F1    : {0:N4}    IoU {1:N4}   P {2:N4}   R {3:N4}" -f $ck.f1, $ck.iou, $ck.precision, $ck.recall)
        Write-Host ("    best-thr  : F1 {0:N4} @ tau {1:N2}" -f $ck.best_thr_f1, $ck.best_thr)
        Write-Host ("    written   : {0:HH:mm:ss}  ({1:N1} min ago)" -f $fi.LastWriteTime, ((Get-Date)-$fi.LastWriteTime).TotalMinutes)
    } else {
        Write-Host "  BEST CHECKPOINT : not written yet (no epoch has completed)" -ForegroundColor DarkYellow
    }

    # ---- ETA ----
    Write-Host "  ---------------------------------------------------------------"
    Write-Host ("  elapsed     : {0}" -f (Format-Span $elapsed))
    if ($done -gt 0 -and $elapsed) {
        $perEpoch = $elapsed.TotalSeconds / $done
        $remain   = [Math]::Max($total - $done, 0)
        $eta      = [TimeSpan]::FromSeconds($perEpoch * $remain)
        Write-Host ("  per epoch   : {0:N0}s  (wall clock / {1} epochs)" -f $perEpoch, $done)
        Write-Host ("  ETA         : {0}   -> approx {1:HH:mm:ss}" -f (Format-Span $eta), ((Get-Date) + $eta)) -ForegroundColor Cyan
        if ($doneCk -gt $doneLog) {
            Write-Host "                (epochs counted from checkpoint = last IMPROVING epoch," -ForegroundColor DarkGray
            Write-Host "                 so ETA is conservative; early stop may end it sooner)" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  ETA         : not computable until one epoch completes" -ForegroundColor DarkYellow
    }

    # ---- GPU ----
    Write-Host "  ---------------------------------------------------------------"
    $smi = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    if ($smi) {
        $p = $smi -split ',' | ForEach-Object { $_.Trim() }
        Write-Host ("  GPU         : {0}% util | {1} / {2} MiB ({3:N0}%) | {4} C | {5} W" -f `
            $p[0], $p[1], $p[2], (100.0*[double]$p[1]/[double]$p[2]), $p[3], $p[4])
        Write-Host "                (util samples instantaneously; dips to 0% between batches are normal)" -ForegroundColor DarkGray
    }

    # ---- recent history ----
    if ($rows.Count -gt 1) {
        Write-Host "  ---------------------------------------------------------------"
        Write-Host "  last epochs:"
        $rows | Select-Object -Last 8 | ForEach-Object {
            Write-Host ("    ep {0,3}  loss {1:N4}  F1 {2:N4}  IoU {3:N4}  {4,4}s" -f $_.Epoch, $_.Loss, $_.F1, $_.IoU, $_.Sec)
        }
    }

    Write-Host ""
    Write-Host ("  log: {0}" -f $log) -ForegroundColor DarkGray
    if (-not $Once) { Write-Host "  Ctrl+C to stop watching (training is unaffected)" -ForegroundColor DarkGray }
    Write-Host ""
}

if ($Once) { Show-Snapshot }
else { while ($true) { Show-Snapshot; Start-Sleep -Seconds $RefreshSeconds } }
