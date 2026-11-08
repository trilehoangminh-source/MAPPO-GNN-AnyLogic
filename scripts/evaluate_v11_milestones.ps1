# Chay tai thu muc goc C:\MARL_AnyLogic\MAPPO_NEW\thesis.
# Script KHONG train, KHONG sua checkpoint, KHONG ghi de ket qua cu.
param(
    [switch]$WithMilestones,
    [string]$Device = 'cuda:0',
    [int]$Port = 51881
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$V11Run = Join-Path $Root 'mappo_runs\V11_CONGESTION_FROM_V8_EP304'
$Out = Join-Path $V11Run 'eval_5seeds'
$StateFile = Join-Path $V11Run 'state.json'
$MetaFile = Join-Path $V11Run 'experiment.json'
$V11Script = Join-Path $Root 'scripts\evaluate_v11_congestion.py'
$V8Script = Join-Path $Root 'scripts\evaluate_V8_jobdone2.py'
$Python = Join-Path $Root '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
foreach ($required in @($StateFile, $MetaFile, $V11Script, $V8Script)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Thieu file: $required" }
}
$State = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
$Meta = Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json
$Episode = [int]$State.last_episode
$Source = [string]$Meta.source_checkpoint
$Current = [string]$State.checkpoint
if (-not (Test-Path -LiteralPath $Source)) { throw "Khong tim thay V8 EP304 source: $Source" }
if (-not (Test-Path -LiteralPath $Current)) { throw "Khong tim thay V11 checkpoint: $Current" }
if ([int]$Meta.source_episode -ne 304) { throw "V11 source khong phai EP304; can kiem tra lai." }
if ($Episode -le 304) { throw "Chua co V11 checkpoint da train." }
New-Item -ItemType Directory -Path $Out -Force | Out-Null
$Seeds = @(900001,900002,900003,900004,900005)
Write-Host "[EVAL INFO] V11 committed EP$Episode; training weights KHONG bi thay doi."
Write-Host "[EVAL INFO] V8 source: $Source"
Write-Host "[EVAL INFO] 5 seeds: $($Seeds -join ', ') | 1800 simulated seconds"
Write-Host "[EVAL INFO] Dung tat ca tien trinh TRAIN truoc khi bat dau, va bao dam port $Port ranh."

function Run-Once([string]$Script, [string[]]$EvalArgs, [string]$ResultPath) {
    if (Test-Path -LiteralPath $ResultPath) {
        Write-Host "[SKIP EXISTING] $ResultPath" -ForegroundColor Yellow
        return
    }
    Write-Host "[EVAL START] $ResultPath" -ForegroundColor Cyan
    & $Python $Script @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation that bai, code=$LASTEXITCODE; $ResultPath" }
}

Push-Location $Root
try {
    $V8Out = Join-Path $Out 'heldout_V8_source_EP304_5seeds.jsonl'
    # V8 source checksum se duoc V11 evaluator kiem tra tu experiment.json.
    $v8Args = @('--checkpoint', $Source, '--device', $Device, '--port', "$Port", '--sim-seconds', '1800',
        '--guide-strength', '1.5', '--output', $V8Out, '--seeds') + @($Seeds | ForEach-Object { "$_" })
    Run-Once $V8Script $v8Args $V8Out

    $cps = @()
    if ($WithMilestones) {
        foreach ($ep in @(424, 464, 504, 524)) {
            $snap = Join-Path $V11Run ('snapshot_ep_{0:D4}.pt' -f $ep)
            if (Test-Path -LiteralPath $snap) {
                $cps += @{ episode = $ep; path = $snap }
            } else {
                Write-Host "[SKIP MISSING] snapshot EP$ep" -ForegroundColor Yellow
            }
        }
    }
    # Dong bang ban checkpoint hien tai de tranh thay doi rolling_slot khi resume ve sau.
    $Frozen = Join-Path $V11Run ('eval_frozen_ep_{0:D4}.pt' -f $Episode)
    if (-not (Test-Path -LiteralPath $Frozen)) {
        Copy-Item -LiteralPath $Current -Destination $Frozen -ErrorAction Stop
        Write-Host "[FROZEN] Da luu ban doc lap: $Frozen"
    } else {
        Write-Host "[FROZEN EXISTS] $Frozen (khong ghi de)"
    }
    $cps += @{ episode = $Episode; path = $Frozen }
    foreach ($cp in $cps) {
        $V11Out = Join-Path $Out ('heldout_V11_EP{0:D4}_5seeds.jsonl' -f [int]$cp.episode)
        $v11Args = @('--run', $V11Run, '--checkpoint', [string]$cp.path, '--device', $Device,
            '--port', "$Port", '--sim-seconds', '1800', '--output', $V11Out, '--seeds') + @($Seeds | ForEach-Object { "$_" })
        Run-Once $V11Script $v11Args $V11Out
    }
    Write-Host "[DONE] Ket qua: $Out"
    $Compare = Join-Path $Root 'scripts\compare_v8_v11_heldout.py'
    if (Test-Path -LiteralPath $Compare) {
        & $Python $Compare --directory $Out
        if ($LASTEXITCODE -ne 0) { throw 'Compare metrics that bai' }
    }
} finally {
    Pop-Location
}
