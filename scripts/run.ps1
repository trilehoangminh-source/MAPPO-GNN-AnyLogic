<#
  Khởi tạo run Joint16 mới từ checkpoint V15 Joint16 EP0.
  Run mới lưu tại mappo_runs/TRAFFIC_EDGE_MESSAGE_GA512.
  Lệnh PowerShell:
    .\scripts\run.ps1 -Action dryrun
    .\scripts\run.ps1 -Action init
    .\scripts\run.ps1 -Action train -MaxEpisodes 1000
    .\scripts\run.ps1 -Action eval -Seeds 20

  -Action init từ chối ghi đè run đã có dữ liệu.
  -Action train/eval xác minh checkpoint nguồn và metadata.
  Routing dùng bảy edge features trong attention và message; không áp action mask.
#>
#========================== MODULE 01 - THAM SỐ POWERSHELL ==========================
# Entrypoint dùng chung cho khởi tạo, huấn luyện và đánh giá run hiện hành.

param(
    [ValidateSet('dryrun','init','train','eval')][string]$Action = 'dryrun',
    [int]$MaxEpisodes = 500,
    [int]$Seeds = 10,
    [int]$FirstSeed = 900001,
    [int]$Port = 51890
)

#====================== KẾT THÚC MODULE 01 - THAM SỐ POWERSHELL ======================


#======================= MODULE 02 - ĐỊNH VỊ PYTHON VÀ CÁC RUN =======================
# Đọc cấu hình và checkpoint khởi tạo, ghi run mới vào thư mục riêng.

$ErrorActionPreference = 'Stop'
$thesis = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $thesis
$py = Join-Path $thesis '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw "Python not found: $py" }

$warmstartRoot = Join-Path $thesis 'mappo_runs\V15_FULLROLLOUT_GA512'
$runRoot = Join-Path $thesis 'mappo_runs\TRAFFIC_EDGE_MESSAGE_GA512'
$reference = Join-Path $warmstartRoot 'joint16'
$refMetaFile = Join-Path $reference 'experiment.json'
if (-not (Test-Path -LiteralPath $refMetaFile)) {
  throw 'Missing V15/joint16/experiment.json; cannot derive source run settings.'
}
$meta = Get-Content -LiteralPath $refMetaFile -Raw | ConvertFrom-Json
$bp = (Resolve-Path (Join-Path $reference 'run_config_blueprint.json')).Path

# IMPORTANT: EP0 is joint16's saved initial state, not snapshot_ep_0850.pt.
$sourceCheckpoint = Join-Path $reference 'initial_ep_0000.pt'
if (-not (Test-Path -LiteralPath $sourceCheckpoint)) {
  throw "Missing V15 Joint16 EP0 checkpoint: $sourceCheckpoint"
}
$sourceCheckpoint = (Resolve-Path -LiteralPath $sourceCheckpoint).Path
$sourceHash = (Get-FileHash -LiteralPath $sourceCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()

#================= MODULE CON 02A - TẠO DANH SÁCH THAM SỐ DÙNG CHUNG =================
$common = @(
  '--phase','joint16', '--output-root',$runRoot,
  '--blueprint',$bp, '--model',"$($meta.model_file)",
  '--device',"$($meta.device)", '--seed',"$($meta.base_seed)",
  '--max-events',"$($meta.max_events)",
  '--microbatch-size',"$($meta.ppo_microbatch_size)",
  '--cpu-threads',"$($meta.cpu_threads)",
  '--snapshot-every',"$($meta.snapshot_every)",
  '--port',"$Port"
)

#================== MODULE CON 02B - KIỂM TRA ĐÚNG CHECKPOINT NGUỒN =================
# -Action train chỉ resume được run V17-B bắt đầu từ đúng V15 Joint16 EP0.
function Assert-CorrectRunSource {
    param([string]$RunRoot, [string]$ExpectedSource, [string]$ExpectedHash)

    $expFile = Join-Path $RunRoot 'joint16\experiment.json'
    $stateFile = Join-Path $RunRoot 'joint16\state.json'
    if (-not (Test-Path -LiteralPath $expFile) -or -not (Test-Path -LiteralPath $stateFile)) {
        throw 'Run has not been initialized. Run .\scripts\run.ps1 -Action init first.'
    }

    $runExp = Get-Content -LiteralPath $expFile -Raw | ConvertFrom-Json
    $runState = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    $recordedSource = [string]$runExp.source_checkpoint
    $recordedHash = [string]$runExp.source_checkpoint_sha256
    $stateSource = [string]$runState.source_checkpoint

    if (($recordedSource -ne $ExpectedSource) -or
        ($stateSource -ne $ExpectedSource) -or
        ($recordedHash.ToLowerInvariant() -ne $ExpectedHash)) {
        throw @'
STOP: V17-B run hien tai KHONG khoi tao tu V15 Joint16 initial_ep_0000.pt.
KHONG ghi de run cu. Kiem tra experiment.json/state.json, sau do xu ly thu cong.
Khong tu dong khoi tao lai neu dang co checkpoint.
'@
    }
}

#========================= MODULE 03 - CHỌN CHẾ ĐỘ HOẠT ĐỘNG =========================
$runner = Join-Path $thesis 'scripts\train.py'

if ($Action -eq 'dryrun') {
    Write-Host "[RUN JOINT16 DRYRUN] Source: $sourceCheckpoint"
    Write-Host "[RUN JOINT16 DRYRUN] SHA256: $sourceHash"
    & $py -u $runner @common '--init-checkpoint' $sourceCheckpoint '--max-episodes' '0' '--dry-run'

} elseif ($Action -eq 'init') {
    # Chặn chạy init lần thứ hai để không mất checkpoint V17-B.
    Write-Host "[RUN JOINT16 INIT] Source: $sourceCheckpoint"
    Write-Host "[RUN JOINT16 INIT] SHA256: $sourceHash"
    if (Test-Path -LiteralPath $runRoot) {
        $existing = @(Get-ChildItem -LiteralPath $runRoot -Force)
        if ($existing.Count -gt 0) {
            throw "V17-B output already exists. Do NOT re-init or overwrite: $runRoot. Use -Action train."
        }
    }
    & $py -u $runner @common '--init-checkpoint' $sourceCheckpoint '--max-episodes' '0' '--init-only'

} elseif ($Action -eq 'train') {
    Assert-CorrectRunSource -RunRoot $runRoot -ExpectedSource $sourceCheckpoint -ExpectedHash $sourceHash
    & $py -u $runner @common '--max-episodes' "$MaxEpisodes" '--resume'

} else {
    Assert-CorrectRunSource -RunRoot $runRoot -ExpectedSource $sourceCheckpoint -ExpectedHash $sourceHash
    $eval = Join-Path $thesis 'scripts\evaluate.py'
    $evalargs = @('--phase','joint16','--output-root',$runRoot,
      '--device',"$($meta.device)",'--port',"$Port",'--seeds',"$Seeds",
      '--first-seed',"$FirstSeed")
    & $py -u $eval @evalargs
}

#======================= MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON =======================
if ($LASTEXITCODE -ne 0) { throw "Run $Action failed, exit code $LASTEXITCODE" }
#=================== KẾT THÚC MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON ===================
