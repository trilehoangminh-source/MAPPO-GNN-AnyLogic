<#
  V17-B / Joint16: khởi tạo mới từ V15 Joint16 EP0 (7 features + message).
  Nguồn DUY NHẤT: mappo_runs/V15_FULLROLLOUT_GA512/joint16/initial_ep_0000.pt
  Sau khi giải nén đè vào thư mục thesis, chạy trong PowerShell:
    .\scripts\v16b.ps1 -Action dryrun
    .\scripts\v16b.ps1 -Action init
    .\scripts\v16b.ps1 -Action train -MaxEpisodes 1000

  CHÚ Ý: DỪNG training V16-B trước khi giải nén code V17-B; V17-B output tách biệt.
  -Action init TỪ CHỐI khi thư mục V17-B đã có dữ liệu.
  Không đụng đến checkpoint/metrics V16-B hay V15.
  -Action train từ chối resume run có checkpoint nguồn khác Joint16 EP0.
  Routing đổi thành 7 Edge Features + edge-conditioned Attention/Message;
  giữ nguyên PPO, Reward, Dispatch, Java, Routing action mask=OFF.
#>
#========================== MODULE 01 - THAM SỐ POWERSHELL ==========================
# Script entrypoint GỐC v16b.ps1 được dùng cho V17-B (không tạo script mới).

param(
    [ValidateSet('dryrun','init','train','eval')][string]$Action = 'dryrun',
    [int]$MaxEpisodes = 500,
    [int]$Seeds = 10,
    [int]$FirstSeed = 900001,
    [int]$Port = 51890
)

#====================== KẾT THÚC MODULE 01 - THAM SỐ POWERSHELL ======================


#======================= MODULE 02 - ĐỊNH VỊ PYTHON VÀ CÁC RUN =======================
# Nguồn cố định V15 JOINT16 EP0; V17-B luôn ghi output riêng, không đụng V16-B.

$ErrorActionPreference = 'Stop'
$thesis = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $thesis
$py = Join-Path $thesis '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw "Python not found: $py" }

$v15Root = Join-Path $thesis 'mappo_runs\V15_FULLROLLOUT_GA512'
$v17Root = Join-Path $thesis 'mappo_runs\V17B_EDGE7_MSG_GA512'
$reference = Join-Path $v15Root 'joint16'
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
  '--phase','joint16', '--output-root',$v17Root,
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
        throw 'V17-B has not been initialized. Run .\scripts\v16b.ps1 -Action init first.'
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
$runner = Join-Path $thesis 'scripts\train_v16b_fullrollout.py'

if ($Action -eq 'dryrun') {
    Write-Host "[V17B JOINT16 EP0 DRYRUN] Source: $sourceCheckpoint"
    Write-Host "[V17B JOINT16 EP0 DRYRUN] SHA256: $sourceHash"
    & $py -u $runner @common '--init-checkpoint' $sourceCheckpoint '--max-episodes' '0' '--dry-run'

} elseif ($Action -eq 'init') {
    # Chặn chạy init lần thứ hai để không mất checkpoint V17-B.
    Write-Host "[V17B JOINT16 EP0 INIT] Source: $sourceCheckpoint"
    Write-Host "[V17B JOINT16 EP0 INIT] SHA256: $sourceHash"
    if (Test-Path -LiteralPath $v17Root) {
        $existing = @(Get-ChildItem -LiteralPath $v17Root -Force)
        if ($existing.Count -gt 0) {
            throw "V17-B output already exists. Do NOT re-init or overwrite: $v17Root. Use -Action train."
        }
    }
    & $py -u $runner @common '--init-checkpoint' $sourceCheckpoint '--max-episodes' '0' '--init-only'

} elseif ($Action -eq 'train') {
    Assert-CorrectRunSource -RunRoot $v17Root -ExpectedSource $sourceCheckpoint -ExpectedHash $sourceHash
    & $py -u $runner @common '--max-episodes' "$MaxEpisodes" '--resume'

} else {
    Assert-CorrectRunSource -RunRoot $v17Root -ExpectedSource $sourceCheckpoint -ExpectedHash $sourceHash
    $eval = Join-Path $thesis 'scripts\evaluate_v16b_traffic.py'
    $evalargs = @('--phase','joint16','--output-root',$v17Root,
      '--device',"$($meta.device)",'--port',"$Port",'--seeds',"$Seeds",
      '--first-seed',"$FirstSeed")
    & $py -u $eval @evalargs
}

#======================= MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON =======================
if ($LASTEXITCODE -ne 0) { throw "V17B $Action failed, exit code $LASTEXITCODE" }
#=================== KẾT THÚC MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON ===================
