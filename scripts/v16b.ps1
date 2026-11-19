<#[
  Run this ONLY after backing up/pushing V15 to GitHub.
  From thesis/: .\scripts\v16b.ps1 -Action dryrun
  .\scripts\v16b.ps1 -Action init
  .\scripts\v16b.ps1 -Action train -MaxEpisodes 500
  .\scripts\v16b.ps1 -Action eval -Seeds 10
#>
#========================== MODULE 01 - THAM SỐ POWERSHELL ==========================
# Người dùng chọn dryrun / init / train / eval và số episode, seed, cổng.
# Đây là file điều phối V16-B, không chứa PPO hay GNN.

param(
    [ValidateSet('dryrun','init','train','eval')][string]$Action = 'dryrun',
    [int]$MaxEpisodes = 500,
    [ValidateSet('joint12','joint16')][string]$WarmstartPhase = 'joint12',
    [int]$WarmstartEp = 850,
    [int]$Seeds = 10,
    [int]$FirstSeed = 900001,
    [int]$Port = 51890
)

#====================== KẾT THÚC MODULE 01 - THAM SỐ POWERSHELL ======================



#======================= MODULE 02 - ĐỊNH VỊ PYTHON VÀ CÁC RUN =======================
# Xác định thư mục thesis, .venv311, V15 checkpoint nguồn và V16-B output.

$ErrorActionPreference = 'Stop'
$thesis = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $thesis
$py = Join-Path $thesis '.venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Python not found: $py" }
$v15Root = Join-Path $thesis 'mappo_runs\V15_FULLROLLOUT_GA512'
$v16Root = Join-Path $thesis 'mappo_runs\V16B_TRAFFIC_EDGE_GATV2_GA512'
$reference = Join-Path $v15Root 'joint16'
if (-not (Test-Path (Join-Path $reference 'experiment.json'))) {
  throw 'Missing V15/joint16/experiment.json; configure reference path in script.'
}
$meta = Get-Content (Join-Path $reference 'experiment.json') -Raw | ConvertFrom-Json
$bp = (Resolve-Path (Join-Path $reference 'run_config_blueprint.json')).Path

#================= MODULE CON 02A - TẠO DANH SÁCH THAM SỐ DÙNG CHUNG =================
# Dùng mảng @() để truyền đúng options cho Python qua PowerShell.

$common = @(
  '--phase','joint16', '--output-root',$v16Root,
  '--blueprint',$bp, '--model',"$($meta.model_file)",
  '--device',"$($meta.device)", '--seed',"$($meta.base_seed)",
  '--max-events',"$($meta.max_events)",
  '--microbatch-size',"$($meta.ppo_microbatch_size)",
  '--cpu-threads',"$($meta.cpu_threads)",
  '--snapshot-every',"$($meta.snapshot_every)",
  '--port',"$Port"
)

#================== KẾT THÚC MODULE 02 - ĐỊNH VỊ PYTHON VÀ CÁC RUN ==================



#========================= MODULE 03 - CHỌN CHẾ ĐỘ HOẠT ĐỘNG =========================
# dryrun: kiểm tra không train; init: tạo EP0 từ V15.
# train: --resume; eval: chạy held-out checkpoint V16-B.

$runner = Join-Path $thesis 'scripts\train_v16b_fullrollout.py'

#=== PHẦN 03A: DRYRUN HOẶC INIT (TÌM SNAPSHOT V15) ===
if ($Action -eq 'dryrun' -or $Action -eq 'init') {
  $src = Join-Path $v15Root $WarmstartPhase
  $file = ('snapshot_ep_{0:D4}.pt' -f $WarmstartEp)
  $ck = Join-Path $src $file
  if (-not (Test-Path $ck)) { throw "V15 warmstart checkpoint missing: $ck" }
  $ck = (Resolve-Path $ck).Path
  Write-Host "V16B source: $ck"
  Write-Host "V16B output root: $v16Root"
  if ($Action -eq 'dryrun') {
    & $py -u $runner @common '--init-checkpoint' $ck '--max-episodes' '0' '--dry-run'
  } else {
    if (Test-Path (Join-Path $v16Root 'joint16\state.json')) {
      throw 'V16B Joint16 already initialized. Use -Action train, not init.'
    }
    & $py -u $runner @common '--init-checkpoint' $ck '--max-episodes' '0' '--init-only'
  }

#=== PHẦN 03B: RESUME TRAINING V16-B ===
} elseif ($Action -eq 'train') {
  if (-not (Test-Path (Join-Path $v16Root 'joint16\state.json'))) {
    throw 'Initialize first: .\scripts\v16b.ps1 -Action init'
  }
  & $py -u $runner @common '--max-episodes' "$MaxEpisodes" '--resume'

#=== PHẦN 03C: HELD-OUT EVALUATION ===
} else {
  $eval = Join-Path $thesis 'scripts\evaluate_v16b_traffic.py'
  $evalargs = @('--phase','joint16','--output-root',$v16Root,
    '--device',"$($meta.device)",'--port',"$Port",'--seeds',"$Seeds",
    '--first-seed',"$FirstSeed")
  & $py -u $eval @evalargs
}

#==================== KẾT THÚC MODULE 03 - CHỌN CHẾ ĐỘ HOẠT ĐỘNG ====================



#======================= MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON =======================
# Nếu chương trình Python báo lỗi, PowerShell ném exception và không im lặng bỏ qua.

if ($LASTEXITCODE -ne 0) { throw "V16B $Action failed, exit code $LASTEXITCODE" }

#=================== KẾT THÚC MODULE 04 - KIỂM TRA MÃ THOÁT PYTHON ===================
