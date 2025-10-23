# Usage: .\bin\ffmpeg_prep.ps1 input.MOV out.mp4 60
param(
  [Parameter(Mandatory=$true)][string]$In,
  [string]$Out = "videos\plasma_fixed.mp4",
  [int]$Fps = 60
)
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null
ffmpeg -y -i $In `
  -vf "zscale=t=linear,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709" `
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 `
  -r $Fps -vsync cfr -c:v libx264 -crf 18 -pix_fmt yuv420p `
  $Out
Write-Host "Wrote $Out"
