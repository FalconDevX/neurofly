# Osadza fonty (Roboto) w NeuroFly.pptx przez PowerPoint, żeby prezentacja wyglądała tak samo na komputerach bez Roboto.
#   powershell -ExecutionPolicy Bypass -File docs\prezentacja\embed_fonts.ps1 [ścieżka.pptx]
# Wymaga PowerPointa i zainstalowanych fontów: Roboto (Apache 2.0, github.com/googlefonts/roboto) i Michroma (OFL, Google Fonts).
param([string]$Path = (Join-Path $PSScriptRoot "NeuroFly.pptx"))
$Path = (Resolve-Path $Path).Path
$tmp = [System.IO.Path]::ChangeExtension($Path, ".embedded.pptx")
$pp = New-Object -ComObject PowerPoint.Application
try {
    $p = $pp.Presentations.Open($Path, $false, $false, $false)
    $p.SaveAs($tmp, 24, -1)   # 24 = ppSaveAsOpenXMLPresentation, -1 = msoTrue: EmbedTrueTypeFonts
    $p.Close()
} finally { $pp.Quit() }
Move-Item -Force $tmp $Path
"osadzono fonty: $Path"
