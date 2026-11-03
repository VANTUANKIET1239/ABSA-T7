param([int]$ImageSkip=0, [int]$ImageCount=1, [switch]$UsePrimary)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Drawing
$cache = Join-Path $PSScriptRoot 'analysis_output/brand_enrichment_cache'
$records = Get-Content -LiteralPath (Join-Path $cache 'page_results.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$keys = @('lazada_1183568457','lazada_1390015105','lazada_1349435472','lazada_2580335108','lazada_3107077469','lazada_2130979972','lazada_13444523616','lazada_13490592285','lazada_1240662612','lazada_2138927227','lazada_2024360655','lazada_1603563800','lazada_1764711464','lazada_1478001713','lazada_2136560702','lazada_2978103069')
if ($UsePrimary) { $keys=@('lazada_3179835721','lazada_2138292897','lazada_2138927227','lazada_2133925097','lazada_1360449799','lazada_1595129087','lazada_1890750722','lazada_1366467189','lazada_3012884527','lazada_2212380832','lazada_2563892965','lazada_2970500410','lazada_2661718606','lazada_924984885') }
$suffix=if ($UsePrimary) { 'primary' } else { 'package' }
$imageRecords = @()
foreach ($key in $keys) {
    $record = $records | Where-Object { $_.key -eq $key }
    $images = @([regex]::Matches([string]$record.product_fields.product.desc, '(?is)<img[^>]*src=["'']([^"'']+)') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique | Select-Object -Skip $ImageSkip -First $ImageCount)
    if ($UsePrimary -and $record.ld_product.image) { $images=@($record.ld_product.image | Select-Object -First 1 | ForEach-Object { if ($_.StartsWith('//')) { 'https:'+$_ } else { $_ } }) }
    $imageIndex=$ImageSkip
    foreach ($url in $images) {
        $path = Join-Path $cache ($key + '_' + $suffix + $imageIndex + '.jpg')
        try {
            if (-not (Test-Path -LiteralPath $path)) { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 15 -OutFile $path }
            $imageRecords += [pscustomobject]@{key=($key+'_'+$imageIndex);record_key=$key;url=$url;path=$path}
        } catch { Write-Output ($key + ': ' + $_.Exception.GetBaseException().Message) }
        $imageIndex++
    }
}
$imageRecords | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $cache ('image_sources_'+$suffix+$ImageSkip+'.json')) -Encoding UTF8
$cols=4; $cell=440; $height=[int][Math]::Ceiling($imageRecords.Count / $cols)*$cell
$bitmap=New-Object System.Drawing.Bitmap(($cols*$cell),$height)
$graphics=[Drawing.Graphics]::FromImage($bitmap)
$graphics.Clear([Drawing.Color]::White)
$font=New-Object Drawing.Font('Arial',12)
for ($i=0; $i -lt $imageRecords.Count; $i++) {
    $image=[Drawing.Image]::FromFile($imageRecords[$i].path)
    $x=($i % $cols)*$cell; $y=[int][Math]::Floor($i/$cols)*$cell
    $scale=[Math]::Min(($cell-12)/$image.Width,($cell-40)/$image.Height)
    $w=[int]($image.Width*$scale); $h=[int]($image.Height*$scale)
    $graphics.DrawString($imageRecords[$i].key,$font,[Drawing.Brushes]::Black,$x+5,$y+3)
    $graphics.DrawImage($image,$x+[int](($cell-$w)/2),$y+30,$w,$h)
    $image.Dispose()
}
$bitmap.Save((Join-Path $cache ($suffix+'_contact_sheet'+$ImageSkip+'.jpg')),[Drawing.Imaging.ImageFormat]::Jpeg)
$font.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
Write-Output ('Saved '+$imageRecords.Count+' product images and contact sheet')
