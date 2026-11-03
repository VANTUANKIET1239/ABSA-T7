param(
    [string]$InputCsv = "$PSScriptRoot/data/combined_reviews.csv",
    [string]$CacheDir = "$PSScriptRoot/analysis_output/brand_enrichment_cache",
    [switch]$RetryIncomplete,
    [string[]]$RefreshKeys = @()
)
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Net.Http
$null = New-Item -ItemType Directory -Path $CacheDir -Force
$rows = @(Import-Csv -LiteralPath $InputCsv -Encoding UTF8)
$groups = @($rows | Where-Object { [string]::IsNullOrWhiteSpace($_.brand) } | Group-Object product_url | Sort-Object { $_.Group[0].product_name })
$handler = New-Object System.Net.Http.HttpClientHandler
$handler.AutomaticDecompression = [Net.DecompressionMethods]::GZip -bor [Net.DecompressionMethods]::Deflate
$client = New-Object System.Net.Http.HttpClient($handler)
$client.Timeout = [TimeSpan]::FromSeconds(20)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36')
$results = New-Object 'System.Collections.Generic.List[object]'
try {
    for ($offset = 0; $offset -lt $groups.Count; $offset += 4) {
        $pending = @()
        for ($i = $offset; $i -lt [Math]::Min($offset + 4, $groups.Count); $i++) {
            $group = $groups[$i]
            $url = $group.Name
            $key = if ($url -match 'pdp-i(\d+)') { 'lazada_' + $Matches[1] } elseif ($url -match '-i\.(\d+)\.(\d+)') { 'shopee_' + $Matches[1] + '_' + $Matches[2] } else { throw "Unrecognized product URL: $url" }
            $cache = Join-Path $CacheDir ($key + '.json')
            $cached = if (Test-Path -LiteralPath $cache) { Get-Content -LiteralPath $cache -Raw -Encoding UTF8 | ConvertFrom-Json } else { $null }
            $retry = $cached -and $RetryIncomplete -and $key.StartsWith('lazada_') -and $cached.fetch_status -eq 'no_product_data'
            if ($cached -and -not $retry -and $key -notin $RefreshKeys) {
                $results.Add($cached)
            } else {
                $pending += [pscustomobject]@{ group=$group; key=$key; cache=$cache; task=$client.GetAsync($url) }
            }
        }
        foreach ($job in $pending) {
            $record = [ordered]@{ key=$job.key; product_url=$job.group.Name; product_name=$job.group.Group[0].product_name; review_count=$job.group.Count; platform=$job.group.Group[0].platform; fetched_at=[DateTime]::UtcNow.ToString('o'); http_status=$null; fetch_status=''; page_title=''; direct_brand=''; product_fields=$null; ld_product=$null; error='' }
            try {
                $response = $job.task.GetAwaiter().GetResult()
                $record.http_status = [int]$response.StatusCode
                $html = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                $record.fetch_status = 'http_' + [int]$response.StatusCode
                if ($response.IsSuccessStatusCode) {
                    $title = [regex]::Match($html, '(?is)<title[^>]*>(.*?)</title>')
                    $record.page_title = [Net.WebUtility]::HtmlDecode($title.Groups[1].Value).Trim()
                    $moduleMatch = [regex]::Match($html, '(?m)^\s*var __moduleData__\s*=\s*(\{[^\r\n]*\})\s*;\s*$')
                    if ($moduleMatch.Success) {
                        $module = $moduleMatch.Groups[1].Value | ConvertFrom-Json
                        $fields = $module.data.root.fields
                        $record.product_fields = [ordered]@{ product=$fields.product; specifications=$fields.specifications; skuInfos=$null }
                        if ($fields.product.brand.name) { $record.direct_brand = [string]$fields.product.brand.name }
                        elseif ($fields.product.brandName) { $record.direct_brand = [string]$fields.product.brandName }
                        elseif ($fields.tracking.extraInfo.brand_name) { $record.direct_brand = [string]$fields.tracking.extraInfo.brand_name }
                        $record.fetch_status = 'product_data_loaded'
                    }
                    foreach ($match in [regex]::Matches($html, '(?is)<script[^>]*type=["'']application/ld\+json["''][^>]*>(.*?)</script>')) {
                        try {
                            $ld = $match.Groups[1].Value | ConvertFrom-Json
                            if ($ld.'@type' -eq 'Product') {
                                $record.ld_product=$ld
                                if (-not $record.product_fields) { $record.product_fields = [ordered]@{ product=$ld; specifications=$null; skuInfos=$null } }
                                if (-not $record.direct_brand -and $ld.brand.name) { $record.direct_brand = [string]$ld.brand.name }
                                $record.fetch_status = 'product_data_loaded'
                            }
                        } catch { }
                    }
                    if ($record.fetch_status -ne 'product_data_loaded') { $record.fetch_status = 'no_product_data' }
                }
                $response.Dispose()
            } catch {
                $record.fetch_status = 'request_failed'
                $record.error = $_.Exception.GetBaseException().Message
            }
            $record | ConvertTo-Json -Depth 35 | Set-Content -LiteralPath $job.cache -Encoding UTF8
            $results.Add([pscustomobject]$record)
        }
        $results | ConvertTo-Json -Depth 35 | Set-Content -LiteralPath (Join-Path $CacheDir 'page_results.json') -Encoding UTF8
        Write-Output ('Processed {0}/{1}; product pages {2}; brands {3}' -f $results.Count,$groups.Count,@($results | Where-Object { $_.fetch_status -eq 'product_data_loaded' }).Count,@($results | Where-Object { $_.direct_brand }).Count)
        if ($pending.Count) { Start-Sleep -Milliseconds 700 }
    }
} finally { $client.Dispose(); $handler.Dispose() }
$results | Group-Object platform,fetch_status | Select-Object Name,Count | ConvertTo-Json
