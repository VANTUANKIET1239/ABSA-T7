param(
    [string]$InputCsv = "$PSScriptRoot/data/combined_reviews.csv",
    [string]$OutputCsv = "$PSScriptRoot/data/combined_reviews_brand_filled.csv",
    [string]$RulesFile = "$PSScriptRoot/analysis_output/brand_enrichment_rules.json",
    [string]$PageResults = "$PSScriptRoot/analysis_output/brand_enrichment_cache/page_results.json",
    [string]$ReportDir = "$PSScriptRoot/analysis_output/brand_enrichment",
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if ([IO.Path]::GetFullPath($InputCsv) -eq [IO.Path]::GetFullPath($OutputCsv)) { throw 'Output must be a separate CSV.' }
$sourceHash=(Get-FileHash -LiteralPath $InputCsv -Algorithm SHA256).Hash
$rows=@(Import-Csv -LiteralPath $InputCsv -Encoding UTF8)
$columns=@($rows[0].PSObject.Properties.Name)
$config=Get-Content -LiteralPath $RulesFile -Raw -Encoding UTF8 | ConvertFrom-Json
$pages=Get-Content -LiteralPath $PageResults -Raw -Encoding UTF8 | ConvertFrom-Json
$groups=@($rows | Where-Object { [string]::IsNullOrWhiteSpace($_.brand) } | Group-Object product_url)
$audit=@()
foreach ($group in $groups) {
    $page=@($pages | Where-Object { $_.product_url -ceq $group.Name })
    if ($page.Count -ne 1) { throw ('Missing or duplicated page evidence: '+$group.Name) }
    $page=$page[0]
    $override=@($config.overrides | Where-Object { $_.key -ceq $page.key })
    $unresolved=@($config.unresolved | Where-Object { $_.key -ceq $page.key })
    if ($override.Count -gt 1 -or $unresolved.Count -gt 1 -or ($override.Count -and $unresolved.Count)) { throw ('Conflicting rules for '+$page.key) }
    $brand=''; $method=''; $evidence=''; $sourceUrl=''; $sourceType=''; $imageUrl=''; $note=''; $reason=''
    if ($unresolved.Count) { $reason=$unresolved[0].reason }
    elseif ($override.Count) {
        $item=$override[0]; $brand=$item.brand; $method=$item.method; $evidence=$item.evidence; $sourceUrl=$item.source_url; $imageUrl=$item.image_url; $note=$item.note
        if ($sourceUrl) { $sourceType='manual_review_supporting_source' }
    } else {
        # Rules are ordered: specific names precede their broader families.
        $rule=$config.rules | Where-Object { $group.Group[0].product_name -match $_.pattern } | Select-Object -First 1
        if ($rule) {
            $brand=$rule.brand; $sourceUrl=$rule.source_url; $sourceType=$rule.source_type; $note=$rule.note
            if ($page.fetch_status -eq 'product_data_loaded') {
                if ($page.product_fields.product.title -notmatch $rule.pattern) { throw ('Live title does not support the selected brand: '+$page.key) }
                $method='listing_title'; $evidence=$page.product_fields.product.title
            } elseif ($sourceUrl) {
                $method='csv_title_and_external_source'; $evidence=$group.Group[0].product_name
            } else {
                $existing=$rows | Where-Object { $_.brand -ieq $brand -and $_.product_name -match $rule.pattern } | Select-Object -First 1
                if ($existing) {
                    $method='csv_title_and_existing_dataset_brand'; $evidence=$group.Group[0].product_name
                    $sourceUrl=$existing.product_url; $sourceType='existing_dataset_product_reference'
                } else { $brand=''; $reason='Trang không đọc được và chưa có nguồn đối chiếu đủ chắc.' }
            }
        } else { $reason='Chưa có bằng chứng xác định brand; không suy đoán từ thành phần hoặc tên shop.' }
    }
    $audit += [pscustomobject][ordered]@{
        product_key=$page.key; platform=$page.platform; product_name=$group.Group[0].product_name; product_url=$group.Name
        affected_review_count=$group.Count; brand=$brand; status=$(if ($brand) {'filled'} else {'unresolved'})
        method=$method; evidence=$evidence; listing_fetch_status=$page.fetch_status; listing_http_status=$page.http_status
        listing_fetched_at_utc=$page.fetched_at; listing_page_title=$page.page_title
        supporting_source_url=$sourceUrl; supporting_source_type=$sourceType; package_image_url=$imageUrl
        note=$note; unresolved_reason=$reason
    }
}
$filled=@($audit | Where-Object { $_.status -eq 'filled' })
$remaining=@($audit | Where-Object { $_.status -eq 'unresolved' })
$filledRows=($filled | Measure-Object affected_review_count -Sum).Sum
$missingRows=($audit | Measure-Object affected_review_count -Sum).Sum
$summary=[ordered]@{
    total_rows=$rows.Count; missing_brand_before=$missingRows; filled_rows=$filledRows
    missing_brand_after=($missingRows-$filledRows); reviewed_products=$audit.Count
    filled_products=$filled.Count; unresolved_products=$remaining.Count; input_sha256=$sourceHash
    original_file_unchanged=$true; checked_at_utc=[DateTime]::UtcNow.ToString('o')
    note='Brand inferred from the recorded listing evidence or matching external sources; not every value came from a structured brand field. Existing nonblank brands are preserved.'
}
if ($DryRun) {
    $summary | ConvertTo-Json
    $remaining | Select-Object product_key,affected_review_count,product_name,unresolved_reason | ConvertTo-Json
    exit
}
$map=@{}; foreach ($item in $filled) { $map[$item.product_url]=$item.brand }
$changed=0
foreach ($row in $rows) {
    if ([string]::IsNullOrWhiteSpace($row.brand) -and $map.ContainsKey($row.product_url)) { $row.brand=$map[$row.product_url]; $changed++ }
}
if ($changed -ne $filledRows) { throw 'Filled-row count does not reconcile.' }
$null=New-Item -ItemType Directory -Path $ReportDir -Force
$rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
$audit | Export-Csv -LiteralPath (Join-Path $ReportDir 'brand_audit.csv') -NoTypeInformation -Encoding UTF8
$remaining | Export-Csv -LiteralPath (Join-Path $ReportDir 'unresolved_products.csv') -NoTypeInformation -Encoding UTF8
# Reopen both CSVs and compare every cell, preserving row order and all non-brand values.
$before=@(Import-Csv -LiteralPath $InputCsv -Encoding UTF8)
$after=@(Import-Csv -LiteralPath $OutputCsv -Encoding UTF8)
if ($before.Count -ne $after.Count -or (@($after[0].PSObject.Properties.Name) -join '|') -cne ($columns -join '|')) { throw 'CSV dimensions or headers changed.' }
$checkedChanges=0
for ($i=0; $i -lt $before.Count; $i++) {
    foreach ($column in $columns) {
        if ($before[$i].$column -cne $after[$i].$column) {
            if ($column -cne 'brand' -or -not [string]::IsNullOrWhiteSpace($before[$i].brand)) { throw ('Unexpected change at row '+($i+2)+', column '+$column) }
            if ($after[$i].brand -cne $map[$before[$i].product_url]) { throw 'Brand does not match reviewed mapping.' }
            $checkedChanges++
        }
    }
}
if ($checkedChanges -ne $filledRows) { throw 'Saved CSV change count does not reconcile.' }
if ((Get-FileHash -LiteralPath $InputCsv -Algorithm SHA256).Hash -cne $sourceHash) { throw 'Original file changed during enrichment.' }
$actualMissing=@($after | Where-Object { [string]::IsNullOrWhiteSpace($_.brand) }).Count
if ($actualMissing -ne $summary.missing_brand_after) { throw 'Remaining missing brand count does not reconcile.' }
$summary['verified_changed_cells']=$checkedChanges
$summary['all_non_brand_cells_preserved']=$true
$summary['existing_brands_preserved']=$true
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $ReportDir 'summary.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
