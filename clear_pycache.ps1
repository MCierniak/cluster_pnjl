$current_path = Get-Location
#Get-ChildItem -Path $current_path\src\ -Recurse | Where-Object {$_.Name -Match "__pycache__"} | ForEach-Object { $_.FullName }
Get-ChildItem -Path $current_path\src\ -Recurse | Where-Object {$_.Name -Match "__pycache__"} | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Confirm}