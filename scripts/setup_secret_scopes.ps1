# scripts/setup_secret_scopes.ps1
# Creates Databricks secret scopes and populates secrets for each environment.
# Run once per workspace after initial provisioning.
#
# Usage (from repo root in PowerShell):
#   .\scripts\setup_secret_scopes.ps1 -Env dev
#   .\scripts\setup_secret_scopes.ps1 -Env preprod
#   .\scripts\setup_secret_scopes.ps1 -Env prod

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev","preprod","prod")]
    [string]$Env
)

switch ($Env) {
    "dev"     { $Scope = "scf-cohort-dev";     $Profile = "dbc-a2b9471c-0cac" }
    "preprod" { $Scope = "scf-cohort-preprod"; $Profile = "dbc-preprod" }
    "prod"    { $Scope = "scf-cohort-prod";    $Profile = "dbc-prod" }
}

Write-Host ("Setting up secret scope " + $Scope + " for " + $Env) -ForegroundColor Cyan

# Create scope - idempotent, ignore error if already exists
$result = databricks secrets create-scope $Scope --profile $Profile 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ("Scope already exists, continuing: " + $Scope) -ForegroundColor Yellow
}

# SMTP credentials
Write-Host ""
$SmtpUser = Read-Host "  smtp-user"
$SmtpPassSecure = Read-Host "  smtp-password" -AsSecureString
$SmtpPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SmtpPassSecure)
)

databricks secrets put-secret $Scope smtp-user     --string-value $SmtpUser --profile $Profile
databricks secrets put-secret $Scope smtp-password --string-value $SmtpPass --profile $Profile

# Redshift credentials (optional — leave blank if not using Redshift)
Write-Host ""
$RsUser = Read-Host "  redshift-user (leave blank to skip)"
if ($RsUser -ne "") {
    $RsPassSecure = Read-Host "  redshift-password" -AsSecureString
    $RsPass = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($RsPassSecure)
    )
    databricks secrets put-secret $Scope redshift-user     --string-value $RsUser --profile $Profile
    databricks secrets put-secret $Scope redshift-password --string-value $RsPass --profile $Profile
}

# Teams webhook (optional)
Write-Host ""
$Webhook = Read-Host "  Teams webhook URL (leave blank to skip)"
if ($Webhook -ne "") {
    databricks secrets put-secret $Scope webhook-url --string-value $Webhook --profile $Profile
}

Write-Host ""
Write-Host ("Done. Listing secrets in scope: " + $Scope) -ForegroundColor Green
databricks secrets list-secrets $Scope --profile $Profile
