@echo off
set DATABRICKS=C:\Users\velmu\AppData\Local\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe\databricks.exe
set REPO=c:\Users\velmu\OneDrive\Desktop\scf-cohort-dab

echo === Step 1: Deploy bundle to dev ===
cd /d "%REPO%"
"%DATABRICKS%" bundle deploy -t dev
if %ERRORLEVEL% NEQ 0 (
    echo DEPLOY FAILED - check errors above
    pause
    exit /b 1
)
echo Deploy succeeded.

echo.
echo === Step 2: Submit create_all_tables notebook ===
"%DATABRICKS%" jobs submit --json "@create_all_tables_run.json" --no-wait
if %ERRORLEVEL% NEQ 0 (
    echo Submit failed
    pause
    exit /b 1
)

echo.
echo === Done! Check Databricks UI for table creation progress ===
echo Catalog: pd_dtl_ds_dev ^> savings_cashflow ^> Tables
pause
