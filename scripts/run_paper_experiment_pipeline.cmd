@echo off
cd /d "E:\PycharmProjects\Neural Network\SGGT-Net"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\PycharmProjects\Neural Network\SGGT-Net\scripts\run_paper_experiment_pipeline.ps1"
exit /b %ERRORLEVEL%
