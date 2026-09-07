@echo off
cd /d "E:\PycharmProjects\Neural Network\SGGT-Net"
"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "scripts\run_p1_mtp_go_training.ps1" -Stage All -Seed 123 -SmokeEpochs 6
