@echo off
cd /d "%~dp0"
py -3.11 ankles.py develop %*
pause

