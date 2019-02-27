@echo off
break on

setlocal enabledelayedexpansion

rem   scriptpath
set scriptpath=%~dpsnx0
rem removing short batch name path
set scriptpath=!scriptpath:\%~snx0=!
rem removing long batch name from path
set scriptpath=!scriptpath:\%~nx0=!

set plink=255
if exist "!scriptpath!\plink.exe" set plink="!scriptpath!\plink.exe"
if exist "c:\program files\putty\plink.exe" set plink="c:\program files\putty\plink.exe"
if exist "c:\program files (x86)\putty\plink.exe" set plink="c:\program files (x86)\putty\plink.exe"
if exist "c:\windows\plink.exe" set plink="c:\windows\plink.exe"
if "!plink!"=="255" (
	echo.
	echo   [error] : plink.exe not found under these path
	echo.
	echo     "!scriptpath!\plink.exe"
	echo     "C:\Program Files\Putty\plink.exe"
	echo     "C:\Program Files (X86)\Putty\plink.exe"
	echo     "C:\Windows\plink.exe"
	echo.
	exit /b 255
)

set id_rsa_path=255
if exist "!scriptpath!\id_rsa.ppl" set id_rsa_path="!scriptpath!\id_rsa.ppl"
if "!id_rsa_path!" == "255" (
    	echo.
   	echo   [error] : id_rsa not found under these path
	echo.
	echo     "!scriptpath!\id_rsa.ppl"
    	echo   You should ask your teacher for this file
	echo.
	exit /b 255
)

echo.
echo.
set /p remote_login=Login name ? [e.g. sm20_1]
set /p remote_host=Host name  ? [e.g. cam01]

if "!remote_login!"=="" (
	echo.
	echo   [error] : login name can't be empty
	exit /b 255
)
if "!remote_host!"=="" (
	echo.
	echo   [error] : host name can't be empty
	exit /b 255
)

echo.
echo.
echo   Trying to port forward ssh connection from host [!remote_host!] with login [!remote_login!] to localhost:10000
echo.
start "plink" /min cmd /c echo y^| !plink! -v -N -l !remote_login! -agent -i !id_rsa_path! -L 10000:!remote_host!:22 ghome.metz.supelec.fr
timeout /t 5 /nobreak >nul
echo.
echo   !errorlevel! = plink exit code
echo.
echo.
echo   Trying to port forward jupyter lab from host [!remote_host!] with login [!remote_login!] to localhost:8888
echo.
start "plink" /min cmd /c echo y^| !plink! -N -P 10000 -l !remote_login! -agent  -i !id_rsa_path! -L 8888:127.0.0.1:8888 localhost
echo.
echo   !errorlevel! = plink exit code
echo.
echo.
echo   Trying to port forward tensorboard from host [!remote_host!] with login [!remote_login!] to localhost:6006
echo.
start "plink" /min cmd /c echo y^| !plink! -N -P 10000 -l !remote_login! -agent  -i !id_rsa_path! -L 6006:127.0.0.1:6006 localhost
echo.
echo   !errorlevel! = plink exit code
timeout /t 10 > nul

