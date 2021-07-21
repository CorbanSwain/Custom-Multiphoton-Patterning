<# : pa_begin_sequential_import.bat
:: Begins sequential import of masks into PrarieView

@echo off
setlocal EnableDelayedExpansion

set "cacheDir=TEMP"

echo Beginning script to continue a sequential import.

set "dirFilename=%cacheDir%\sequential_dir.txt"
set "counterFilename=%cacheDir%\sequential_counter.txt"

set /p currentIdx=<%counterFilename%

echo Current index is %currentIdx%.
set /A nextIdx=1 + %currentIdx%

for /f %%n in ('powershell -NoLogo -NoProfile -Command "([double]!currentIdx!).ToString('000')"') do (set "idxStr=%%n")
set /p seqDir=<%dirFilename%
set "pamFilename=%seqDir%\!idxStr!.txt"

if exist "%pamFilename%" (
    echo PA command file found.
    echo Loading PA command file # %idxStr% from: "%pamFilename%"
    echo Saving to the following temp file: "%1"
    copy "%pamFilename%" "%1" /Y
    echo Next index is %nextIdx%.
    >%counterFilename% echo %nextIdx%
) else (
    echo *** PA command file NOT found; end of sequence potentially reached. ***
)

echo Exiting action.
Exit /B
