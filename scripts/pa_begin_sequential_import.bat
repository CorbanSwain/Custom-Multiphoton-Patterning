<# : pa_begin_sequential_import.bat
:: Begins sequential import of masks into PrarieView

@echo off
setlocal EnableDelayedExpansion

set "cacheDir=TEMP"

echo Beginning script to initiate a sequential import.

set "browsePrompt="Please select the folder containing split PA files.""
set "folderBrowse="(new-object -COM 'Shell.Application').BrowseForFolder(0, '%browsePrompt%', 0x200, 0).self.path""
for /f "usebackq delims=" %%# in (`PowerShell %folderBrowse%`) do set "selected_dir=%%#"

echo The selected directory is: "%selected_dir%"

set "dirFilename=%cacheDir%\sequential_dir.txt"
echo %selected_dir%> %dirFilename%

set "counterFilename=%cacheDir%\sequential_counter.txt"
>%counterFilename% echo 1

echo Importing tht first file in this sequence.

set /p currentIdx=<%counterFilename%

echo Current index is %currentIdx%.

for /f %%n in ('powershell -NoLogo -NoProfile -Command "([double]%currentIdx%).ToString('000')"') do (set "idxStr=%%n")
set /p seqDir=<%dirFilename%
set "pamFilename=%seqDir%\%idxStr%.txt"

if exist "%pamFilename%" (
    echo PA command file found.
    echo Loading PA command file # %idxStr% from: "%pamFilename%"
    echo Saving to the following temp file: "%1"
    copy "%pamFilename%" "%1" /Y
    set /A nextIdx=%currentIdx%+1
    echo Next index is %nextIdx%.
    >%counterFilename% echo %nextIdx%
) else (
    echo PA command file "%pamFilename%" not found; end of sequence potentially reached.
)

echo Exiting action.
Exit /B
