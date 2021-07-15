@ECHO OFF
ECHO Beginning to get clipboard.
PAUSE
FOR /F "tokens=* USEBACKQ" %%F IN (
  `powershell -sta "add-type -as System.Windows.Forms; [windows.forms.clipboard]::GetText()"`
) DO (
   SET var=%%F
)

ECHO Clip text =
ECHO %var%

IF [%1]==[] GOTO nofile

ECHO Making file named %1
powershell -sta "add-type -as System.Windows.Forms; [windows.forms.clipboard]::GetText()" > %1
goto :eof

:nofile
ECHO skipping making a file.

exit /B 1
