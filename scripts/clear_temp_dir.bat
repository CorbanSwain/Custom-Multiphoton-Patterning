@ECHO OFF

:: get the current directory
SET _startdir="%CD%"

:: move to the users TEMP directory (agnostic to the starting drive)
CD /D %TEMP%

:: setup a counter to keep track of the number of deleted files
SET "_count=0"

:: perform a loop over filenames with the dir command (1) giving a
:: bare display of filenames [\b], (2) not including directories
:: [/a-d], and (3) only matching filenames with the pattern `tmp*.tmp`
:: "delims=" ensures that filenames with spaces are not broken
:: up into seperate tokens
FOR /f "delims=" %%G IN ('DIR /b /a-d "tmp*.tmp"') DO (
  :: increment the counter
  SET /A "_count+=1"

  :: delete the matched file
  DEL "%%G"
)

:: log the summary of the operation
ECHO Deleted %_count% temp file(s).

:: return back to the starting directory
CD /D %_startdir%
