@ECHO OFF
powershell -sta "add-type -as System.Windows.Forms; [windows.forms.clipboard]::GetText()" > %1
