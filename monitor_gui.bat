ECHO OFF

cd /d %~dp0
C:\Python\Python36_JDSong\Scripts\pyinstaller --onefile monitor_gui.spec

cmd /k