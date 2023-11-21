# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['monitor_gui.py'],
             pathex=['C:\\Users\\Mirtec\\PycharmProjects\\verify_exe'],
             binaries=[],
             datas=[('monitor_gui.ui', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='AI_Verify',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , version='version.txt' )
