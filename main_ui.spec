# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main_ui.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[ 'chumpy', 'vtkmodules',
                            'vtkmodules.all',
                            'vtkmodules.qt.QVTKRenderWindowInteractor',
                            'vtkmodules.util',
                            'vtkmodules.util.numpy_support',
                            'vtkmodules.numpy_interface.dataset_adapter',
                            'PyQt5.sip',
                            'PyQt5.QtCore',
                           ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_ui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False, #True
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False, #True
    upx_exclude=[],
    name='main_ui',
)
