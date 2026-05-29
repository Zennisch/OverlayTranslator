"""
Build script for OverlayTranslator
Builds CPU-only executable with PyTorch (~658MB distribution)
"""

import argparse
import os
import subprocess
import sys

import py7zr


def build(release=False):
    print("Building OverlayTranslator executable (CPU-only PyTorch)...\n")

    # Define paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    entry_point = os.path.join(script_dir, "app", "__main__.py")

    if not os.path.exists(entry_point):
        print(f"Error: Entry point not found at {entry_point}")
        sys.exit(1)

    # PyInstaller command
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir",
        "--name=OverlayTranslator",
        "--console",
        f"--paths={script_dir}",
        f"--paths={os.path.join(script_dir, 'app')}",
        # Exclude test files and unnecessary modules to reduce size
        "--exclude-module=pytest",
        "--exclude-module=test",
        "--exclude-module=tests",
        # Optimize bytecode
        "--optimize=2",
        # Core dependencies
        "--hidden-import=torch",
        "--hidden-import=einops",
        "--hidden-import=shapely",
        "--hidden-import=pyclipper",
        "--hidden-import=skimage",
        "--hidden-import=cv2",
        "--hidden-import=py3langid",
        "--hidden-import=networkx",
        "--hidden-import=tqdm",
        "--hidden-import=requests",
        # App modules
        "--hidden-import=app.bootstrap",
        "--hidden-import=app.config",
        "--hidden-import=app.service",
        "--hidden-import=app.core",
        "--hidden-import=app.core.exceptions",
        "--hidden-import=app.core.logger",
        "--hidden-import=app.manga_translator",
        "--hidden-import=app.manga_translator.detection",
        "--hidden-import=app.manga_translator.detection.common",
        "--hidden-import=app.manga_translator.detection.default",
        "--hidden-import=app.manga_translator.detection.default_utils",
        "--hidden-import=app.manga_translator.ocr",
        "--hidden-import=app.manga_translator.ocr.common",
        "--hidden-import=app.manga_translator.ocr.model_48px",
        "--hidden-import=app.manga_translator.ocr.xpos_relative_position",
        "--hidden-import=app.manga_translator.textline_merge",
        "--hidden-import=app.manga_translator.translators",
        "--hidden-import=app.manga_translator.translators.common",
        "--hidden-import=app.manga_translator.utils",
        "--hidden-import=app.manga_translator.utils.textblock",
        "--hidden-import=app.manga_translator.utils.sort",
        "--hidden-import=app.manga_translator.utils.generic",
        "--hidden-import=app.manga_translator.utils.inference",
        "--hidden-import=app.manga_translator.utils.log",
        "--clean",
        entry_point,
    ]

    try:
        subprocess.run(args, check=True)
        print("\n✅ Build completed successfully!")
        ext = ".exe" if sys.platform == "win32" else ""
        exe_path = os.path.join(script_dir, 'dist', f'OverlayTranslator{ext}')
        print(f"📦 Executable: {exe_path}")

        # Release mode: compress to 7z
        if release:
            print("\n🔄 Creating 7z archive for release...")
            dist_folder = os.path.join(script_dir, 'dist', 'OverlayTranslator')
            archive_path = os.path.join(script_dir, 'dist', 'OverlayTranslator.7z')

            if not os.path.exists(dist_folder):
                print(f"❌ Distribution folder not found: {dist_folder}")
                sys.exit(1)

            try:
                # Remove existing archive if it exists
                if os.path.exists(archive_path):
                    os.remove(archive_path)

                # Create 7z archive with maximum compression
                with py7zr.SevenZipFile(archive_path, 'w') as archive:
                    archive.writeall(dist_folder, arcname='OverlayTranslator')

                archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
                print(f"📦 Release archive: {archive_path}")
                print(f"📊 Archive size: {archive_size_mb:.2f} MB")
            except Exception as e:
                print(f"❌ Archive creation failed: {e}")
                sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build OverlayTranslator executable (CPU-only PyTorch)"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Create 7z archive for release after building",
    )
    args = parser.parse_args()

    build(release=args.release)
