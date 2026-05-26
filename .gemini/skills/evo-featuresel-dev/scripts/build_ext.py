import os
import sys
import subprocess
import shutil

def build():
    print("🚀 Building C extensions (evo.core)...")
    
    # 1. Clear old build artifacts if any
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".pyd") or f.endswith(".so"):
                print(f"🧹 Removing old extension: {f}")
                os.remove(os.path.join(root, f))
    
    # 2. Execute setup.py build_ext --inplace
    try:
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
        print("✅ Build completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

    # 3. Verify Import
    try:
        from evo.core import pack_bits
        print("🔍 Import test: SUCCESS. pack_bits is available.")
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
