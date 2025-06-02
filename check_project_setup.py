#!/usr/bin/env python3
"""
Author: @ChatGPT
Date: 2025-06-02
Project structure verification script.

"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Check if the project structure is correct and provide guidance."""
    
    print("🔍 Checking Project Structure...")
    print("=" * 60)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Check for key files that should be in project root
    required_files = [
        "utils.py",
        "collection.py", 
        "demo_preprocessing.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "src",
        "src/preprocessing",
        "src/classification"
    ]
    
    # Check files
    print("\n📄 Checking required files:")
    all_files_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_files_present = False
    
    # Check directories
    print("\n📁 Checking required directories:")
    all_dirs_present = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING!")
            all_dirs_present = False
    
    # Check optional data directories
    print("\n📊 Checking data directories:")
    data_dirs = ["MP_data", "processed_data"]
    for dir_path in data_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/ - Present")
        else:
            print(f"  ⚠️  {dir_path}/ - Not found (will be created when needed)")
    
    # Check specific preprocessing files
    print("\n🔧 Checking preprocessing module files:")
    preprocessing_files = [
        "src/preprocessing/__init__.py",
        "src/preprocessing/data_preprocessor.py",
        "src/preprocessing/data_augmentor.py", 
        "src/preprocessing/feature_engineer.py",
        "src/preprocessing/preprocessing_pipeline.py"
    ]
    
    all_preprocessing_present = True
    for file in preprocessing_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_preprocessing_present = False
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("📋 ASSESSMENT:")
    
    if all_files_present and all_dirs_present and all_preprocessing_present:
        print("✅ Project structure is CORRECT!")
        print("✅ You are in the right directory!")
        print("✅ All required files are present!")
        
        print("\n🚀 You can now run:")
        print("  python demo_preprocessing.py")
        print("  python collection.py")
        
    else:
        print("❌ Project structure has ISSUES!")
        
        if not all_files_present or not all_dirs_present:
            print("\n💡 SOLUTION:")
            print("  Make sure you are in the PROJECT ROOT directory!")
            print("  The project root should contain utils.py, collection.py, etc.")
            
            # Try to find project root
            current_path = Path.cwd()
            for parent in [current_path] + list(current_path.parents):
                if (parent / "utils.py").exists():
                    print(f"\n🎯 Found project root at: {parent}")
                    print(f"   Change directory with: cd {parent}")
                    break
        
        if not all_preprocessing_present:
            print("\n💡 PREPROCESSING MODULES:")
            print("  Some preprocessing files are missing.")
            print("  Make sure you have the complete src/preprocessing/ structure.")
    
    # Python path check
    print(f"\n🐍 Python executable: {sys.executable}")
    print(f"🗂️  Python path includes current dir: {'.' in sys.path or current_dir in sys.path}")
    
    # Import test
    print("\n🧪 Testing imports...")
    try:
        from src.preprocessing import PreprocessingPipeline
        print("  ✅ Successfully imported PreprocessingPipeline")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("     Make sure you're in the project root directory!")
    
    print("\n" + "=" * 60)
    print("💡 REMEMBER: Always run scripts from the project root directory!")
    print("💡 Project root is the directory containing utils.py")

def main():
    """Main function."""
    print("Hand Sign Language Recognition - Project Setup Checker")
    print("Author: @Chen YANG")
    print()
    
    check_project_structure()

if __name__ == "__main__":
    main() 