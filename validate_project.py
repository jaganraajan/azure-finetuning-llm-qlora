"""
Basic validation script for the healthcare QLoRA project structure.
"""

import os
import sys
from pathlib import Path

def validate_project_structure():
    """Validate that all required files and directories exist."""
    project_root = Path(__file__).parent
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "config/config.yaml",
        "config/conda_env.yaml",
        "src/azure_qlora_healthcare/__init__.py",
        "src/azure_qlora_healthcare/utils/config.py",
        "src/azure_qlora_healthcare/utils/logger.py",
        "src/azure_qlora_healthcare/utils/azure_ml.py",
        "src/azure_qlora_healthcare/data/processor.py",
        "src/azure_qlora_healthcare/training/qlora_trainer.py",
        "src/azure_qlora_healthcare/evaluation/metrics.py",
        "src/azure_qlora_healthcare/deployment/bot_service.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/setup_azure.py",
        "scripts/run_bot.py",
        "scripts/prepare_data.py",
        "tests/test_basic.py",
        "data/examples/sample_healthcare_qa.csv",
        "data/examples/README.md",
    ]
    
    required_dirs = [
        "src",
        "src/azure_qlora_healthcare",
        "src/azure_qlora_healthcare/utils",
        "src/azure_qlora_healthcare/data",
        "src/azure_qlora_healthcare/training",
        "src/azure_qlora_healthcare/evaluation",
        "src/azure_qlora_healthcare/deployment",
        "config",
        "scripts",
        "tests",
        "data",
        "data/examples",
        "notebooks",
        "docs",
        "models",
    ]
    
    print("🔍 Validating project structure...")
    
    # Check directories
    missing_dirs = []
    for directory in required_dirs:
        dir_path = project_root / directory
        if not dir_path.exists():
            missing_dirs.append(directory)
        else:
            print(f"✅ Directory: {directory}")
    
    # Check files
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ File: {file_path}")
    
    # Report results
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
    
    if not missing_dirs and not missing_files:
        print(f"\n🎉 All required files and directories are present!")
        return True
    else:
        print(f"\n⚠️  Project structure validation failed.")
        return False

def validate_file_contents():
    """Validate that key files have expected content."""
    project_root = Path(__file__).parent
    
    print("\n🔍 Validating file contents...")
    
    # Check README
    readme_path = project_root / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()
        if "QLoRA" in readme_content and "Healthcare" in readme_content and "Azure" in readme_content:
            print("✅ README.md has expected content")
        else:
            print("❌ README.md missing expected keywords")
    
    # Check requirements.txt
    req_path = project_root / "requirements.txt"
    if req_path.exists():
        req_content = req_path.read_text()
        expected_packages = ["torch", "transformers", "azure-ai-ml", "peft", "bitsandbytes"]
        missing_packages = [pkg for pkg in expected_packages if pkg not in req_content]
        if not missing_packages:
            print("✅ requirements.txt has expected packages")
        else:
            print(f"❌ requirements.txt missing packages: {missing_packages}")
    
    # Check config files
    config_path = project_root / "config/config.yaml"
    if config_path.exists():
        config_content = config_path.read_text()
        if "azure:" in config_content and "qlora:" in config_content:
            print("✅ config.yaml has expected structure")
        else:
            print("❌ config.yaml missing expected structure")
    
    return True

def validate_python_syntax():
    """Validate Python files for syntax errors."""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    scripts_dir = project_root / "scripts"
    
    print("\n🔍 Validating Python syntax...")
    
    python_files = []
    for directory in [src_dir, scripts_dir]:
        if directory.exists():
            python_files.extend(directory.rglob("*.py"))
    
    syntax_errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
            print(f"✅ Syntax OK: {py_file.relative_to(project_root)}")
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
            print(f"❌ Syntax Error: {py_file.relative_to(project_root)} - {e}")
    
    if not syntax_errors:
        print("🎉 All Python files have valid syntax!")
        return True
    else:
        print(f"⚠️  Found {len(syntax_errors)} syntax errors")
        return False

def main():
    """Run all validations."""
    print("=" * 60)
    print("Healthcare QLoRA Project Validation")
    print("=" * 60)
    
    structure_ok = validate_project_structure()
    content_ok = validate_file_contents()
    syntax_ok = validate_python_syntax()
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if structure_ok and content_ok and syntax_ok:
        print("🎉 All validations passed! Project is ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure Azure: cp .env.example .env && edit .env")
        print("3. Run training: python scripts/train.py")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())