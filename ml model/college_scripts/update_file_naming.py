"""
Update File Naming Convention
==============================

Updates file names to include both years (e.g., 2012-2013, 2015-2016, 2017-2018).

This script:
1. Renames manual_scrapes folders (2012 -> 2012-2013)
2. Updates references in scripts
3. Updates output file names

Note: NCAA seasons span two calendar years (e.g., 2012-2013 season).
"""

import os
import shutil
from pathlib import Path
import re

# Mapping: season start year -> season label
SEASON_MAPPING = {
    2012: "2012-2013",
    2015: "2015-2016",
    2017: "2017-2018",
}

def rename_folders():
    """Rename manual_scrapes folders to include both years."""
    base_dir = Path("data/manual_scrapes")
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return False
    
    renamed = []
    for old_name, new_name in SEASON_MAPPING.items():
        old_path = base_dir / str(old_name)
        new_path = base_dir / new_name
        
        if old_path.exists() and not new_path.exists():
            print(f"📁 Renaming: {old_path} -> {new_path}")
            old_path.rename(new_path)
            renamed.append((old_name, new_name))
        elif old_path.exists() and new_path.exists():
            print(f"⚠️  Both exist: {old_path} and {new_path}. Skipping.")
        elif not old_path.exists():
            print(f"⚠️  Not found: {old_path}. Skipping.")
    
    if renamed:
        print(f"\n✅ Renamed {len(renamed)} folders")
        return True
    else:
        print("\n⚠️  No folders renamed")
        return False

def update_clean_script():
    """Update clean_historical_pbp_v2.py to handle new folder names."""
    script_path = Path("college_scripts/utils/clean_historical_pbp_v2.py")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update folder detection to handle both formats (YYYY and YYYY-YYYY)
    # Current: checks for 4-digit year folders
    # New: checks for 4-digit year OR YYYY-YYYY format
    
    old_pattern = r'if os\.path\.isdir\(full_path\) and item\.isdigit\(\) and len\(item\) == 4:'
    new_pattern = r'''if os.path.isdir(full_path) and (
            (item.isdigit() and len(item) == 4) or  # Old format: 2012
            (re.match(r'^\d{4}-\d{4}$', item))  # New format: 2012-2013
        ):'''
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        # Also need to extract season from folder name
        # Current: season = int(os.path.basename(folder))
        # New: Extract first year from folder name
        
        old_season_extract = r"season = int\(os\.path\.basename\(folder\)\)"
        new_season_extract = """# Extract season from folder name (handle both YYYY and YYYY-YYYY)
                    folder_name = os.path.basename(folder)
                    if '-' in folder_name:
                        season = int(folder_name.split('-')[0])  # 2012-2013 -> 2012
                    else:
                        season = int(folder_name)"""
        
        if old_season_extract in content:
            content = content.replace(old_season_extract, new_season_extract)
        
        # Add re import if not present
        if 'import re' not in content:
            # Find the import section
            import_section = content.split('\n')[:10]
            if 'import re' not in '\n'.join(import_section):
                # Add after other imports
                content = content.replace('import json', 'import json\nimport re', 1)
        
        with open(script_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated: {script_path}")
        return True
    else:
        print(f"⚠️  Pattern not found in {script_path}. May already be updated.")
        return False

def update_documentation():
    """Update PROJECT_MAP.md with new naming convention."""
    doc_path = Path("PROJECT_MAP.md")
    
    if not doc_path.exists():
        print(f"⚠️  Documentation not found: {doc_path}")
        return False
    
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Update references to manual_scrapes folders
    old_ref = r'`data/manual_scrapes/\{YEAR\}/`: Landing zone for raw historical CSVs \(2012, 2015, 2017 active\)\.'
    new_ref = r'`data/manual_scrapes/\{YEAR-YEAR\}/`: Landing zone for raw historical CSVs (2012-2013, 2015-2016, 2017-2018 active).'
    
    if old_ref in content:
        content = content.replace(old_ref, new_ref)
    
    # Update any other references
    content = content.replace('(2012, 2015, 2017 active)', '(2012-2013, 2015-2016, 2017-2018 active)')
    
    with open(doc_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated: {doc_path}")
    return True

def main():
    print("🔄 Updating File Naming Convention")
    print("="*60)
    
    # Step 1: Rename folders
    print("\n📁 Step 1: Renaming folders...")
    rename_folders()
    
    # Step 2: Update scripts
    print("\n📝 Step 2: Updating scripts...")
    update_clean_script()
    
    # Step 3: Update documentation
    print("\n📚 Step 3: Updating documentation...")
    update_documentation()
    
    print("\n" + "="*60)
    print("✅ File Naming Update Complete!")
    print("="*60)
    print("\nNew folder structure:")
    for old, new in SEASON_MAPPING.items():
        print(f"  data/manual_scrapes/{old}/ -> data/manual_scrapes/{new}/")
    print("\nNote: The clean script now handles both formats (YYYY and YYYY-YYYY)")

if __name__ == "__main__":
    main()
