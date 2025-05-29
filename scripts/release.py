#!/usr/bin/env python3
"""
Release script for Proctor.

Usage:
  python scripts/release.py [major|minor|patch]

This script will:
1. Bump the version in proctor/__init__.py
2. Update the CHANGELOG.md file
3. Commit the changes
4. Create a new git tag
5. Push the changes and tag
"""
import re
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def get_current_version():
    """Get current version from __init__.py."""
    init_file = Path("proctor/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find version in __init__.py")
    return match.group(1)


def bump_version(current_version, bump_type):
    """Bump the version according to the specified type."""
    major, minor, patch = map(int, current_version.split("."))
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_in_file(new_version):
    """Update the version in __init__.py."""
    init_file = Path("proctor/__init__.py")
    content = init_file.read_text()
    updated_content = re.sub(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        f'__version__ = "{new_version}"',
        content
    )
    init_file.write_text(updated_content)
    print(f"Updated version in __init__.py to {new_version}")


def update_changelog(new_version):
    """Update the CHANGELOG.md file."""
    changelog_path = Path("CHANGELOG.md")
    
    # Create changelog if it doesn't exist
    if not changelog_path.exists():
        changelog_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        changelog_path.write_text(changelog_content)
    
    # Read existing content
    changelog_content = changelog_path.read_text()
    
    # Add new version section
    today = datetime.now().strftime("%Y-%m-%d")
    new_version_section = f"## [{new_version}] - {today}\n\n### Added\n\n- \n\n### Changed\n\n- \n\n### Fixed\n\n- \n\n"
    
    # Find the position to insert (after the header)
    lines = changelog_content.split("\n")
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            insert_pos = i
            break
        elif i > 10:  # Look only in the first few lines
            break
    
    if insert_pos == 0:
        # Insert after the header
        for i, line in enumerate(lines):
            if line.startswith("# Changelog"):
                insert_pos = i + 2  # Skip an empty line
                break
    
    # Insert the new version section
    lines.insert(insert_pos, new_version_section)
    updated_content = "\n".join(lines)
    
    # Write back to file
    changelog_path.write_text(updated_content)
    print(f"Updated CHANGELOG.md with new version {new_version}")
    print("Please edit CHANGELOG.md to add your changes!")


def run_command(command):
    """Run a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def git_commit_and_tag(version):
    """Commit the version changes and create a new tag."""
    # Check if there are changes to commit
    status = run_command("git status --porcelain")
    if not status:
        print("No changes to commit.")
        return
    
    # Commit the changes
    run_command(f'git commit -a -m "chore: bump version to {version}"')
    print(f"Committed changes for version {version}")
    
    # Create a new tag
    run_command(f'git tag -a v{version} -m "Version {version}"')
    print(f"Created tag v{version}")


def main():
    """Main function."""
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python scripts/release.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    
    # Confirm with the user
    confirmation = input("Do you want to continue? [y/N] ")
    if confirmation.lower() != "y":
        print("Aborted.")
        sys.exit(0)
    
    # Update version and changelog
    update_version_in_file(new_version)
    update_changelog(new_version)
    
    # Let the user edit the changelog
    input("Press Enter after you've updated the CHANGELOG.md file...")
    
    # Commit and tag
    git_commit_and_tag(new_version)
    
    # Push changes and tag
    push_confirmation = input("Do you want to push the changes and tag? [y/N] ")
    if push_confirmation.lower() == "y":
        run_command("git push")
        run_command("git push --tags")
        print("Pushed changes and tags to remote repository.")
        print(f"Release v{new_version} is now ready!")
    else:
        print("Changes and tags were not pushed. You can push them later with:")
        print("  git push && git push --tags")


if __name__ == "__main__":
    main()