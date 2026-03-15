from pathlib import Path


def list_files(directory=".", pattern="*", recursive=False, files_only=False, dirs_only=False):
    """
    List files in a directory using pathlib.

    Args:
        directory   (str): Path to the directory. Defaults to current directory.
        pattern     (str): Glob pattern to filter results e.g. "*.txt". Defaults to "*" (all).
        recursive  (bool): If True, searches subdirectories recursively. Defaults to False.
        files_only (bool): If True, returns only files. Defaults to False.
        dirs_only  (bool): If True, returns only directories. Defaults to False.

    Returns:
        list[Path]: A sorted list of matching Path objects.

    Examples:
        list_files()                              # all items in current directory
        list_files("/path/to/dir")                # all items in a specific directory
        list_files(".", "*.py")                   # only .py files
        list_files(".", "*.txt", recursive=True)  # all .txt files, including subdirectories
        list_files(".", files_only=True)          # only files, no folders
        list_files(".", dirs_only=True)           # only folders
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path.resolve()}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path.resolve()}")

    entries = path.rglob(pattern) if recursive else path.glob(pattern)

    if files_only:
        entries = (e for e in entries if e.is_file())
    elif dirs_only:
        entries = (e for e in entries if e.is_dir())

    return sorted(entries)
