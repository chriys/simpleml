from pathlib import Path


def check_folder_exists(lookup) -> bool:
    base_path = get_project_fullpath()
    filepath = Path(lookup)

    if filepath.is_absolute():
        return filepath.exists()
    else:
        new_path = base_path.joinpath(filepath)
        return new_path.exists()


def get_fullpath(given_path: str):
    given_path = Path(given_path)

    if given_path.is_absolute():
        return given_path

    base_path = get_project_fullpath()
    return base_path.joinpath(given_path)


def get_project_fullpath():
    return Path(__file__).parent.parent
