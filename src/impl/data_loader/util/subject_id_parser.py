import re


class SubjectIdParser:
    """
    Utility class for extracting subject ID from directory name.
    """

    _SUBJECT_ID_PATTERN = re.compile(r"(\d+)")

    @staticmethod
    def parse(directory_name: str) -> int | None:
        match = SubjectIdParser._SUBJECT_ID_PATTERN.search(directory_name)
        if match is None:
            return None

        return int(match.group(1))