import re


class SubjectIdParser:
    """
    Extracts the first numeric sequence from a directory name and converts it to int.

    Examples:
    - sub-01 -> 1
    - subject_007 -> 7
    - participant12 -> 12
    - 0005 -> 5
    """

    _SUBJECT_ID_PATTERN = re.compile(r"(\d+)")

    @staticmethod
    def parse(self, directory_name: str) -> int | None:
        match = self._SUBJECT_ID_PATTERN.search(directory_name)
        if match is None:
            return None

        return int(match.group(1))