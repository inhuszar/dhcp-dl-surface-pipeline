import os


def resource(relpath: str):
    return os.path.join(os.path.dirname(__file__), relpath)
