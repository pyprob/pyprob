version_file = open(os.path.join('.', 'VERSION'))
__version__ = version_file.read().strip()
