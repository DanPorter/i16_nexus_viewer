"""
Define Instrument class
 An instrument is a generator of Scans and FolderMonitors (experiements) with specific default settings
"""

from . import file_loader, FolderMonitor


class Instrument:
    """
    Instrument class
     An instrument is a generator of Scans and FolderMonitors (experiements) with specific default settings

    beamline = Instrument('name', default_names, functions, filename_format)
    :param name: str : name of instrument
    :param default_names: dict : Scan objects created will
    :param formats: dict :
    :param options: dict :
    """
    def __init__(self, name, default_names=None, formats=None, default_values=None,
                 options=None, filename_format=None):
        self.name = name
        self._default_names = {} if default_names is None else default_names
        self._formats = {} if formats is None else formats
        self._default_values = {} if default_values is None else default_values
        self._options = {} if options is None else options

    def __repr__(self):
        return "Instrument(%s)" % self.name

    def __str__(self):
        return '%r' % self

    def set_format(self, filename_format='%06d.nxs'):
        """Set the file format to monitor, uses printf-style string format, e.g. '%5d.nxs'"""
        self._options['filename_format'] = filename_format

    def _add_items(self, scan):
        """Add Insturment defaults to Scan"""
        scan.options(**self._options)
        for name, alt_names in self._default_names.items():
            scan.add2namespace(name, other_names=alt_names)
        for name, value in self._default_values.items():
            scan.add2namespace(name, default_value=value)
        for name, operation in self._formats.items():
            string = scan.string_format(operation)
            scan.add2namespace(name, string)

    def _scan_loader(self, filename, **kwargs):
        """Loads a babelscan.Scan and adds """
        scan = file_loader(filename, **kwargs)
        for name, alt_names in self._default_names.items():
            scan.add2namespace(name, other_names=alt_names)
        for name, value in self._default_values.items():
            scan.add2namespace(name, default_value=value)
        for name, operation in self._formats.items():
            string = scan.string_format(operation)
            scan.add2namespace(name, string)
        return scan

    def experiment(self, directory, working_dir='.', **kwargs):
        """Create FolderMonitor"""
        options = self._options.copy()
        options.update(kwargs)
        return FolderMonitor(directory, working_dir, self._scan_loader, **options)

    def scan(self, filename, **kwargs):
        """return babelscan.Scan"""
        options = self._options.copy()
        options.update(kwargs)
        return self._scan_loader(filename, **options)
