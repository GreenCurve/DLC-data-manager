"""Native Qt file/folder dialogs for use in Jupyter notebooks."""
from qtpy.QtWidgets import QApplication, QFileDialog


def _setup_qt():
    """Integrate Qt with the Jupyter kernel (if any) and return the QApplication."""
    try:
        ip = get_ipython()  # defined only inside IPython/Jupyter
    except NameError:
        ip = None

    if ip is not None:
        ip.run_line_magic("gui", "qt")  # same as typing %gui qt in a cell

    # Reuse an existing app (e.g. napari's) or create one
    return QApplication.instance() or QApplication([])


_app = _setup_qt()  # runs once, when the module is first imported


def _flush():
    _app.processEvents()


def select_folder(title="Select a folder", start_dir=""):
    path = QFileDialog.getExistingDirectory(None, title, start_dir)
    _flush()
    return path


def select_file(title="Select a file", start_dir="", filter="All files (*)"):
    path, _ = QFileDialog.getOpenFileName(None, title, start_dir, filter)
    _flush()
    return path


def select_files(title="Select files", start_dir="", filter="All files (*)"):
    paths, _ = QFileDialog.getOpenFileNames(None, title, start_dir, filter)
    _flush()
    return paths


def select_save_path(title="Save as", start_dir="", filter="All files (*)"):
    path, _ = QFileDialog.getSaveFileName(None, title, start_dir, filter)
    _flush()
    return path