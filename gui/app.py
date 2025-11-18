"""
Main file for the abetico application
"""
import sys
import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Adw
from .ui import MainWindow

class Abetico(Adw.Application):
    """Main application class"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.win = None
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        """Callback activated when the application starts"""
        self.win = MainWindow(application=app)
        self.win.present()


def main():
    """Main function to start the application"""
    app = Abetico(application_id="com.erick-nix.abetico")
    app.run(sys.argv)


if __name__ == "__main__":
    main()
