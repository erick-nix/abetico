"""
User interface module for the Abético application
"""
import os
import gi
from gi.repository import Gtk, Adw, Gio, Gdk
from .events import EventHandlers

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

@Gtk.Template(filename=os.path.join(os.path.dirname(__file__), 'ui/main.ui'))
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    # Template children - correspond to the IDs in the Blueprint file
    main_stack: Gtk.Stack = Gtk.Template.Child()
    convert_button: Gtk.Button = Gtk.Template.Child()
    back_header_button: Gtk.Button = Gtk.Template.Child()
    submit_button: Gtk.Button = Gtk.Template.Child()
    progress_bar: Gtk.ProgressBar = Gtk.Template.Child()
    toast_overlay: Adw.ToastOverlay = Gtk.Template.Child()

    # Additional template children for question inputs and error display
    error_label: Gtk.Label = Gtk.Template.Child()
    age_entry: Gtk.Entry = Gtk.Template.Child()
    bmi_entry: Gtk.Entry = Gtk.Template.Child()
    glucose_entry: Gtk.Entry = Gtk.Template.Child()
    family_history_entry: Gtk.Entry = Gtk.Template.Child()
    result_percentage_label: Gtk.Label = Gtk.Template.Child()
    result_description_label: Gtk.Label = Gtk.Template.Child()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize event handlers
        self.event_handlers = EventHandlers(self)

        # Setup about dialog action
        self.setup_actions()

        # Load CSS
        self.load_css()

    def setup_actions(self):
        """Setup menu actions"""
        # Create action for "About"
        about_action = Gio.SimpleAction.new("show_about", None)
        about_action.connect("activate", self.on_show_about)
        self.get_application().add_action(about_action)

    def on_show_about(self, _button, _param):
        """Show About window"""
        about = Adw.AboutDialog.new()

        about.set_application_name("Abético")
        about.set_version("1.0.0")
        about.set_developer_name("Erick Henrique")
        about.set_issue_url("https://github.com/erick-nix/abetico")

        #about.set_comments("Application for diabetes prediction using Machine Learning")
        #about.set_copyright("© 2025 Erick Henrique")
        #about.set_license_type(Gtk.License.MIT_X11)

        # Developers
        about.set_developers([
            "Erick Henrique Souza de Paula",
            "Gabriel Oliveira Delorenzo"
        ])

        # Additional information
        about.add_acknowledgement_section(
            "Pessoas envolvidas no projeto",
            [
                "Emilli Giuliane Pereira Lima",
                "Erick Henrique Souza de Paula",
                "Gabriel Oliveira Delorenzo",
                "Ketlhen Nunes de Carvalho",
                "Letícia Ferreira Pinto"
            ]
        )

        about.present(self)

    def load_css(self):
        """Loads the custom CSS file"""
        css_provider = Gtk.CssProvider()
        css_file = os.path.join(os.path.dirname(__file__), 'ui/style.css')

        try:
            css_provider.load_from_file(Gio.File.new_for_path(css_file))
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            print(f"CSS loaded: {css_file}")
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading CSS: {e}")

    @Gtk.Template.Callback()
    def on_go_questions(self, button):
        """Callback to go to questions page"""
        self.event_handlers.on_go_questions(button)

    @Gtk.Template.Callback()
    def on_back_to_home(self, button):
        """Callback to go back to home"""
        self.event_handlers.on_back_to_home(button)

    @Gtk.Template.Callback()
    def on_submit_questions(self, button):
        """Callback to submit answers"""
        self.event_handlers.on_submit_questions(button)
