import unittest
from unittest.mock import patch
from app.env.screen import Screen

class TestScreen(unittest.TestCase):
    def setUp(self):
        self.screen = Screen()
        self.screen.initialize()

    @patch('app.env.screen.tk')
    def test_initialize(self, mock_tk):
        self.screen.initialize()
        mock_tk.Tk.assert_called_once()
        mock_tk.Tk.return_value.title.assert_called_with("Simulated Screen")
        mock_tk.Tk.return_value.geometry.assert_called_with("800x600")

    def test_display_ui(self):
        ui_elements = ["button", "text_input", "checkbox"]
        self.screen.display_ui(ui_elements)
        self.assertEqual(self.screen.ui_elements, ui_elements)

    @patch('app.env.screen.tk')
    def test_render(self, mock_tk):
        ui_elements = ["button", "text_input"]
        self.screen.display_ui(ui_elements)
        self.screen.render()
        self.assertEqual(mock_tk.Tk.return_value.winfo_children.call_count, 2)
        self.assertEqual(mock_tk.Button.call_count, 1)
        self.assertEqual(mock_tk.Entry.call_count, 1)
        self.assertEqual(mock_tk.Checkbutton.call_count, 0)

    def tearDown(self):
        self.screen.window.destroy()

if __name__ == '__main__':
    unittest.main()
