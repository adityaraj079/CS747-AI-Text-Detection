from streamlit.testing.v1 import AppTest


def test_runtime_note_remains_visible_after_rerun():
    app = AppTest.from_string('''
from app_components.inference import show_live_predictions
class Adapter:
    runtime_note = "Using six ensemble members."
    def predict_proba(self, texts):
        return [0.7] * len(texts)
show_live_predictions(("yash",), {"yash": {"short_name": "Yash"}}, lambda _: Adapter())
''').run()
    app.text_area[0].set_value("A sample essay.")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.table[0].value) == 1
    assert any("Using six ensemble members" in c.value for c in app.caption)
    app.run()
    assert not app.exception
    assert any("Saved benchmark scores describe the original ensemble" in c.value for c in app.caption)
