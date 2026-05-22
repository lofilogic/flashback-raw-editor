"""
processor_v2 editor — subclasses FlashbackEditor and adds the "Push / Pull"
slider beneath Exposure. Nothing in the v1 editor is touched; main.py picks
this class when FB_PROCESSOR=v2.

Push / Pull: pulling left scales the pre-LUT exposure down by 2^pp and
counteracts it post-LUT (brightness ~unchanged, the film curve's toe gets
more pronounced) and shifts grain toward highlights; pushing right does the
opposite. Stored in processor.user_settings['push_pull_ev'].
"""
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QEvent

from ui.editor import FlashbackEditor
from ui.scrub_slider import ScrubSlider
from core.config import PUSH_PULL_RANGE_EV

# Slider works in 0.1-EV steps.
_PP_STEPS = int(round(PUSH_PULL_RANGE_EV * 10))


def _fmt_pp(pp: float) -> str:
    return "0.0" if abs(pp) < 1e-9 else f"{pp:+.1f}"


class FlashbackEditorV2(FlashbackEditor):
    _DEFAULT_USER_SETTINGS = {'exposure_ev': 0.0, 'wb_temp': 0, 'tint': 0.0,
                              'push_pull_ev': 0.0}

    # ---- UI ----------------------------------------------------------------

    def _build_tone_section(self):
        sec = super()._build_tone_section()
        layout = sec.layout()

        self.label_pushpull = QLabel(_fmt_pp(0.0))
        self.label_pushpull.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider_pushpull = ScrubSlider(dual=True)
        self.slider_pushpull.setMinimum(-_PP_STEPS)
        self.slider_pushpull.setMaximum(_PP_STEPS)
        self.slider_pushpull.setValue(0)
        self.slider_pushpull.valueChanged.connect(self.on_pushpull_slider_moved)
        self.slider_pushpull.sliderReleased.connect(self.on_pushpull_released)
        self.slider_pushpull.installEventFilter(self)
        row = self._slider_row("PUSH / PULL", self.label_pushpull,
                               self.slider_pushpull)
        row.setVisible(False)
        layout.addWidget(row)
        return sec

    # ---- handlers ----------------------------------------------------------

    def on_pushpull_slider_moved(self, value):
        pp = value / 10.0
        self.label_pushpull.setText(_fmt_pp(pp))
        self.processor.user_settings['push_pull_ev'] = pp
        self._render_worker.request(downscale=True)

    def on_pushpull_released(self):
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    def reset_pushpull_slider(self):
        self.slider_pushpull.blockSignals(True)
        self.slider_pushpull.setValue(0)
        self.slider_pushpull.blockSignals(False)
        self.label_pushpull.setText(_fmt_pp(0.0))
        self.processor.user_settings['push_pull_ev'] = 0.0
        self.save_current_settings()
        self._render_needs_commit = True
        self._render_worker.request(downscale=False)

    # ---- overrides to keep the new slider in sync --------------------------

    def eventFilter(self, source, event):
        if (event.type() == QEvent.Type.MouseButtonDblClick
                and source is getattr(self, 'slider_pushpull', None)):
            self.reset_pushpull_slider()
            return True
        return super().eventFilter(source, event)

    def update_sliders_from_processor(self):
        super().update_sliders_from_processor()
        pp = float(self.processor.user_settings.get('push_pull_ev', 0.0))
        self.slider_pushpull.blockSignals(True)
        self.slider_pushpull.setValue(int(round(pp * 10)))
        self.slider_pushpull.blockSignals(False)
        self.label_pushpull.setText(_fmt_pp(pp))

    def reset_all_sliders(self):
        super().reset_all_sliders()
        # super() rebuilt user_settings without push_pull_ev; restore it and
        # zero the slider.
        self.processor.user_settings['push_pull_ev'] = 0.0
        self.slider_pushpull.blockSignals(True)
        self.slider_pushpull.setValue(0)
        self.slider_pushpull.blockSignals(False)
        self.label_pushpull.setText(_fmt_pp(0.0))
        img_array = self.processor.render_preview()
        self.display_image(img_array)
        self.update_current_thumbnail(img_array)
        self.save_current_settings()
