from lib_speak import Speaker
from lib_sl_text import SeleroText

text = SeleroText("Пример текста для синтеза речи")
speaker = Speaker(model_id="ru_v3", language="ru", speaker="aidar", device="cpu")
speaker.speak(text=text, sample_rate=48000, speed=1.0)