from gtts import gTTS
import os
import datetime
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import threading
import io
def textToSpeechVietnamese(text, speed=1.0):
    def generateAndPlayAudio():
        tts = gTTS(text=text, lang='vi', slow=False)
        fp =io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio = AudioSegment.from_mp3(fp)

        if speed != 1.0:
            audio = audio.speedup(playback_speed=speed)

        play(audio)
        # now = datetime.datetime.now()
        # current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        # file_name = f"{current_time}.mp3"
        # tts.save(file_name)

        # sound = AudioSegment.from_mp3(file_name)
        
        # faster_sound = sound.speedup(playback_speed=1.2)
        
        # new_file_name = f"faster_{file_name}"
        # faster_sound.export(new_file_name, format="mp3")

        # os.system(f"mpg321 {new_file_name}")
        
        # os.remove(file_name)
        # os.remove(new_file_name)
    thread = threading.Thread(target=generateAndPlayAudio)
    thread.daemon=True
    thread.start()

# text = "Xin chào, tôi là trợ lý ảo!"
# textToSpeechVietnamese(text)
