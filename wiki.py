import wikipedia
k=wikipedia.summary("How to die")
from gtts import gTTS 
tts = gTTS(text=k,lang='en')
tts.save("wiki.mp3")
