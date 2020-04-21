import wikipedia
k=wikipedia.summary("Computer vision")
from gtts import gTTS 
tts = gTTS(text=k,lang='en')
tts.save("wiki.mp3")
