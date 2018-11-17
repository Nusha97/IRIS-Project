import os
import datetime
now=datetime.datetime.now()
hour=now.hour
minute=now.minute
if(hour>12):
    ap="P.M"
    hour=hour%12
else:
    ap="A.M"
a = "The time is %d %d %s" %(hour,minute,ap)
a = "espeak -ven+m1 -s150 \"%s\"" %(a)
print a
os.system(a)
#from gtts import gTTS 
#tts = gTTS(text=a,lang='en')
#tts.save("time.mp3")
