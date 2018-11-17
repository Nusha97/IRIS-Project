import os
os.system("cd ~/Desktop/IRIS")
os.system("omxplayer /home/pi/Desktop/IRIS/reply.mp3")
os.system("sh ~/Desktop/IRIS/capture.sh")
os.system("tesseract ocr.jpg ocr")
fo = open("ocr.txt","r")
t= fo.read(1)
fo.seek(0,0)
a = fo.read()
m= fo.tell()
print m
fo.close()
if t==" ":
    os.system("espeak -ven+m1 -s150 \"image appears to be blurred\"")
a = "espeak -ven+m1 -s150 \"%s\"" %(a)
print a
os.system(a)
#from gtts import gTTS 
#tts = gTTS(text=line,lang='en')
#tts.save("ocr.mp3")

