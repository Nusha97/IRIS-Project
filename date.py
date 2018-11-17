import datetime
import os
now=datetime.datetime.now()
year=now.year
mon=now.month
day=now.day
if (mon==1):
    month="january"
elif (mon==2):
    month="february"
elif (mon==3):
    month="march"
elif (mon==4):
    month="april"
elif (mon==5):
    month="may"
elif (mon==6):
    month="june"
elif (mon==7):
    month="july"
elif (mon==8):
    month="august"
elif (mon==9):
    month="september"
elif (mon==10):
    month="october"
elif (mon==11):
    month="november"
elif (mon==12):
    month="december"
a = "today is %s %d %d" %(month,day,year)
a = "espeak -ven+m1 -s150 \"%s\"" %(a)
print a
os.system(a)

#from gtts import gTTS 
#tts = gTTS(text=a,lang='en')
#tts.save("date.mp3")
