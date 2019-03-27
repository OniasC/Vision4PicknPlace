from __future__ import print_function
import numpy as np
from datetime import datetime

time_begin_filt = float(datetime.now().strftime("%H%M%S.%f"))
time = datetime.now()
print(time_begin_filt)
print (time)
time_now = time_begin_filt
while (time_now < time_begin_filt + 4):
    time_now = float(datetime.now().strftime("%H%M%S.%f")) 

print (time_now)
print (datetime.now())

