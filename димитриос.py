#!/usr/bin/env python3

import numpy as np
import sys
import struct
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

f = open("agri_dataset_2020.txt", "r")
lines = 0

while True:
	line = f.readline()
	if not line:
		break
	lines += 1

Stemp0Array = np.zeros(lines, dtype=np.float32)
Stemp1Array = np.zeros(lines, dtype=np.float32)
Stemp2Array = np.zeros(lines, dtype=np.float32)
RtempArray = np.zeros(lines, dtype=np.float32)
RhumArray = np.zeros(lines, dtype=np.float32)
RainArray = np.zeros(lines, dtype=np.int16)
LumArray = np.zeros(lines, dtype=np.int16)
SmoistArray = np.zeros(lines, dtype=np.int16)
timestamps = []

f = open("agri_dataset_2020.txt", "r")
count = 0

missing_val = []

while True:
	#count += 1
	line = f.readline()
	if not line:
		break
	#print(line)
	fields = [n for n in line.split('""')]
	payload = ""
	timestamp = ""
	if (len(fields) < 20):
		missing_val.append(count)
		Stemp0Array[count] = -1
		Stemp1Array[count] = -1
		Stemp2Array[count] = -1
		RtempArray[count] = -1
		RhumArray[count] = -1
		RainArray[count] = -1
		LumArray[count] = -1
		SmoistArray[count] = -1
		timestamps.append("-1")
		count += 1
		continue
	#print(fields)
	count += 1
	for i in range(0, len(fields)):
		if (fields[i] == "dataFrame"):
			payload = fields[i+2]
			timestamp = fields[len(fields)-2]
			break
	payload = payload[2:]
	#print (count, payload, timestamp)
	soil_temp_0 = payload[0:8]
	soil_temp_1 = payload[8:16]
	soil_temp_2 = payload[16:24]
	room_temp = payload[24:32]
	room_hum = payload[32:40]
	rain_lvl = payload[40:44]
	lumin = payload[44:48]
	moisture = payload[48:52]
	
	soil_temp_0 = struct.unpack('!f', bytearray.fromhex(soil_temp_0))[0]
	soil_temp_1 = struct.unpack('!f', bytearray.fromhex(soil_temp_1))[0]
	soil_temp_2 = struct.unpack('!f', bytearray.fromhex(soil_temp_2))[0]
	room_temp = struct.unpack('!f', bytearray.fromhex(room_temp))[0]
	room_hum = struct.unpack('!f', bytearray.fromhex(room_hum))[0]
	rain_lvl = int(rain_lvl, 16)
	lumin = int(lumin, 16)
	moisture = int(moisture, 16)
	
	Stemp0Array[count-1] = soil_temp_0
	Stemp1Array[count-1] = soil_temp_1
	Stemp2Array[count-1] = soil_temp_2
	RtempArray[count-1] = room_temp
	RhumArray[count-1] = room_hum
	RainArray[count-1] = rain_lvl
	LumArray[count-1] = lumin
	SmoistArray[count-1] = moisture
	timestamps.append(timestamp)
	
	#print ("---------"+str(count)+"---------")
	#print ("At " + timestamp + " we received the following readings:")
	#print ("Soil temperature 1 = " + str(soil_temp_0) + " C")
	#print ("Soil temperature 2 = " + str(soil_temp_1) + " C")
	#print ("Soil temperature 3 = " + str(soil_temp_2) + " C")
	#print ("Room temperature = " + str(room_temp) + " C")
	#print ("Room humidity = " + str(room_hum) + " %")
	#print ("Rain level = " + str(rain_lvl) + " %")
	#print ("Luminosity = " + str(lumin) + " lux")
	#print ("Soil moisture = " + str(moisture) + " %")

f.close()


### find the max air temperature value
#x = np.amin(RtempArray[RtempArray >= 0])
#print(x)

### find when luminosity exceeded 80 lux
#x = np.where(LumArray > 50)
#print(len(x[0])) 
#for i in x[0]:
	#print(timestamps[i])

### find when was the last time the plant was watered and the moisture value
#x = np.where(SmoistArray > 20)
#print(timestamps[x[0][-1]], SmoistArray[x[0][-1]])


### PLOTS ###

# temp 0

#timestamps = np.array(timestamps)
#timestamps = [i for i in timestamps if i != "-1"]
#RtempArray = RtempArray[RtempArray >= 0]
#plt.plot(timestamps, RtempArray)
#plt.xticks(np.arange(1, lines, 10000.0), rotation=10)
#plt.ylabel('C')
#plt.show()

# Luminosity

#timestamps = [i for i in timestamps if i != "-1"]
#LumArray = LumArray[LumArray >= 0]
#plt.plot(timestamps, LumArray)
#plt.xticks(np.arange(1, lines, 10000.0), rotation=10)
#plt.ylabel('lux')
#plt.show()


### IMPUTATION ###

print(Stemp0Array)
print(Stemp0Array.reshape(-1, 1))
imp = SimpleImputer(missing_values=-1, strategy='median')
a = imp.fit_transform(Stemp0Array.reshape(-1, 1))
print(Stemp0Array[4222], a[4222][0], Stemp0Array[4221], Stemp0Array[4223])

#x = np.average(Stemp0Array[Stemp0Array >= 0])
#print(x)

#imp = KNNImputer(missing_values=-1, n_neighbors=2, weights='uniform')
#a = imp.fit_transform(Stemp0Array.reshape(-1, 1))
#print(Stemp0Array[4222], a[4222][0], Stemp0Array[4220], Stemp0Array[4221], Stemp0Array[4223], Stemp0Array[4224])

#for i in missing_val:
	#neighbors = 0
	#for k in range(1, 3):
		#neighbors += Stemp0Array[i-k]
		#neighbors += Stemp0Array[i+k]
	#neighbors /= 4
	#print(neighbors)
	
	#Stemp0Array[i] = neighbors


