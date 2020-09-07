import time

start = time.perf_counter()
for i in range(1,10) :
    continue
a = start*100
b = time.perf_counter()*1000
print(a,b,b-a)