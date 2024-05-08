import os
import json


def read_json(filename):
    j = open(filename, 'r')
    json_object =  json.loads(j.read())
    j.close()
    return json_object


def save_json(request_body, filename):
    json_object = json.dumps(request_body, indent=4)    
    with open(filename, "w") as outfile:
        outfile.write(json_object)



key = "But here's where it gets even crazier: he bumps into another patient who's having the exact same nightmares. "
dir = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/clip_data'


arr = []
for file in os.listdir(dir):
    path = os.path.join(dir,file)
    data = read_json(path)[key]
    arr.extend(data)

arr.sort()

arr2 = []

for i in range(0,len(arr),5):

    mean_value = 0
    for j in range(i,min(len(arr),i+5+1)):
        mean_value += arr[i][1]
    
    st = i
    en = min(len(arr)-1,i+5)
    mean_value = float(mean_value)
    mean_value /= (en-st+1)

    median_value = (st + en)/2

    arr2.append((median_value, mean_value))


arr = arr2
print(arr)
import pdb; pdb.set_trace()
import matplotlib.pyplot as plt

import numpy as np
sorted_y = sorted(arr, key=lambda pair: pair[1], reverse=True)[:7]
x_sorted, y_sorted = zip(*sorted(arr))

# Creating the continuous plot again with correct data
plt.figure(figsize=(10, 6))
plt.plot(x_sorted, y_sorted, marker='o', linestyle='-', color='blue')

# Highlighting and labeling the top 5 peak y values correctly
for x_val, y_val in sorted_y:
    plt.text(x_val, y_val + 0.5, f'({x_val}, {y_val})', ha='center', va='bottom', color='red')

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Continuous Plot of (x, y) Tuples with Correctly Highlighted Peaks')
plt.grid(True)
plt.show()
# Sorting data by x values for a continuous plot

