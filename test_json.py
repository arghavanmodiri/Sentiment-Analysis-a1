import json

data = json.load(open("preproc.json"))
i = 0
for line in data:
    i += 1
    print(i)
