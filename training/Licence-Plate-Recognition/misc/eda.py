import os
import pandas as pd
with open('st.txt', 'r') as f:
    lines = f.readlines()
f.close()
state_dict = dict()
for line in lines:
    code = line[-3:].strip()
    state_dict[f'{code}']=0
num_dict = {i:0 for i in range(25)}

path = "./images_1/"
for dirs in os.listdir(path):
        for img in os.listdir(os.path.join(path+dirs)):
            try:
                state_dict[img[:2].upper()]+=1
                num_dict[len(img.split('.')[0])]+=1
            except:
                pass
pd.DataFrame(state_dict,index=[0]).transpose().to_csv("eda.csv")
pd.DataFrame(num_dict,index=[0]).transpose().to_csv("eda2.csv")

