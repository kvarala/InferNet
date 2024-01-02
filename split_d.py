import pandas as pd
import numpy as np
import os

ins1=open("training_samples.txt")

lines=[]

for line in ins1:
        lines.append(line.strip().split("\t")[0])


df = pd.read_table('combined_AD_Log2files.txt')
counter=len(df.columns.values)
i=0
while(i<counter):
	print df.columns.values[i]
        if(df.columns.values[i] not in lines):
                df=df.drop(columns=df.columns.values[i])
                counter=counter-1
        elif(df.columns.values[i] in lines):
                i=i+1

df.to_csv("Training_data.txt", sep='\t')


