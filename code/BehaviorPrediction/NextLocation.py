import pandas as pd
import numpy as np
from hmmlearn.hmm import MultinomialHMM , GaussianHMM , GMMHMM
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv", 
                  sep=',', 
                  names=["ID", "Day", "Time(Hours)", "Time(Minutes)","Time(Seconds)","Location", "Next Destination", "Action","Bathroom Light","Kitchen Light","TV"])
valid = pd.read_csv("test.csv", 
                  sep=',', 
                  names=["ID", "Day", "Time(Hours)", "Time(Minutes)","Time(Seconds)","Location", "Next Destination" , "Action","Bathroom Light","Kitchen Light","TV"])

 
input_sequences = []
# for idx in train.index:
count =0
idx = 0  
flag1 = True
flag2 = True   

while flag1:
    sub_list = []
    temp_list = []
    flag2 = True
    
    if len(train) > idx :
        sub_list.append(train["Location"][idx])
        sub_list.append(train["Day"][idx])
        sub_list.append(train["Time(Hours)"][idx])
        sub_list.append(train["Time(Minutes)"][idx])                            
        temp_list.append(sub_list )
        sub_list = []  
        
        while flag2:
            next_idx = idx + 1
            if len(train)  > next_idx :
               
                if not train["Day"][idx] == train["Day"][next_idx]:
                    flag2 = False
                else:
                    if (train["Location"][idx] == 2 ) or  (not train["Bathroom Light"][idx] == train["Bathroom Light"][next_idx]) or (not train["Kitchen Light"][idx] == train["Kitchen Light"][next_idx]) or (not train["TV"][idx] == train["TV"][next_idx]):
                        if temp_list and len(temp_list) > 1: 
                            sub_list.append(train["Location"][next_idx])
                            sub_list.append(train["Day"][next_idx])
                            sub_list.append(train["Time(Hours)"][next_idx])
                            sub_list.append(train["Time(Minutes)"][next_idx])                            
                            temp_list.append(sub_list )
                            sub_list = []
                            
                            input_sequences.append(temp_list)
                        temp_list = []
                        idx += 1
                    else:
                        
                        sub_list.append(train["Location"][next_idx])
                        sub_list.append(train["Day"][next_idx])
                        sub_list.append(train["Time(Hours)"][next_idx])
                        sub_list.append(train["Time(Minutes)"][next_idx])                            
                        temp_list.append(sub_list )
                        sub_list = []
                        idx += 1
            else:
                flag2 = False
        if temp_list and len(temp_list) > 1:            
            input_sequences.append(temp_list)
        idx += 1
    else:
         flag1 = False

final_array = []
count_array = []       
for x in input_sequences:
    count = 0
    for y in x:
        count += 1
        final_array.append(y)
    count_array.append(count)
        

data = np.loadtxt('train.csv' , delimiter=',')
sample_vector = np.array(final_array)
sequence_lengths = np.array(count_array)
num_components = 3
model = MultinomialHMM(n_components=num_components  , n_iter = 1000)
model.fit(sample_vector , lengths = sequence_lengths)


#------------------------------------------------------------------------------------------------------------------------
print("Second Phase")
validating_sequences = []
# for idx in train.index:
count =0
idx = 0  
flag1 = True
flag2 = True   

while flag1:
    temp_list = []
    flag2 = True
    
    if len(valid) > idx :
        sub_list.append(valid["Location"][idx])
        sub_list.append(valid["Day"][idx])
        sub_list.append(valid["Time(Hours)"][idx])
        sub_list.append(valid["Time(Minutes)"][idx])                            
        temp_list.append(sub_list )
        sub_list = []        
        while flag2:
            next_idx = idx + 1
            if len(valid)  > next_idx :
               
                if not valid["Day"][idx] == valid["Day"][next_idx]:
                    flag2 = False
                else:
                    if (valid["Location"][idx] == 2 ) or  (not valid["Bathroom Light"][idx] == valid["Bathroom Light"][next_idx]) or (not valid["Kitchen Light"][idx] == valid["Kitchen Light"][next_idx]) or (not valid["TV"][idx] == valid["TV"][next_idx]):
                        if temp_list and len(temp_list) > 1:  
                            sub_list.append(valid["Location"][next_idx])
                            sub_list.append(valid["Day"][next_idx])
                            sub_list.append(valid["Time(Hours)"][next_idx])
                            sub_list.append(valid["Time(Minutes)"][next_idx])                            
                            temp_list.append(sub_list )
                            sub_list = []
                            
                            validating_sequences.append(temp_list)
                        temp_list = []
                        idx += 1
                    else:
                        sub_list.append(valid["Location"][next_idx])
                        sub_list.append(valid["Day"][next_idx])
                        sub_list.append(valid["Time(Hours)"][next_idx])
                        sub_list.append(valid["Time(Minutes)"][next_idx])                            
                        temp_list.append(sub_list )
                        sub_list = []
                        idx += 1
            else:
                flag2 = False
        if temp_list and len(temp_list) > 1:            
            validating_sequences.append(temp_list)
        idx += 1
    else:
          flag1 = False

#------------------------------------------------------------------------------------------------------------------------
print("Third Phase")
count = 0
hits = 0
for seq in validating_sequences:
    seq_list = []      
    for inst in seq:
        prob_list = []
        len_list = []
        pred_list = []
        
        if len(seq_list) > 1:
            for i in range(8):
                seq_list.append(i)
                a = np.array(seq_list).reshape(-1,1)
                len_list = []
                len_list.append(len(seq_list))
                b = np.array(len_list)
                prob = model.decode(a , lengths = b)
                prob_list.append(prob[0])
                seq_list.pop()
            max_value = max(prob_list)
            pred_value = prob_list.index(max_value)
            if pred_value == inst[0]:
                hits+=1
            count+=1
            seq_list.append(inst[0])
            
        else:
            seq_list.append(inst[0])
            len_list.append(len(seq_list))
        

print(count)
print(hits)
        
        
        
        
        