import pandas as pd

GRGs_data = pd.read_csv('GRGs.csv')
names = GRGs_data['Name'].values

cm_id = []
with open('cross_matched.txt') as f:
     for line in f:
         cm_name = line.split('\n')[0]
         #print(cm_name)
         for m in range(len(names)):
             if cm_name == names[m]:
                cm_id.append(m)
                
GRGs_data = GRGs_data.drop(index=cm_id)
GRGs_data.to_csv('GRGs_final_RAA.csv', index=False)
