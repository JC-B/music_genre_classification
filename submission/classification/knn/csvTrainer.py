#import pandas as pd

#data = pd.read_csv('out2.csv', sep='|')

#dataset_array = data.values
#print(dataset_array.dtype)
#print(dataset_array)
import pandas as pd

data = pd.read_csv('out.csv', sep='|')

dataset_array = data.values

dataset = list(dataset_array)
print(dataset)