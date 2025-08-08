import pandas as pd
data=pd.read_csv('raw.csv')
data['ProductPrice']=2*data['ProductPrice']
data.to_csv('raw.csv', index=False)
