from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
student_performance = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 


# Show up to 200 rows/columns
pd.set_option("display.max_rows", 200)       # show up to 200 rows
pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.width", 1000)         # avoid line breaks
pd.set_option("display.colheader_justify", "left")

print(X)
#