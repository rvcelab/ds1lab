import pandas as pd
import seaborn as sb
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data=pd.read_csv('basket.csv')

data=data.fillna('')

data.isna().sum()

oht=pd.get_dummies(data)

frequent_item_sets=apriori(oht,min_support=0.4,use_colnames=True)

association_rules_data=association_rules(frequent_item_sets,metric='confidence',min_threshold=0.7)
sb.scatterplot(data=association_rules_data,x='support',y='confidence',hue='lift',palette='colorblind')

frequent_item_sets

association_rules_data