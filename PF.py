import pandas as pd
af = pd.read_csv('af1.csv',index_col=None,header=None)
new_df = pd.Dataframe()
lines = []
for i in range(4):
    cos = 3*i
    aline = []
    for ids in af.index:
        aline.append(af.loc[ids,0+cos])
        aline.append(af.loc[ids,1+cos])
        aline.append(af.loc[ids,2+cos])
    lines.append(aline)
print(lines)
new_df = pd.DataFrame(lines)
