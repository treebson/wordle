# todo: populate

import config
import words

df = words.df
df = df.sort_values(by='freq', ascending=False)
print(df.word[:10].tolist())