import glob
import pandas as pd

from img_preprocess_project import pre_processing_raw
from VGG_block_pool_4 import vgg_block4

import time

# record start time
t0 = time.time()

# retrieve file paths and luminosity
files = glob.glob('../diid17/data/google_image/*/*_*.png')
df_files = pd.DataFrame(files, columns=['path'])
df = pd.merge(df_files, df_files.path.str.split(pat='/', expand=True)[[4,5]], left_index=True, right_index=True)
df.rename(columns={4:'intensity', 5:'filename'}, inplace=True)
df['intensity'] = df['intensity'].astype(int)
df['light_group'] = pd.cut(df['intensity'], [-1, 3, 35, 64], labels=['low', 'medium', 'high'])
df = pd.merge(df, pd.get_dummies(df['light_group'], prefix='light'), left_index=True, right_index=True)
pd.crosstab(df.intensity, df.light_group)

# iterate through all files
#df = df.head(5) # test
X = []
for i, row in df.iterrows():
    print('image %s of %s' % (i, len(df)))
    img_array = pre_processing_raw(row['path'])
    X.append(img_array)

vgg_block4(X)

# record finish time
t1 = time.time() - t0
print('total time elapsed: %s seconds' % t1)
