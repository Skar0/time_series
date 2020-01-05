import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/train.csv", index_col = "Day", parse_dates = True)

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output2.pdf")
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'size'   : 7}

matplotlib.rc('font', **font)
plt.xticks(rotation=70)

figs = []
n_figs = data.shape[1]

plt.figure(1) # create figure outside loop

for j in range(n_figs): # create all figures
    series= data.iloc[:, j]
    plt.figure(j)
    plt.suptitle("figure {}" .format(j+1))
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        if (i == 0) or (i == 2):
            plt.plot(series)
        elif (i == 1) or (i == 3):
            med = np.median(series)
            q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
            iqr = q75 - q25
            cut_off = iqr * 1.8
            upper = q75 + cut_off
            outliers = series > upper
            series[outliers] = np.nan
            series.fillna(med, inplace=True)
            plt.plot(series)
    pdf.savefig(j) # save on the fly
    plt.close() # close figure once saved

pdf.close()