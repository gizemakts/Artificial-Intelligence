## Pandas ile tanımlayıcı analizler, veri sütunlarının birbirliyle ilişkisi, grafik

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv.csv')
print(df)
df.columns
df.head()
df.tail()
df.sample(5)
df.info()
df.shape
df.describe()
df.drop(["iki"],axis=1)
df.groupby('bir').iki.sum()
df.ndim()
df.dtype()
print(df.max())
print(df.sum())
print(df.mean())
print(df.median())
df["bir"].describe()
df.count() 
df['bir'].value_counts().plot.bar().set_title('bir')

from scipy.stats import pearsonr
veri1 = df["bir"]
veri2 =  df["iki"]
istatistik, p=pearsonr(veri1, veri2)
corr, _ = pearsonr(veri1, veri2)
print('Pearsons correlation: %.3f' % corr)

from scipy.stats import mannwhitneyu
veri1= df["bir"]
veri2= df["iki"]
stat, p = mannwhitneyu(veri1, veri2)
print("stat =%.3f, p=%.3f" % (stat,p))
if p > 0.05:
     print("aynı dağılımdan olması mümkündür")
else:
     print("farklı dağılımlardandır")

from scipy.stats import wilcoxon
veri1= df["bir"]
veri2= df["iki"]
stat, p = wilcoxon (veri1, veri2) 
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05: 
    print('aynı dağılımdan')
else: 
    print('farklı dağılımdan')

from scipy.stats import spearmanr
veri1= df["bir"]
veri2= df["iki"]
istatistik, p=spearmanr(veri1, veri2)
print("istatistik = %.3f, p=%.3f" % (istatistik, p))
if p>0.05:
    print("bağımsız olması mümkün")
else:
    print("bağımlı olması mümkündür")


from scipy.stats import chi2_contingency
from scipy.stats import chi2
tablo=[df["bir"],df["iki"]]
print(tablo)
istatistik, p, kritik_değer, beklenen = chi2_contingency(tablo)
print('dof=%d' % kritik_değer)
print(beklenen)
güven_aralığı = 0.95
kritik = chi2.ppf(güven_aralığı, kritik_değer)
print('güven aralığı=%.3f, kritik değer=%.3f, istatistik=%.3f' % (güven_aralığı, kritik_değer, istatistik))
if abs(istatistik) >= kritik_değer:
    print('bağımlı (h0 reddedilir)')
else: 
    print('bağımsız (h0 reddedilmez)')
alfa= 1.0 * güven_aralığı 
print('önem=%.3f, p=%.3f' % (alfa, p))
if p <= alfa:
    print('Bağımlı (H0 reddet)')
else: 
    print('Bağımsız (H0 reddetmez)')

from scipy.stats import ttest_ind
veri1= df["bir"]
veri2= df["iki"]
istatistik, p = ttest_ind (veri1, veri2) 
print('stat=%.3f, p=%.3f' % (istatistik, p))
if p > 0.05:
    print('Muhtemelen aynı dağılım')
else:
    print('Muhtemelen farklı dağılım')
    
from scipy.stats import ttest_rel 
veri1= df["bir"]
veri2= df["iki"]
İstatistik, p = ttest_rel (veri1, veri2)
print( 'İstatistik=%.3f, p=%.3f' % (İstatistik, p))
alpha = 0.05
if p > alpha:
    print( 'Ayni Dağılım (HO Reddedilmez)')
else:
    print('Farklı Dağılım (HO Reddedilir)')

from scipy.stats import kendalltau
veri1= df["bir"]
veri2= df["iki"]
istatistik, p=kendalltau(veri1, veri2)
print("istatistik = %.3f, p=%.3f" % (istatistik, p))
if p>0.05:
    print("bağımsız olması mümkün")
else:
    print("bağımlı olması mümkündür")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset.csv.csv')
# Kolon isimlerini değiştirme
df = df.rename(columns = {"bir" : "bir",
                     "iki" : "iki",
                     "uc" : "uc",
                     "dort" : "dort",
                     "bes" : "bes",
                     "alti" : "alti"})

# shapiro-wilk normallik testi
for i in df.columns:
    _ , p = stats.shapiro(df[i])
    print(f"{i}: {round(p,6)}")
 
bir = df.bir.sort_values()
# dağılım eğrisi
mean, std = stats.norm.fit(bir, loc=0)
pdf_norm = stats.norm.pdf(bir, mean, std)
# Histogram ve dağılım eğrisi
plt.hist(bir, bins='auto', density = True
         ,color = "grey", ec="skyblue")
plt.plot(bir, pdf_norm, label='Dağılım Eğrisi'
         ,color = "red", linewidth=4, linestyle=':')
plt.legend()
plt.show()

# Orjinal değişken
bir = df.bir.sort_values()
# karakök alınmış hali
bir_sr = bir ** (1/2)
# orjinal dağılım eğrisi
mean, std = stats.norm.fit(bir, loc=0)
pdf_norm = stats.norm.pdf(bir, mean, std)
# orjinal dağılım eğrisi
mean, std = stats.norm.fit(bir_sr, loc=0)
pdf_norm_sk = stats.norm.pdf(bir_sr, mean, std)

# Grafikler
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# Orjinal veriler ve dağılımı
ax1.hist(bir, bins= 20 , density=True
         ,color = "grey", ec= "skyblue")
ax1.plot(bir, pdf_norm
         ,color = "red", linewidth=4, linestyle=':')
ax1.set_ylabel('Olasılık')
ax1.set_title('Orjinal Boyut')
# karakök dönüşüm grafiği
ax2.hist(bir_sr, bins= 20, density=True
         ,color = "green", ec="skyblue")
ax2.plot(bir_sr, pdf_norm_sk
         ,color = "red", linewidth=4, linestyle=':')
ax2.set_title('Karakök Dönüşüm')
plt.show()