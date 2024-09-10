import pandas as pd 

dataset = pd.read_csv("diamonds.csv")
data1= dataset['carat']
data2= dataset['depth']

###################################### Pearson Korelasyon Katsayısı ###################################
from scipy.stats import pearsonr

istatistik, p=pearsonr(data1, data2)
print('istatistik =%.3f, p=%.3f' % (istatistik, p))

if p>0.05:
    print('Bağımsız olması mümkün.')
else:
    print('Bağımlı olması mümkümdür.')
    
# Burada çıkan sonuç istatistik =0.028, p=0.000 Bağımlı olması mümkümdür.
    

############################### Spearman Sıralama Korelasyonu #########################################
from scipy.stats import spearmanr

istatistik, p=spearmanr(data1, data2)
print('istatistik =%.3f, p=%.3f' % (istatistik, p))

if p>0.05:
    print('Bağımsız olması mümkün.')
else:
    print('Bağımlı olması mümkümdür.')
    
# Burada çıkan sonuç istatistik =0.030, p=0.000 Bağımlı olması mümkümdür.

################################# Kendall Sıralama Korelasyonu ########################################
from scipy.stats import kendalltau

istatistik, p=kendalltau(data1, data2)
print('istatistik =%.3f, p=%.3f' % (istatistik, p))

if p>0.05:
    print('Bağımsız olması mümkün.')
else:
    print('Bağımlı olması mümkümdür.')

# Burada çıkan sonuç istatistik =0.020, p=0.000 Bağımlı olması mümkümdür.

################################# Chi-Squared (Ki Kare) Testi ########################################
from scipy.stats import chi2_contingency

olasılık_tablosu = pd.crosstab(pd.cut(data1, bins=10), pd.cut(data2, bins=10))

istatistik, p, özgürlük_katsayısı, beklenen = chi2_contingency(olasılık_tablosu)
print('istatistik =%.3f, p=%.3f' % (istatistik, p))

if p>0.05:
    print('Bağımsız olması mümkün.')
else:
    print('Bağımlı olması mümkümdür.')

# Burada çıkan sonuç istatistik =14868.735, p=1.000 Bağımsız olması mümkündür.
