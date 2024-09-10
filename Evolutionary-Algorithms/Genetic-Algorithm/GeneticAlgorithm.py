import numpy as np

def İNSAN(n):
    return [np.random.choice([0,1]) for _ in range(n)]

def UYUM(İNSAN):
    return np.mean(İNSAN)

def MUTASYON(İNSAN, p):
    if np.random.rand() < p:
        m = np.random.choice(len(İNSAN))
        İNSAN[m] = 1 - İNSAN[m]

def OLASILIK(TOPLUM, UYUM=UYUM, elitist=True):
    UYUMLAR = np.array([UYUM(İNSAN) for İNSAN in TOPLUM])
    if elitist:
        UYUMLAR = np.exp(UYUMLAR / UYUMLAR.mean())
    GENELOLASILIK = UYUMLAR / UYUMLAR.sum()
    return GENELOLASILIK

def SEÇİM(GENELOLASILIK):
    return np.random.choice(len(GENELOLASILIK), 2, replace=False, p=GENELOLASILIK)

def ÇAPRAZLAMA(TOPLUM, SEÇİM):
    İNSAN0 = TOPLUM[SEÇİM[0]]
    İNSAN1 = TOPLUM[SEÇİM[1]]
    n = len(İNSAN0) // 2
    return np.hstack((İNSAN0[:n], İNSAN1[n:]))

def YENİ_TOPLUM(TOPLUM, OLASILIKLAR, MUTASYON=MUTASYON, p=0.05):
    k = len(TOPLUM) // 2
    ÇEKİNGEN = OLASILIKLAR.argsort()[:k]  # En kötü k birey
    for i in range(k):
        s = SEÇİM(OLASILIKLAR)
        YENİ_İNSAN = ÇAPRAZLAMA(TOPLUM, s)
        MUTASYON(YENİ_İNSAN, p)
        TOPLUM[ÇEKİNGEN[i]] = YENİ_İNSAN
    return TOPLUM

def EN_İYİ(TOPLUM, OLASILIKLAR):
    ENİYİ = OLASILIKLAR.argmax()
    return TOPLUM[ENİYİ]

N = 10
n = 10
TOPLUM = np.array([İNSAN(n) for _ in range(N)])
OLASILIKLAR = OLASILIK(TOPLUM)

if N < 20:
    print(TOPLUM)
    print(OLASILIKLAR)
    print(EN_İYİ(TOPLUM, OLASILIKLAR))

for i in range(1000):
    TOPLUM = YENİ_TOPLUM(TOPLUM, OLASILIKLAR)
    OLASILIKLAR = OLASILIK(TOPLUM)

if N < 20:
    print(TOPLUM)
    print(OLASILIKLAR)
    print(EN_İYİ(TOPLUM, OLASILIKLAR))
