import cv2
import numpy as np

def skaluj_obraz(obraz, wspolczynnik_skalowania):
    rozmiar_jadra = int(1/wspolczynnik_skalowania) * 2 + 1
    jadro = np.ones((rozmiar_jadra, rozmiar_jadra), np.float32) / (rozmiar_jadra**2)
    obraz_skalowany = cv2.filter2D(obraz, -1, jadro)
    return obraz_skalowany

def interpolacja_sasiada(obraz, wspolczynnik_skalowania):
    wysokosc, szerokosc, _ = obraz.shape
    nowa_wysokosc = int(wysokosc * wspolczynnik_skalowania)
    nowa_szerokosc = int(szerokosc * wspolczynnik_skalowania)
    obraz_skalowany = np.zeros((nowa_wysokosc, nowa_szerokosc, 3), dtype=np.uint8)

    for i in range(nowa_wysokosc):
        for j in range(nowa_szerokosc):
            x = int(j / wspolczynnik_skalowania)
            y = int(i / wspolczynnik_skalowania)
            obraz_skalowany[i, j] = obraz[y, x]

    return obraz_skalowany

def interpolacja_dwuliniowa(obraz, k_skalowania):
    wysokosc, szerokosc, _ = obraz.shape
    nowa_wysokosc = int(wysokosc * k_skalowania)
    nowa_szerokosc = int(szerokosc * k_skalowania)
    obraz_skalowany = np.zeros((nowa_wysokosc, nowa_szerokosc, 3), dtype=np.uint8)

    for i in range(nowa_wysokosc):
        for j in range(nowa_szerokosc):
            x = j / k_skalowania
            y = i / k_skalowania
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, szerokosc - 1), min(y0 + 1, wysokosc - 1)
            fx, fy = x - x0, y - y0
            for c in range(3): 
                interp_wartosc = (1 - fx) * (1 - fy) * obraz[y0, x0, c] + \
                fx * (1 - fy) * obraz[y0, x1, c] + \
                (1 - fx) * fy * obraz[y1, x0, c] + \
                fx * fy * obraz[y1, x1, c]
                obraz_skalowany[i, j, c] = int(interp_wartosc)

    return obraz_skalowany

def max_pooling(obraz1, rozmiar_poolingu):
    wysokosc, szerokosc, _ = obraz1.shape
    nowa_wysokosc = wysokosc // rozmiar_poolingu
    nowa_szerokosc = szerokosc // rozmiar_poolingu
    obraz_poolowany = np.zeros((nowa_wysokosc, nowa_szerokosc, 3), dtype=np.uint8)

    for i in range(nowa_wysokosc):
        for j in range(nowa_szerokosc):
            wierszs, wierszk = i * rozmiar_poolingu, (i + 1) * rozmiar_poolingu
            kolumnas, kolumnak = j * rozmiar_poolingu, (j + 1) * rozmiar_poolingu
            region = obraz1[wierszs:wierszk, kolumnas:kolumnak, :]
            obraz_poolowany[i, j, :] = np.max(region, axis=(0, 1))

    return obraz_poolowany

def skaluj_obraz1(obraz1, wspolczynnik_skalowania):
    if wspolczynnik_skalowania > 1:
        return interpolacja_dwuliniowa(obraz1, wspolczynnik_skalowania)
    else:
        rozmiar_poolingu = int(1 / wspolczynnik_skalowania)
        return max_pooling(obraz1, rozmiar_poolingu)

def sekwencyjnie(obraz, wspolczynnik_skalowania):
    if wspolczynnik_skalowania > 1:
        return interpolacja_dwuliniowa(obraz, wspolczynnik_skalowania)
    elif wspolczynnik_skalowania < 1:
        rozmiar_poolingu = int(1 / wspolczynnik_skalowania)
        return max_pooling(obraz, rozmiar_poolingu)
    else:
        return obraz

def mse(oryginal, przerobiony):
    min_wysokosc = min(oryginal.shape[0], przerobiony.shape[0])
    min_szerokosc = min(oryginal.shape[1], przerobiony.shape[1])
    oryginal = oryginal[:min_wysokosc, :min_szerokosc, :]
    przerobiony = przerobiony[:min_wysokosc, :min_szerokosc, :]
    return np.mean((oryginal - przerobiony)**2)


oryginalny_obraz = cv2.imread('C:/Users/matiu/Downloads/lew.jpg')
wspolczynnik_skalowania = 0.5
obraz_skalowany = skaluj_obraz(oryginalny_obraz, wspolczynnik_skalowania)
obraz_skalowany1 = skaluj_obraz1(oryginalny_obraz, wspolczynnik_skalowania)

bezposrednio = interpolacja_dwuliniowa(oryginalny_obraz, wspolczynnik_skalowania)
obraz_sekwencyjnie = sekwencyjnie(oryginalny_obraz, wspolczynnik_skalowania)

mse_bezposrednio = mse(oryginalny_obraz, bezposrednio)
mse_sekwencyjnie = mse(oryginalny_obraz, obraz_sekwencyjnie)
mse_skalowanego = mse(oryginalny_obraz, obraz_skalowany)

wspolczynnik_skalowania1 = 1.5
powiekszony_obraz = interpolacja_dwuliniowa(oryginalny_obraz, wspolczynnik_skalowania1)
powiekszony_obraz1 = interpolacja_sasiada(oryginalny_obraz, wspolczynnik_skalowania1)
mse_od_powiekszonego = mse(oryginalny_obraz, powiekszony_obraz)

cv2.imshow('Oryginalny obraz', oryginalny_obraz)
cv2.imshow('Skalowany obraz jadro usredniajace', obraz_skalowany)
cv2.imshow('Skalowany obraz max pooling', obraz_skalowany1)
cv2.imshow('Skalowany obraz bezposrednio', bezposrednio)
cv2.imshow('Skalowany obraz sekwencyjnie', obraz_sekwencyjnie)
cv2.imshow('Powiekszony obraz interpolacja dwuliniowa', powiekszony_obraz)
cv2.imshow('Powiekszony obraz interpolacja najblizszym sasiadem', powiekszony_obraz1)
print(f'MSE od oryginału do pomniejszonego: {mse_skalowanego:.4f}')
print(f'MSE od oryginału do powiększonego: {mse_od_powiekszonego:.4f}')
print(f'MSE bezposrednio: {mse_bezposrednio:.4f}')
print(f'MSE sekwencyjnie: {mse_sekwencyjnie:.4f}')

cv2.waitKey(0)
cv2.destroyAllWindows()
