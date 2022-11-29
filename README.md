# Estimacija pokreta
Estimacija pokreta pomoću optical flow je projekat rađen na letnjem kampu za stare polaznike 2022. godine od Milice Gojak i Novaka Stijepića.

PARAMETRI

Za "daisy i flann.py" postoji 3 parametra:

(1) broj para slike koji se obrađuje. NE SME biti veći od 99

(2) broj backward - 0 ako se traže vektori u smeru napred i 1 ako se traze unazad

(3) dopython - uvek je 1 ako odlučimo da se BCD radi u Pythonu i uvek 0 ako proradi kôd u C++. Za sada je uvek 1


Za "python bcd.py" postoji 3 parametra:

(1) broj para slike koji se obrađuje. NE SME biti veći od 99

(2) broj backward - 0 ako se traže vektori u smeru napred i 1 ako se traze unazad (isti kao malopre)

(3) bcd_times - broj BCD algoritama koje želimo da izvršimo.



Dakle za jedan par slika (npr. par broj 6 u bazi podataka) bi poziv izgledao ovako:

Obrađivanje unapred:

python "daisy i flann.py" 6 0 1

python "bcd.py" 6 0 4

Obrađivanje unazad:

python "daisy i flann.py" 6 1 1

python "bcd.py" 6 1 4


Obrađivanje unapred i unazad su nezavisni procesi i mogu se paralelizovati 

"visualisation.py":  u srednja_greska.txt i procenat_outliera.txt cuva metriku 

Parametri:

(1) path fajla sa ground truthom

(2) path fajla ciji se flow field testira

(3) (opciono) path gde se cuva vizuelna reprezentacija slike kao .jpg fajl

"spremiZaEpic.py" : Pokrece postprocessing( koji sada samo izvrava consistencyCheck), racuna argumente potrebne za EpicFlow (pronalazi ivice slike i pravi txt fajl u kome u svakom redu nalazi po 2 para koordinata koji opisuju prelazak jedne slike u drugu)
i pokrece EpicFlow

Parametri:

(1) path referentne slike

(2) path ciljane slike

(3) flow field unapred

(4) flow field unazad

(5) consistency threshold (u ref radu =10)

(6) vrsta algoritma pronalazenja ivica (sed ili canny)
