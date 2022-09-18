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


python "daisy i flann.py" 6 0 1

python "bcd.py" 6 0 4



python "daisy i flann.py" 6 1 1

python "bcd.py" 6 1 4


Obrađivanje unapred i unazad su nezavisni procesi i mogu se paralelizovati 
