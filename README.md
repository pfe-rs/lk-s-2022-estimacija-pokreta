# Estimacija pokreta
Estimacija pokreta pomoću optical flow je projekat rađen na letnjem kampu za stare polaznike 2022. godine od Milice Gojak i Novaka Stijepića.

PARAMETRI

Za daisy i flann.py postoji 3 parametra:
-broj para slike koji se obrađuje. NE SME biti veći od 99
-broj backward - 0 ako se traže vektori u smeru napred i 1 ako se traze unazad
-dopython - uvek je 1 ako odlučimo da se BCD radi u Pythonu i uvek 0 ako proradi kôd u C++

Za pyton bcd.py postoji 3 parametra:
-broj para slike koji se obrađuje. NE SME biti veći od 99
-broj backward - 0 ako se traže vektori u smeru napred i 1 ako se traze unazad (isti kao malopre)
-bcd_times - broj BCD algoritama koje želimo da izvršimo.
