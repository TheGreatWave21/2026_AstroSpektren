# Lesen im Sternenlicht – Sternanalyse mittels Spektroskopie

Repository zur Dokumentation des Projekts **„Lesen im Sternenlicht – Sternanalyse mittels Spektroskopie“** von **Bennet Eisfeld & Kenan Busch**.

Dieses Repository bündelt die schriftliche Arbeit als PDF sowie die vollständige Python-Auswertung, damit Vorgehen, Datenverarbeitung und Ergebnisse nachvollziehbar dokumentiert und reproduzierbar gemacht werden können.

---

## Projektüberblick

In diesem Projekt wurden Eigenschaften von Sternen, insbesondere ihre **Temperatur**, **chemische Zusammensetzung** und **Spektralklasse**, mithilfe der **Spektroskopie** untersucht. Dazu wurde das von Sternen ausgesandte Licht mit einem Objektivgitter aufgespalten und analysiert.

Ziel war es, mit einem vergleichsweise **einfachen und kostengünstigen Amateuraufbau** hochwertige Sternspektren aufzunehmen und daraus wissenschaftlich brauchbare Aussagen abzuleiten. Zusätzlich wurde die gesamte Auswertung mit **selbst entwickelten Python-Programmen** durchgeführt, um alle Verarbeitungsschritte wirklich zu verstehen und kontrollieren zu können.

Ursprünglich sollte die effektive Sterntemperatur vor allem durch einen **Planck-Fit** bestimmt werden. Da reale Sternspektren jedoch deutlich vom idealen Schwarzkörpermodell abweichen, wurde zusätzlich der **B–V-Farbenindex** verwendet. Die Kombination mehrerer Methoden erlaubt eine gegenseitige Überprüfung und erhöht die Zuverlässigkeit der Ergebnisse.

---

## Ziel dieses Repositories

Dieses Repository dient dazu, die komplette Arbeit in einer transparenten und sauberen Form zusammenzuführen. Es soll insbesondere enthalten:

- die vollständige schriftliche Arbeit als PDF
- die komplette Python-Pipeline zur Auswertung der Sternspektren
- gegebenenfalls Rohdaten, Zwischenprodukte und verarbeitete Spektren
- Abbildungen zum Aufbau, zur Auswertung und zu den Ergebnissen
- eine nachvollziehbare Struktur, damit andere den Workflow verstehen und reproduzieren können

---

## Wissenschaftlicher Hintergrund

**Astronomischer Hintergrund**: Sterne sind selbstleuchtende Himmelskörper, die ihre Energie durch Kernfusionsprozesse in ihrem Inneren erzeugen. Im Zentrum der meisten Sterne verschmelzen Wasserstoffkerne zu Helium, wobei große Energiemengen freigesetzt werden. Diese Energie wird im Stern transportiert und schließlich als elektromagnetische Strahlung von der Photosphäre abgestrahlt. Die sichtbare Strahlung eines Sterns hängt wesentlich von seiner Oberflächentemperatur und der Beschaffenheit seiner Photosphäre ab. Deshalb lassen sich aus Sternspektren grundlegende physikalische Eigenschaften ableiten.

**Spektralklassen von Sternen**: Sterne werden anhand ihrer spektralen Eigenschaften in die Klassen O, B, A, F, G, K und M eingeteilt. Diese reichen von sehr heißen O-Sternen mit Temperaturen über 30.000 K bis hin zu kühlen M-Sternen mit Temperaturen unter 3.500 K. Die Klassifikation basiert auf charakteristischen Absorptionslinien im Spektrum. Je nach Temperatur treten unterschiedliche Elemente und Ionisationszustände besonders deutlich hervor.

**Schwarzkörperstrahlung**: Das Strahlungsverhalten von Sternen lässt sich in erster Näherung durch das Schwarzkörpermodell beschreiben. Ein idealer Schwarzkörper absorbiert sämtliche einfallende Strahlung und emittiert Strahlung ausschließlich abhängig von seiner Temperatur. Die spektrale Strahldichte eines idealen Schwarzkörpers wird durch das Plancksche Strahlungsgesetz beschrieben. Mit steigender Temperatur verschiebt sich das Maximum der Intensitätsverteilung zu kürzeren Wellenlängen. Dieser Zusammenhang wird durch das Wiensche Verschiebungsgesetz beschrieben.

**Absorptions- und Balmerlinien**: Das kontinuierliche Schwarzkörperspektrum eines Sterns wird in den äußeren Schichten seiner Atmosphäre teilweise absorbiert. Dadurch entstehen Absorptionslinien, die Rückschlüsse auf die chemische Zusammensetzung und Temperatur erlauben. Besonders wichtig ist die Balmer-Serie des Wasserstoffs mit den Linien: 
> H_alpha bei 6560–6562 Å
> H_beta bei 4860–4861 Å
> H_gamma bei 4340 Å
> H_omega bei 4100 Å
Außerdem spielt die Balmerkante bei etwa 3646 Å eine zentrale Rolle.

**Effektive Temperatur**: Die effektive Temperatur eines Sterns ist eine zentrale physikalische Größe. Sie ist über Leuchtkraft, Radius und die Stefan-Boltzmann-Konstante definiert und beschreibt die Temperatur eines idealisierten Schwarzkörpers mit derselben Gesamtstrahlungsleistung.

**Spektrographen und Beugungsgitter**: In dieser Arbeit wurde ein Objektivgitter verwendet. Ein Beugungsgitter zerlegt einfallendes Licht durch Beugung und Interferenz in verschiedene Richtungen. Dabei entstehen sogenannte Beugungsordnungen. Die nullte Ordnung entspricht dem ungebeugten Licht, während die erste Ordnung das für die Spektralanalyse genutzte Spektrum enthält. Verwendet wurde ein Star Analyser SA100, das direkt vor dem Objektiv positioniert wurde.

> make picture here

## Datenaufnahme

Für Aufnahme und Steuerung wurden zwei Softwaresysteme verwendet:
- CCDciel zur direkten Kamerasteuerung und Bildaufnahme
- ASIAir für GoTo-Funktion und präzise Nachführung
Die ASI 1600MM Pro wurde über CCDciel angesteuert, wodurch die Aufnahmen in Echtzeit betrachtet und die Belichtungszeit optimiert werden konnten. Die ASIAir-Plattform übernahm das automatische Anfahren der Zielsterne sowie die exakte Nachführung während der Aufnahmen. Pro Stern wurden zwischen 10 und 60 Einzelaufnahmen erstellt. Die Belichtungszeiten variierten je nach Helligkeit zwischen wenigen Sekunden und mehreren Minuten. Alle Aufnahmen wurden im FITS-Format gespeichert. Dieses astronomische Standardformat enthält neben den Bilddaten auch wichtige Metadaten wie: 
- Belichtungszeit
- Kameraeinstellungen
- Zeitstempel
- Objektkoordinaten
Die gesamte Datenauswertung erfolgte ausschließlich mit selbst entwickelten Python-Skripten.


##Temperaturbestimmung

Zur Bestimmung der effektiven Temperatur wurden zwei unabhängige Methoden verwendet:
1. **Kontinuumsbasierte Temperaturbestimmung** (Planck-Fit)

Bei dieser Methode wurde das Spektrum oberhalb der Balmerkante bis 7000 Å mit dem Planckschen Strahlungsgesetz gefittet.
Vor dem Fit wurden:
- instrumentelle Korrekturen angewendet
- atmosphärische Korrekturen angewendet
- markante Absorptionslinien, insbesondere Balmerlinien, maskiert

Anschließend wurde die effektive Temperatur 𝑇_eff als freier Parameter variiert, bis die beste Übereinstimmung erreicht war. Die resultierende Temperatur entspricht der Temperatur eines idealisierten Schwarzkörpers, dessen Strahlungsspektrum dem beobachteten Sternspektrum am besten entspricht.

2. **Farbenindexbasierte Temperaturbestimmung** (B–V)
Neben der kontinuierlichen Spektralanalyse wurde die effektive Temperatur auch photometrisch über den B–V-Farbenindex bestimmt.
Dieser beschreibt die Helligkeitsdifferenz eines Sterns im:
- B-Band bei 4450 Å
- V-Band bei 5510 Å
Die Methode ist besonders robust, weil sie auf integrierten Helligkeiten basiert und dadurch weniger empfindlich gegenüber einzelnen Spektralmerkmalen oder Rauschen ist. Für diese Methode wurden ausschließlich atmosphärenkorrigierte Spektren ohne instrumentelle Response-Korrektur verwendet, da die Ballesteros-Formel nur auf dem relativen Intensitätsverhältnis zwischen B- und V-Band basiert. Die instrumentell gemessenen Farbenindizes wurden mithilfe von Referenzsternen linear kalibriert. Die Umrechnung von B–V in die effektive Temperatur erfolgte anschließend mit der Ballesteros-Formel.

##Ergebnisse
**Beobachtete Spektren**
Die Arbeit zeigt normierte Spektren von Sternen verschiedener Spektralklassen im Wellenlängenbereich von 3900–7000 Å.
Die Normierung ermöglicht den direkten Vergleich der spektralen Form unabhängig von der absoluten Helligkeit. Ein zentrales Merkmal ist die Lage des Intensitätsmaximums, das sich mit der effektiven Temperatur systematisch verschiebt:
- heiße Sterne (O, B): Maximum im UV-Bereich, Spektrum steigt zum blauen Ende an
- mittlere Temperaturen (F, G): relativ flacher Verlauf, Maximum im sichtbaren Bereich
- kühle Sterne (K, M): Maximum im nahen Infrarot, Spektrum steigt zum roten Ende an
Die Messungen bestätigen damit qualitativ die theoretischen Erwartungen an die Temperaturabhängigkeit der Sternspektren.

**Methodenvergleich**
Verglichen wurden:
- Planck-Fit
- B–V-Farbenindex
Für O-Sterne ergaben sich Abweichungen von bis zu 60 % vom Literaturwert. In diesem Temperaturbereich über 30.000 K sind beide Methoden ungeeignet, da der Großteil der Strahlung im UV-Bereich außerhalb des gemessenen Wellenlängenbereichs liegt und der B–V-Index seine Sensitivität verliert. Für alle anderen Spektralklassen von B bis M lagen die Temperaturen im Bereich unter Berücksichtigung der statistischen Messfehler.
Die B–V-Methode lieferte systematisch genauere Ergebnisse:

- durchschnittlich 100–200 K geringere Abweichung
- geringe Streuung von etwa 7 %
- statistische Fehler meist nur etwa 10 K
Insgesamt ermöglichen beide Methoden, ergänzt durch spektrale Merkmale, eine erfolgreiche Temperaturbestimmung für den Großteil der untersuchten Sterne.

**Modellgrenzen des Planck-Fits**
Sterne der F- und besonders G-Klasse zeigen die beste Übereinstimmung zwischen Planck-Fit und gemessenem Spektrum, da der Großteil ihrer Strahlung im gemessenen Bereich liegt und dort näherungsweise einer idealisierten Planckkurve folgt. Bei A-Sternen zeigt sich jedoch ein deutliches Problem: Der Planck-Fit liefert teilweise eine Kontinuumstemperatur von etwa 15.000 K, obwohl die effektive Temperatur beispielsweise bei Vega nur etwa 9.500 K beträgt. Als Erklärung wird die Schichtstruktur der Sternatmosphäre diskutiert. Die Photosphäre ist keine einheitliche Fläche, sondern besitzt einen Temperaturgradienten. Verschiedene Wellenlängen entkommen aus unterschiedlich tiefen und damit unterschiedlich heißen Schichten. Zusätzlich wird unterhalb der Balmerkante bei 3646 Å Strahlung durch kontinuierliche Photoionisation stark absorbiert. Dadurch kommt es zu einer Umverteilung der Energie, die das Kontinuum in bestimmten Bereichen verstärken kann. Diese Effekte treten insbesondere bei heißen Sternen mit starken Balmerlinien auf und erklären, warum der Planck-Fit dort systematisch zu hohe Temperaturen liefern kann.

##Fazit

In dieser Arbeit wurde untersucht, inwieweit sich mit einem kostengünstigen Amateuraufbau hochwertige Sternspektren aufnehmen und auswerten lassen und daraus zuverlässige Temperaturbestimmungen möglich sind.
Die aufgenommenen Spektren zeigen eine hohe Qualität und ermöglichen:
- die Klassifizierung nach Spektraltyp anhand charakteristischer Absorptionslinien
- die quantitative Temperaturbestimmung
Getestet wurden zwei unabhängige Methoden:
- Planck-Fit an das Kontinuum
- photometrischer B–V-Farbenindex
Während der Planck-Fit für Sterne mittlerer Temperatur, insbesondere F- und G-Sterne, gute Ergebnisse lieferte, zeigten sich bei A-Sternen systematische Abweichungen durch die Struktur der Sternatmosphäre und die Energieumverteilung ab der Balmerkante.
Der B–V-Index erwies sich insgesamt als robuster und lieferte durchgängig zuverlässigere Temperaturen mit Abweichungen von typischerweise 100–200 K zu Literaturwerten. Die Kombination beider Methoden ermöglicht eine gegenseitige Validierung und erhöht die Verlässlichkeit der Ergebnisse. Für sehr heiße O-Sterne sind beide Ansätze ungeeignet, da der Großteil ihrer Strahlung im UV-Bereich außerhalb des Messbereichs liegt.
Zukünftige Verbesserungen könnten sein:
- Korrektur systematischer Fehler
- Einsatz von Gittern mit höherer Liniendichte
- höhere spektrale Auflösung
- alternative Temperaturbestimmung über Äquivalentbreiten charakteristischer Absorptionslinien
Insgesamt zeigt die Arbeit, dass auch mit einfachen Mitteln wissenschaftlich aussagekräftige Spektroskopie möglich ist.

## Lizenz

Der Code in diesem Repository steht unter der MIT-Lizenz.
Die Dokumentation, das PDF und die Abbildungen stehen unter der Lizenz CC BY 4.0, sofern nicht anders angegeben.
