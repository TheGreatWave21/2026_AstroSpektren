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

#### Wissenschaftlicher Hintergrund

**Astronomischer Hintergrund**: Sterne sind selbstleuchtende Himmelskörper, die ihre Energie durch Kernfusionsprozesse in ihrem Inneren erzeugen. Im Zentrum der meisten Sterne verschmelzen Wasserstoffkerne zu Helium, wobei große Energiemengen freigesetzt werden. Diese Energie wird im Stern transportiert und schließlich als elektromagnetische Strahlung von der Photosphäre abgestrahlt. Die sichtbare Strahlung eines Sterns hängt wesentlich von seiner Oberflächentemperatur und der Beschaffenheit seiner Photosphäre ab. Deshalb lassen sich aus Sternspektren grundlegende physikalische Eigenschaften ableiten.

**Spektralklassen von Sternen**: Sterne werden anhand ihrer spektralen Eigenschaften in die Klassen O, B, A, F, G, K und M eingeteilt. Diese reichen von sehr heißen O-Sternen mit Temperaturen über 30.000 K bis hin zu kühlen M-Sternen mit Temperaturen unter 3.500 K. Die Klassifikation basiert auf charakteristischen Absorptionslinien im Spektrum. Je nach Temperatur treten unterschiedliche Elemente und Ionisationszustände besonders deutlich hervor.

**Schwarzkörperstrahlung**: Das Strahlungsverhalten von Sternen lässt sich in erster Näherung durch das Schwarzkörpermodell beschreiben. Ein idealer Schwarzkörper absorbiert sämtliche einfallende Strahlung und emittiert Strahlung ausschließlich abhängig von seiner Temperatur. Die spektrale Strahldichte eines idealen Schwarzkörpers wird durch das Plancksche Strahlungsgesetz beschrieben. Mit steigender Temperatur verschiebt sich das Maximum der Intensitätsverteilung zu kürzeren Wellenlängen. Dieser Zusammenhang wird durch das Wiensche Verschiebungsgesetz beschrieben.

**Absorptions- und Balmerlinien**: Das kontinuierliche Schwarzkörperspektrum eines Sterns wird in den äußeren Schichten seiner Atmosphäre teilweise absorbiert. Dadurch entstehen Absorptionslinien, die Rückschlüsse auf die chemische Zusammensetzung und Temperatur erlauben. Besonders wichtig ist die Balmer-Serie des Wasserstoffs mit den Linien: 
- H_alpha bei 6560–6562 Å
- H_beta bei 4860–4861 Å
- H_gamma bei 4340 Å
- H_omega bei 4100 Å
Außerdem spielt die Balmerkante bei etwa 3646 Å eine zentrale Rolle.

**Effektive Temperatur**: Die effektive Temperatur eines Sterns ist eine zentrale physikalische Größe. Sie ist über Leuchtkraft, Radius und die Stefan-Boltzmann-Konstante definiert und beschreibt die Temperatur eines idealisierten Schwarzkörpers mit derselben Gesamtstrahlungsleistung.

**Spektrographen und Beugungsgitter**: In dieser Arbeit wurde ein Objektivgitter verwendet. Ein Beugungsgitter zerlegt einfallendes Licht durch Beugung und Interferenz in verschiedene Richtungen. Dabei entstehen sogenannte Beugungsordnungen. Die nullte Ordnung entspricht dem ungebeugten Licht, während die erste Ordnung das für die Spektralanalyse genutzte Spektrum enthält. Verwendet wurde ein Star Analyser SA100, das direkt vor dem Objektiv positioniert wurde.
