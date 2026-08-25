# Cloney

Lokales Voice Cloning für deutsche Langform-Texte. Läuft vollständig auf der
eigenen Maschine — kein Dienst, kein API-Schlüssel, keine Daten nach außen.

Vorbild ist [HiggsAudio-Studio](https://github.com/timoncool/HiggsAudio-Studio).
Cloney entsteht, weil zwei Anforderungen dort offen bleiben:

**Deutsch-Qualität.** Kein TTS-Modell spricht „Am 3. Mai 2024 kostete es
1.250,50 €" von sich aus korrekt aus. Cloney schreibt Ziffern, Symbole und
Abkürzungen vorher aus — regelbasiert, reproduzierbar und mit Tabellentests
abgesichert. Ordinalzahlen werden nach dem vorangehenden Wort dekliniert
(„am **dritten** Mai", „der **dritte** Platz"), Jahreszahlen in der üblichen
Hunderter-Lesung gesetzt („neunzehnhundertvierundachtzig").

**Langform-Stabilität.** Über ein ganzes Kapitel driftet die Stimme, wenn jeder
Abschnitt auf dem vorherigen aufbaut. Cloney konditioniert **jeden** Chunk gegen
dieselbe unveränderte Referenz und speichert den Seed. Dadurch ist jeder Satz
einzeln und identisch reproduzierbar — die Voraussetzung für Wiederaufnahme nach
einem Abbruch und für das gezielte Neuwürfeln eines einzelnen misslungenen Satzes.

## Wie es arbeitet

```
Text ─▶ Sätze ─▶ Normalisierung ─▶ Chunks ─▶ Synthese ─▶ Rückschrift ─▶ Zusammenbau
                                                │            │
                                          (dieselbe      (Fehlerrate zu hoch?
                                           Referenz,      neuer Seed, bis zu
                                           fester Seed)   n Versuchen)
```

Der Lauf ist in Phasen getrennt, weil auf einer Karte mit 8 bis 16 GB VRAM nie
zwei große Modelle gleichzeitig im Speicher liegen dürfen:

| Phase | Was geladen ist | Was passiert |
|---|---|---|
| SYNTH | TTS-Modell | alle offenen Chunks, dann entladen |
| QC | Spracherkennung | Rückschrift und Fehlerrate für alle, dann entladen |
| RETRY | TTS-Modell | auffällige Chunks mit neuem Seed |
| ASSEMBLE | — | Lautheit, Pausen, fertige Spur |

Der gesamte Zustand steht im Projekt-Manifest auf Platte und wird nach jedem
Chunk atomar geschrieben. Es gibt keinen zweiten Zustand, der auseinanderlaufen
könnte — CLI, Web-UI und Wiederaufnahme lesen dieselbe Datei.

## Installation

```bash
uv venv --python 3.11
uv pip install -e ".[dev,asr]"
```

Der `asr`-Extra bringt faster-whisper für die Qualitätskontrolle mit. Ohne ihn
läuft alles außer der Fehlermessung; das Manifest vermerkt dann `cer = null`,
also „nicht geprüft" statt „fehlerfrei".

## Schnellstart

```bash
# Verfügbare Engines mit VRAM-Bedarf und Lizenz der Gewichte
cloney engines

# Referenzstimme anlegen. Die Aufnahme wird sofort geprüft --
# eine schlechte Referenz ist die häufigste Ursache für einen schlechten Klon.
cloney voices add --audio referenz.wav --name erzaehlstimme --auto-transcript

# Kapitel rendern
cloney render --text kapitel1.txt --voice erzaehlstimme --engine higgs

# Unterbrochenen Lauf fortsetzen -- fertige Chunks bleiben unberührt
cloney resume 20260825-105939-kapitel-1

# Weboberfläche mit dem Satz-Editor
cloney web
```

Für die Higgs-Engine muss der Modellserver laufen:

```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

## Der Satz-Editor

Die tragende Ansicht der Weboberfläche ist eine Zeile pro Satz mit Status,
Fehlerrate, der Rückschrift der Spracherkennung und einem eigenen Abspieler.
Von dort lässt sich ein einzelner Satz neu würfeln oder umformulieren, ohne das
ganze Kapitel neu zu rendern. Genau an diesem Arbeitsschritt scheitert eine
Langform-Produktion sonst.

Sichtbar sind pro Satz drei Fassungen: der Rohtext, die daraus erzeugte
Sprechfassung und das, was die Spracherkennung tatsächlich gehört hat. Weichen
die letzten beiden voneinander ab, ist das der konkrete Hinweis, wo es hakt.

## Engines

| Engine | VRAM | Lizenz der Gewichte | Anmerkung |
|---|---|---|---|
| `dummy` | — | — | synthetisches Testsignal, für Entwicklung und CI ohne GPU |
| `higgs` | ~11 GB | Research & Non-Commercial | Higgs Audio v3 (4B) über lokalen SGLang-Omni-Server, versteht Inline-Tags |

Higgs v3 passt in bf16 **nicht** auf eine 8-GB-Karte. Für kleinere Karten sind
ein deutscher F5-TTS-Finetune und Chatterbox Multilingual vorgesehen; beide
brauchen unter 4 GB.

Die Lizenz der Gewichte steht in der Oberfläche, nicht nur in dieser Datei — sie
entscheidet darüber, wofür das Ergebnis verwendet werden darf. Der Code von
Cloney ist davon unabhängig.

Voice Cloning nur mit Einverständnis der Person, deren Stimme geklont wird.

## Konfiguration

Alle Werte lassen sich über `CLONEY_*`-Umgebungsvariablen oder eine `.env`-Datei
setzen; die Standardwerte stehen in [`cloney/config.py`](cloney/config.py).

| Variable | Standard | Bedeutung |
|---|---|---|
| `CLONEY_DATA_DIR` | `./data` | Projekte, Stimmen, Ausgaben |
| `CLONEY_ENGINE` | `dummy` | voreingestellte Engine |
| `CLONEY_CER_THRESHOLD` | `0.10` | ab welcher Fehlerrate ein Satz markiert wird |
| `CLONEY_MAX_RETRIES` | `2` | Neuversuche je Satz, bevor er zur Durchsicht bleibt |
| `CLONEY_TARGET_CHUNK_SECONDS` | `20.0` | angestrebte Chunk-Länge |
| `CLONEY_TARGET_LUFS` | `-16.0` | Ziel-Lautheit der fertigen Spur |
| `CLONEY_HIGGS_BASE_URL` | `http://localhost:8000/v1` | Adresse des Modellservers |
| `CLONEY_ASR_MODEL` | `large-v3-turbo` | Whisper-Modell für die Rückschrift |

## Entwicklung

```bash
pytest          # läuft vollständig ohne GPU und ohne Netz
ruff check .
ruff format .
```

`DummyEngine` und `DummyASR` bilden ein geschlossenes Paar: die Engine kodiert
eine Kennung quantisierungsfest in die ersten Samples, die Spracherkennung liest
sie zurück. Dadurch sind Segmentierung, Manifest, Wiederaufnahme, Metriken,
Retry-Schleife, Zusammenbau und sämtliche Web-Routen prüfbar, ohne ein einziges
Modell zu laden.

## Stand

Fertig: deutsche Normalisierung, Segmentierung, Projekt-Manifest mit
Wiederaufnahme, Phasenmodell, Qualitätskontrolle mit Retry-Schleife, Zusammenbau
mit Lautheitsangleichung, CLI und Weboberfläche mit Satz-Editor.

Als Nächstes: LLM-gestützte Textvorbereitung für das, was Regeln nicht können
(Fremdwörter, Eigennamen), der Emotions-Director, weitere Engines für
8-GB-Karten und ein Benchmark-Harness, das die Engines auf einem deutschen
Testsatz vergleicht statt sie nach Gefühl auszuwählen.

Das Anfrageschema der Higgs-Engine ist gegen die öffentliche Dokumentation
gebaut, aber nicht gegen einen laufenden Server verifiziert. Antwortet der
Server mit einem Feldfehler, gibt Cloney dessen Antwort unverändert weiter.
