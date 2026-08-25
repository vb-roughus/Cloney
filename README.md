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

## Hardware

| Karte | Higgs v3 (~11 GB) | F5-TTS deutsch (~2 GB) |
|---|---|---|
| RTX 5080 / 4080 (16 GB) | passt, ~5 GB Reserve | passt mühelos |
| RTX 3090 / 4090 (24 GB) | passt bequem | passt mühelos |
| 8-GB-Karten | passt nicht | passt |

**Unter Windows steht nur `f5-de` zur Verfügung.** SGLang-Omni, über das Higgs v3
bedient wird, [läuft nicht nativ unter Windows](https://docs.sglang.ai/get_started/install.html) —
es setzt Linux-spezifische CUDA-Kernel voraus. Für Higgs braucht es dort WSL2.
Da `f5-de` ohnehin das auf Deutsch nachtrainierte Modell ist, ist das kein großer
Verlust.

**Blackwell (RTX 50-Serie) braucht eine neuere Toolchain.** Der sm_120-Rechenkern
der 5080 wird erst ab **CUDA 12.8** und **PyTorch 2.7** unterstützt. Ältere
PyTorch-Builds starten zwar, melden aber `no kernel image is available for
execution on the device` oder fallen still auf die CPU zurück. Prüfen mit:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# erwartet: 2.7+ , 12.8+ , (12, 0)
```

Passende Wheels kommen vom cu128-Index:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**FFmpeg wird gebraucht, und zwar als Shared-Build.** Ab torchaudio 2.10 sind die
eigenen Backends entfernt; jedes Laden einer Audiodatei läuft über torchcodec,
und das verlangt die FFmpeg-Bibliotheken. Der übliche Befehl führt dabei in die
Irre: `winget install Gyan.FFmpeg` installiert den *statischen* Build, der nur
`ffmpeg.exe` ablegt. `ffmpeg` steht dann im Suchpfad, torchcodec scheitert
trotzdem. Richtig ist:

```powershell
winget install --id Gyan.FFmpeg.Shared
```

Danach die Konsole neu öffnen, damit der Suchpfad übernommen wird. `cloney doctor`
prüft das durch einen echten Ladeversuch und unterscheidet die beiden möglichen
Ursachen — fehlende Bibliotheken oder eine torchcodec-Fassung, die nicht zur
PyTorch-Version passt.

Auch mit 16 GB liegen TTS-Modell und Text-LLM nie gleichzeitig im Speicher --
das Phasenmodell ist keine Notlösung für kleine Karten, sondern die
Voraussetzung dafür, dass beide Rollen überhaupt zusammenpassen.

## Installation

**Windows** (PowerShell, im Projektordner):

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

**Linux und macOS:**

```bash
./install.sh
```

Der Installer legt die virtuelle Umgebung an, installiert PyTorch aus dem
CUDA-12.8-Index (nötig für RTX-50-Karten, unschädlich für ältere), installiert
Cloney samt Extras, legt die `.env` an, führt die Diagnose aus und startet zum
Schluss die Oberfläche -- der Browser öffnet sich, sobald der Server antwortet.

Argumente werden durchgereicht: `--no-web` startet die Oberfläche nicht,
`--skip-torch` lässt PyTorch unangetastet, `--extras ""` installiert nur den
Kern, `--dry-run` zeigt nur, was liefe. Ohne Terminal im Vordergrund -- etwa in
einem Skript -- bleibt es ohnehin bei der Anleitung, statt zu blockieren.

Von Hand geht es genauso:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,asr,f5]"
cp .env.example .env
```

Die Extras sind einzeln wählbar: `asr` bringt faster-whisper für die
Qualitätskontrolle, `f5` das deutsche F5-TTS-Modell. Ohne beide läuft der Kern.

### Der Befehl `cloney`

`pip install -e .` legt eine ausführbare Datei in der virtuellen Umgebung an:
`.venv\Scripts\cloney.exe` unter Windows, `.venv/bin/cloney` sonst. Dieser Ordner
liegt nur dann im Suchpfad, wenn die Umgebung **aktiviert** ist — ein frisch
geöffnetes Fenster kennt den Befehl nicht.

In den globalen Suchpfad gehört er nicht: er braucht die Python-Umgebung samt
Torch und Modellen aus `.venv` und zeigt ins Leere, sobald die Umgebung neu
angelegt wird.

```powershell
# Umgebung aktivieren, dann kurz tippen -- gilt je Fenster
.\.venv\Scripts\Activate.ps1
cloney doctor

# Oder ohne Aktivierung, direkt
.\.venv\Scripts\cloney.exe doctor

# Oder als Modul, falls die ausführbare Datei zickt
.\.venv\Scripts\python.exe -m cloney.cli doctor
```

Scheitert `Activate.ps1` an der Ausführungsrichtlinie, hilft
`Set-ExecutionPolicy -Scope Process Bypass` im selben Fenster — oder der direkte
Aufruf.

Unter Linux und macOS entsprechend `source .venv/bin/activate` beziehungsweise
`./.venv/bin/cloney`.

### Prüfen, ob alles bereit ist

```bash
cloney doctor
```

Die Diagnose prüft durch Ausführen statt durch Nachschlagen von Versionsnummern:
Sie lädt tatsächlich eine WAV-Datei mit `torchaudio`, hält die Architekturliste
des PyTorch-Builds gegen die Architektur der eingebauten Karte, sieht im
Modell-Repo nach, fragt beim Higgs-Server an und fährt einen vollständigen
Durchstich mit der Dummy-Engine. Zu jedem Befund steht der Befehl, der ihn behebt.

```
[ OK ] PyTorch            2.7.0 (CUDA 12.8) auf NVIDIA GeForce RTX 5080, sm_120, 16 GB VRAM
[ OK ] Audio-Laden        torchaudio liest WAV (24000 Hz, 24000 Samples)
[WARN] Engine higgs       SGLang-Omni läuft nicht nativ unter Windows
                          -> Für Higgs v3 wird WSL2 gebraucht. Unter Windows ist f5-de die Engine der Wahl.
[ OK ] Durchstich         1 Chunks, 7.5s erzeugt, Fehlerrate 0%
```

### Einmal den ganzen Weg gehen

```bash
cloney demo --audio meine_stimme.wav
```

Rendert einen kurzen Satz voller Ziffern, Symbole und Abkürzungen — man soll
hören, dass die Normalisierung greift. Braucht die gewählte Engine einen
Referenztext, wird er unterwegs selbst ermittelt.

Die Oberfläche einzeln starten:

```bash
cloney web              # öffnet den Browser, sobald der Server antwortet
cloney web --no-open    # ohne Browser
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

# Verwalten
cloney projects list
cloney projects discard <kennung>   # Ton verwerfen, Seeds behalten
cloney projects remove <kennung>
cloney voices transcript <name> --text "Der Wortlaut der Aufnahme."
cloney voices remove <name>

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
| `f5-de` | ~2 GB | CC-BY-NC-4.0 | F5-TTS mit deutschem Finetune, läuft im eigenen Prozess, keine Inline-Tags |

`f5-de` ist auf Deutsch nachtrainiert, während Higgs v3 ein generalistisches
Multilingual-Modell ist, dessen deutsche Prosodie sein schwächster Teil ist. Wer
Deutsch-Qualität sucht, sollte beide gegeneinander hören — der Größenunterschied
sagt darüber wenig.

Zwei Eigenheiten von `f5-de` sind zu beachten:

- **Der Referenztext ist Pflicht.** F5-TTS leitet daraus die Sprechgeschwindigkeit
  ab. Cloney bricht deshalb schon beim Anlegen des Projekts ab statt mitten in
  der Synthese. Mit `--auto-transcript` wird er automatisch ermittelt.
- **Eine Generierung umfasst rund 22 Sekunden**, Referenz eingerechnet. Cloney
  schneidet die Chunks entsprechend kleiner — siehe unten.

Die Dateinamen der deutschen Finetunes sind uneinheitlich — mal flach, mal in
Unterordnern, mal `.pt` statt `.safetensors`, mit beliebiger Schrittzahl im Namen.
Cloney rät deshalb keinen Namen, sondern sieht im Repo nach und wählt: bevorzugt
`.safetensors`, den höchsten Stand, ohne BigVGAN (das zusätzliche Abhängigkeiten
mitbringt). Voreingestellt ist
[`aihpi/F5-TTS-German`](https://huggingface.co/aihpi/F5-TTS-German); Alternativen
sind [`hvoss-techfak/F5-TTS-German`](https://huggingface.co/hvoss-techfak/F5-TTS-German)
und [`marduk-ra/F5-TTS-German`](https://huggingface.co/marduk-ra/F5-TTS-German).
Mit `CLONEY_F5_CKPT_FILENAME` lässt sich die Auswahl überschreiben, mit
`CLONEY_F5_CKPT_PATH` direkt auf eine lokale Datei zeigen.

### Wenn es zu schnell oder unnatürlich klingt

F5-TTS berechnet die Dauer des Erzeugten aus der Referenz:

```python
duration = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len / speed
```

Maßgeblich ist also **wie viele Sekunden je Zeichen** die Referenz braucht. Daraus
folgen zwei Stellschrauben:

**Sprechtempo.** Werte unter 1 verlängern die Dauer, es wird langsamer gesprochen.
Klingt das Ergebnis gehetzt, ist 0,85 ein guter erster Versuch. In der Oberfläche
unter „Womit gerendert wird", über die Kommandozeile mit `-o speed=0.85`, dauerhaft
über `CLONEY_F5_SPEED`.

**Der Referenztext.** Ist er länger als das tatsächlich Gesprochene, sinkt der Wert
für Sekunden je Zeichen — und alles wird zu schnell. Die Projektseite zeigt deshalb
die gemessenen Zeichen pro Sekunde; deutsches Sprechtempo liegt bei etwa 14.
Deutlich darüber heißt: der Text passt nicht zur Aufnahme.

Weiter helfen können `nfe_step` (mehr Schritte klingen glatter, kosten Rechenzeit)
und `cfg_strength` (höher bindet enger an die Referenz, wirkt aber steifer).

Die Reglerstellung steht im Projekt-Manifest — ein Lauf bleibt damit auch
nachträglich reproduzierbar. Welche Regler es gibt, bestimmt die Engine selbst;
die Oberfläche zeigt, was da ist.

### Warum die Chunk-Länge von der Engine abhängt

F5-TTS teilt zu lange Eingaben selbst auf und blendet die Teile ineinander. Das
klingt zunächst brauchbar, aber ein so entstandener Chunk enthielte Nähte, die
sich nicht mehr einzeln nachbessern lassen — und genau das einzelne Nachbessern
ist der Zweck der Aufteilung. Cloney schneidet deshalb von vornherein so zu, dass
das Modell nie selbst teilen muss:

| Referenz | Budget je Chunk | Derselbe Text |
|---|---|---|
| 6 s | 16 s | weniger, längere Chunks |
| 9 s | 13 s | 6 Chunks |
| 12 s | 10 s | mehr, kürzere Chunks |

Jede Sekunde Referenz verkürzt also, was pro Chunk erzeugt werden kann. Sechs bis
neun Sekunden sind ein guter Kompromiss; darüber gewinnt die Klonqualität kaum
noch, aber die Chunks werden unnötig kleinteilig. Engines ohne harte Grenze —
`higgs`, `dummy` — nutzen unverändert den Wunschwert aus der Konfiguration.

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
| `CLONEY_F5_REPO_ID` | `aihpi/F5-TTS-German` | Modell für die Engine `f5-de` |
| `CLONEY_F5_CKPT_PATH` | — | lokaler Checkpoint, hat Vorrang vor dem Download |
| `CLONEY_F5_NFE_STEP` | `32` | Qualität gegen Rechenzeit; 16 ist spürbar schneller |

## Entwicklung

```bash
pytest          # läuft vollständig ohne GPU und ohne Netz
ruff check .
ruff format .
```

Entwickelt wird über Pull Requests gegen `main`. Die CI führt dieselben drei
Befehle aus; sie braucht weder GPU noch Modelldownload und läuft deshalb in
Sekunden.

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
(Fremdwörter, Eigennamen), der Emotions-Director, und ein Benchmark-Harness, das
`higgs` und `f5-de` auf einem deutschen Testsatz vergleicht statt sie nach Gefühl
auszuwählen.

Das Anfrageschema der Higgs-Engine ist gegen die öffentliche Dokumentation
gebaut, aber nicht gegen einen laufenden Server verifiziert; die Engine gibt den
Fehler der Gegenstelle unverändert weiter, statt ihn zu verschlucken. Der
Windows-Installer `install.ps1` konnte mangels PowerShell in der
Entwicklungsumgebung nicht ausgeführt werden — die gesamte Logik liegt deshalb in
`scripts/setup.py`, das geprüft ist; das Skript selbst beschränkt sich auf das
Anlegen der Umgebung.
