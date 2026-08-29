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

**Unter Windows braucht Higgs den Umweg über WSL2.** SGLang-Omni
[läuft nicht nativ unter Windows](https://docs.sglang.ai/get_started/install.html) —
es setzt Linux-spezifische CUDA-Kernel voraus. Mit WSL2 läuft es; siehe
[Higgs unter Windows](#higgs-unter-windows). Ohne WSL2 bleibt `f5-de`, das
ohnehin auf Deutsch nachtrainiert ist.

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
pip install -e ".[dev,asr,f5,similarity]"
cp .env.example .env
```

Die Extras sind einzeln wählbar: `asr` bringt faster-whisper für die
Qualitätskontrolle, `f5` das deutsche F5-TTS-Modell, `similarity` das
ECAPA-Modell für die Stimmähnlichkeit. Ohne sie läuft der Kern -- fehlt eines,
entfällt die zugehörige Messung, nicht der Lauf.

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
[ OK ] Engine higgs       Server erreichbar, Modell 'bosonai/higgs-audio-v3-tts-4b'
[ OK ] Higgs-Referenz     geht als Data-URL mit, kein Serverparameter nötig
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

# Reglerstellungen gegeneinander messen statt raten
cloney compare --text probe.txt --voice erzaehlstimme -g speed=0.8,1.0,1.2

# Weboberfläche mit dem Satz-Editor
cloney web
```

Für die Higgs-Engine muss der Modellserver laufen:

```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

### Higgs unter Windows: was in WSL zu tun ist

SGLang-Omni läuft nicht nativ unter Windows. Der Server läuft deshalb in WSL2,
Cloney bleibt auf Windows, und beide reden über `localhost:8000` -- WSL2 leitet
das durch.

**Vorweg, ehrlich:** SGLang-Omni führt Consumer-Karten als
[offene Baustelle](https://github.com/sgl-project/sglang-omni/issues/1120).
Validiert ist die RTX 4090 (SM89); für die RTX-50-Serie (SM120) gibt es
Einzelnachweise für andere Modelle, aber die Installations- und Backend-Matrix
ist ausdrücklich unvollständig, und Higgs v3 ist unter diesen Nachweisen nicht
darunter. Es kann also sein, dass der Server auf einer 5080 gar nicht erst
hochkommt. Das ist keine Frage von Cloney -- die Engine ist fertig und wartet.

#### 1. GPU in WSL

Der Windows-Treiber genügt; **in WSL darf kein NVIDIA-Treiber installiert
werden**, er würde den durchgereichten überschreiben. Nur das CUDA-Toolkit für
WSL, und zwar die Variante `WSL-Ubuntu` von der
[CUDA-Downloadseite](https://developer.nvidia.com/cuda-downloads).

```bash
nvidia-smi          # muss die Karte zeigen, sonst stimmt etwas am Windows-Treiber nicht
```

#### 2. SGLang-Omni

Empfohlen wird das Docker-Image, weil dort UCX, flash-attn und CUDA bereits
zusammenpassen -- genau die drei Dinge, an denen eine Handinstallation scheitert:

```bash
docker run -it --shm-size 32g --gpus all --ipc host --network host --privileged \
  hongccc/sglang-omni:dev /bin/zsh
```

Ohne Docker, in WSL direkt (Python 3.12, `torch==2.13.0`, `flash-attn-4`):

```bash
pip install --upgrade pip && pip install uv
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.3"
```

#### 3. Server starten

```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

Beim ersten Mal lädt das Modell (4B, rund 9 GB) aus dem Hugging-Face-Cache
herunter. Auf 16 GB ist das eng: das Modell belegt in bf16 etwa 11 GB, der Rest
geht für den KV-Cache drauf.

#### 4. Auf der Windows-Seite

```bash
cloney doctor
```

`doctor` fragt `/v1/models` ab und nennt den Namen, unter dem der Server das
Modell führt -- weicht er ab, steht der passende `CLONEY_HIGGS_MODEL=`-Eintrag
gleich in der Meldung.

#### Warum die Referenzaufnahme als Data-URL geht

`audio_path` nimmt laut Kochbuch "local path, file URL, data URL, or HTTP URL"
entgegen. Ein **Dateipfad** hat über die Grenze zwischen Windows und WSL zwei
Haken: `C:\Users\...` ist für den Server kein gültiger Pfad (dieselbe Datei
liegt dort unter `/mnt/c/Users/...`), und er darf ihn nur lesen, wenn der Server
mit `--allowed-local-media-path <ordner>` gestartet wurde.

Deshalb ist `base64` die Voreinstellung: die Aufnahme geht als Data-URL im
selben Feld mit. Das kostet ein paar Megabyte je Anfrage über Loopback und
erspart dafür beides. Wer den Pfadweg will, setzt
`CLONEY_HIGGS_REFERENCE_MODE=auto` und startet den Server mit dem Stimmenordner:

```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000 \
  --allowed-local-media-path /mnt/c/Pfad/zu/Cloney/data/voices
```

`cloney doctor` schreibt diese Zeile mit dem richtigen Pfad hin, wenn der
Pfadweg eingestellt ist.

#### Higgs kennt keinen Seed

Die Schnittstelle nimmt kein solches Feld entgegen, jeder Aufruf würfelt neu.
„Neu würfeln" liefert also weiterhin eine andere Fassung, aber ein früheres
Ergebnis lässt sich nicht wiederherstellen, und „Ton verwerfen" zeigt nicht die
Wirkung eines Reglers allein. Im Vergleichslauf heißt das: die Varianten
unterscheiden sich zusätzlich im Zufall. Cloney führt das als
`EngineInfo.reproducible_seed` und schreibt den Hinweis an beide Stellen, statt
eine Reproduzierbarkeit zu versprechen, die es hier nicht gibt.

## Die Projektseite

Oben der Kopf mit den Kennzahlen, darunter eine Statusleiste, die beim Scrollen
stehen bleibt -- sie ist das, was man während eines Laufs beobachtet. Darunter
drei Reiter:

**Sätze** ist die tragende Ansicht: eine Zeile pro Satz mit Status, Fehlerrate,
Stimmähnlichkeit, der Rückschrift der Spracherkennung und einem eigenen
Abspieler. Von dort lässt sich ein einzelner Satz neu würfeln oder umformulieren,
ohne das ganze Kapitel neu zu rendern. Genau an diesem Arbeitsschritt scheitert
eine Langform-Produktion sonst.

Sichtbar sind pro Satz drei Fassungen: der Rohtext, die daraus erzeugte
Sprechfassung und das, was die Spracherkennung tatsächlich gehört hat. Weichen
die letzten beiden voneinander ab, ist das der konkrete Hinweis, wo es hakt.

**Einstellungen** trägt alles, was den Klang bestimmt -- und dieselben Angaben
wie beim Anlegen: Text, Stimme, Engine, dazu die Regler und die Kennwerte der
Referenzaufnahme. Ein bestehendes Projekt lässt sich damit ändern, statt es neu
anlegen zu müssen.

**Projekt** sammelt Umbenennen, Kopie, Ton verwerfen und Löschen an einer
Stelle, weg von den Dingen, die man ständig braucht.

Die Reiter sind reines CSS über versteckte Radioknöpfe. Ohne Skript heißt: der
gewählte Reiter bleibt stehen, auch wenn htmx die Statusleiste darüber
austauscht.

### Text, Stimme oder Engine nachträglich ändern

Ein anderer Text heißt neu segmentieren, und damit wandern die Chunk-Grenzen.
Trotzdem soll ein Tippfehler in Satz drei nicht die Arbeit an Satz siebzehn
kosten: Sätze, deren **Sprechfassung** wörtlich gleich bleibt, behalten Ton,
Seed und Messwerte. Verglichen wird die normalisierte Fassung, nicht der
Rohtext -- wer nur `3.` zu `dritten` ändert, hört dasselbe und soll nicht neu
rendern müssen.

Ein Wechsel von Stimme oder Engine verwirft dagegen allen Ton. Vorhandene Sätze
stammten dann von einem anderen Sprecher oder Modell, und sie stehen zu lassen
ergäbe eine Spur aus zwei Stimmen.

Die fertige Spur bleibt nur, wenn sich am Satzbestand nichts geändert hat --
sonst wäre sie eine Lüge. Dass sie ein unverändertes Übernehmen übersteht, macht
das Formular gefahrlos wiederholbar.

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
die gemessenen Zeichen pro Sekunde, berechnet auf dem Sprachanteil statt der
Dateilänge: F5-TTS schneidet lange Stille selbst heraus, bevor es das Tempo ableitet.

Übliches deutsches Sprechtempo liegt bei **12 bis 18 Zeichen pro Sekunde** —
hergeleitet aus 120 bis 150 Wörtern je Minute und rund sechs Zeichen je Wort samt
Leerzeichen. Siebzehn ist zügiges Podcast-Tempo, also normal. Erst weit außerhalb
(unter 7 oder über 24) passt der Text vermutlich nicht zur Aufnahme.

**Wichtig dabei:** F5-TTS *übernimmt* die Geschwindigkeit der Referenz. Spricht das
Vorbild zügig, spricht der Klon zügig — das ist kein Fehler, sondern der Zweck. Wer
ruhiger vorgelesen haben will als sein Vorbild spricht, stellt das über das
Sprechtempo ein und nicht über den Referenztext, der schlicht stimmen muss. Cloney
rechnet den passenden Wert aus und bietet ihn an: bei 17 Zeichen/s sind das 0,85.

Weiter helfen können `nfe_step` (mehr Schritte klingen glatter, kosten Rechenzeit)
und `cfg_strength` (höher bindet enger an die Referenz, wirkt aber steifer).

Die Reglerstellung steht im Projekt-Manifest — ein Lauf bleibt damit auch
nachträglich reproduzierbar. Welche Regler es gibt, bestimmt die Engine selbst;
die Oberfläche zeigt, was da ist.

### Zwei Kennzahlen, nicht eine

Die **Fehlerrate** (CER) vergleicht die Rückschrift der Spracherkennung mit der
Sprechfassung. Sie prüft damit, ob die richtigen Wörter herauskommen — über die
Stimme sagt sie nichts. Ein Satz kann fehlerfrei sein und nach jemand anderem
klingen.

Diese Lücke schließt die **Stimmähnlichkeit**: die Einbettungen von Referenz und
Ergebnis werden verglichen (ECAPA-TDNN, rund 20 MB, eigene Phase nach der
Spracherkennung). 1,00 heißt identisch.

```
 #     CER  Stimme  Text
 0   0.000    0.91  Am dritten Mai zweitausendvierundzwanzig begann alles.
 1   0.041    0.62  Doktor Meier sagte zum Beispiel nichts dazu.
```

Satz 1 wäre nach der Fehlerrate allein unauffällig — die Ähnlichkeit zeigt, dass
er aus der Rolle fällt.

Fehlt `speechbrain`, wird der Vergleich übersprungen und der Grund im
Manifest vermerkt; gerendert und zusammengebaut wird trotzdem. Eine freiwillige
Zusatzzahl darf kein fertiges Kapitel kosten.

**Markiert wird erst mit gesetzter Schwelle.** `CLONEY_SIMILARITY_THRESHOLD` steht
auf 0, es wird also nur gemessen und angezeigt. Welchen Wert ein guter Klon
erreicht, hängt an Modell und Aufnahme; eine ungeprüfte Vorgabe erzeugte
Fehlalarme und lenkte vom Wesentlichen ab. Nach ein paar Läufen den niedrigsten
Wert unter den guten Sätzen ablesen und die Schwelle knapp darunter setzen.

Installation: `pip install -e ".[similarity]"` -- der Installer bringt sie mit.

### Der Vergleichslauf: aus Raten wird Messen

Welche Reglerstellung zu einer Stimme passt, steht in keiner Dokumentation. Ohne
Hilfsmittel heißt das: einen Wert raten, ein Kapitel rendern, hören, wieder raten.
Ein Vergleichslauf rendert stattdessen dieselbe kurze Textprobe einmal je
Einstellung und stellt die Ergebnisse nebeneinander.

```bash
cloney compare --text probe.txt --voice erzaehlstimme \
  -g speed=0.8,1.0,1.2 -g nfe_step=16,32
```

Die Ausgabe (die Zahlen hier sind ein Beispiel, keine Messung):

```
Variante                          CER   Stimme    Dauer        Tempo
--------------------------------------------------------------------
Sprechtempo 0.8 · Schritte 16   4.1%     0.71    12.4s  9.8 Zeichen/s
Sprechtempo 0.8 · Schritte 32   1.9%     0.88    12.3s  9.9 Zeichen/s
Sprechtempo 1 · Schritte 16     3.8%     0.74     9.9s 12.3 Zeichen/s
Sprechtempo 1 · Schritte 32     1.2%*    0.89*    9.8s 12.4 Zeichen/s
Sprechtempo 1.2 · Schritte 16   6.4%     0.66     8.2s 14.9 Zeichen/s
Sprechtempo 1.2 · Schritte 32   2.0%     0.85     8.1s 15.1 Zeichen/s
```

In der Weboberfläche steht dieselbe Tabelle unter **Vergleiche**, mit einem
Abspieler je Zeile — die Zahlen engen die Auswahl ein, entschieden wird am Ohr.

Drei Entscheidungen machen das zu einer Messung statt zu einer Sammlung von
Eindrücken:

**Alle Varianten teilen sich dieselben Seeds.** Sie werden aus der Kennung des
Vergleichs abgeleitet, nicht aus der des Projekts. Sonst unterschieden sich zwei
Zeilen in zwei Dingen zugleich — Regler und Zufall — und die Tabelle beantwortete
nicht mehr die gestellte Frage. Aus demselben Grund läuft ein Vergleich ohne
Wiederholungsversuche: ein neuer Seed nach einem auffälligen Satz würde genau das
verwischen, was gemessen werden soll.

**Jede Variante ist ein vollwertiges Projekt** und durchläuft dieselbe Pipeline
wie ein Hörbuch. Gemessen wird damit, was später auch tatsächlich passiert; es
gibt keinen zweiten, abweichenden Renderweg, der auseinanderlaufen könnte. Die
Variantenprojekte liegen im Ordner des Vergleichs und fluten die Projektliste
nicht.

**Markiert wird je Spalte, nicht als Gesamtnote.** Wie eine halbe Prozent
Fehlerrate gegen zwei Hundertstel Ähnlichkeit aufzuwiegen wäre, kann niemand
belegen — eine gewichtete Punktzahl täuschte eine Objektivität vor, die es nicht
gibt. Bei Gleichstand wird niemand markiert.

Die Probe kurz halten: sie wird je Variante einmal vollständig gerendert. Ein
Kreuzprodukt ist auf zwölf Varianten gedeckelt, damit aus drei gut gemeinten
Achsen kein Lauf über Stunden wird.

### Was mit der Tonqualität passiert -- und wo die Grenze liegt

**Die Referenzaufnahme bleibt unangetastet.** Sie wird gelesen, um sie zu prüfen,
und ansonsten Byte für Byte abgelegt: Samplerate, Kanäle und Auflösung wie in der
Quelle, samt Dateiendung. Was in der Oberfläche abgespielt wird, ist die eigene
Datei. Ältere Fassungen von Cloney haben sie nach Mono gemischt und auf 16 Bit
gebracht -- die Referenz im Speicher klang danach schlechter als das Original,
ohne dass irgendwer davon etwas gehabt hätte: F5-TTS mischt und tastet ohnehin
selbst um, und zwar besser aus dem Original als aus einer bereits verkleinerten
Fassung.

**Die erzeugte Spur kann die Aufnahme trotzdem nicht erreichen.** F5-TTS ist ein
24-kHz-Modell: der Vocoder liefert 24000 Samples je Sekunde in Mono, und mehr
gibt es nicht herzustellen. Aus einer 48-kHz-Aufnahme wird eine Spur mit der
halben Bandbreite -- alles über 12 kHz fehlt. Das ist keine Einstellung, die
sich hochdrehen ließe, sondern die Auflösung, in der das Modell rechnet. Der
einzige Weg dorthin wäre ein Modell mit höherer Ausgaberate; die Referenz liegt
dafür im Original bereit.

Innerhalb dieser Grenze wird nichts verschenkt:

- Chunks und fertige Spur werden mit **24 Bit** geschrieben. Ein Chunk wird
  geschrieben, für den Zusammenbau wieder gelesen, angeglichen und erneut
  geschrieben; bei 16 Bit quantisiert das zweimal. Hörbar ist das kaum,
  vermeidbar aber umsonst.
- Die Lautheitsangleichung nimmt einen Satz lieber **ein Dezibel zurück, als
  seine Spitzen abzuschneiden**. Sprache hat einen hohen Scheitelfaktor; die
  Anhebung auf -16 LUFS kann die lautesten Stellen über die Vollaussteuerung
  heben. Vorher wurde dort hart abgeschnitten -- hörbare Verzerrung. Jetzt gilt
  eine Grenze von -1 dBFS, und ein etwas zu leiser Satz ist das kleinere Übel.

### Wenn am Anfang ein Stück der Referenz zu hören ist

F5-TTS erzeugt Referenz und neuen Text in einem Stück und trennt sie danach an
einer berechneten Stelle:

```python
generated = generated[:, ref_audio_len:, :]
```

An den Referenztext hängt es dabei stets ein Satzende samt Pause an, gibt der
Aufnahme aber nur 50 ms Stille. Endet die Aufnahme abrupt statt auszuklingen,
passen Text und Klang am Ende nicht zusammen — das Modell dehnt den Referenzteil,
und ein Rest landet hinter der Schnittstelle.

Cloney geht das von zwei Seiten an. Beim Anlegen einer Stimme wird gewarnt, wenn
die Aufnahme mitten im Klang abbricht; nach dem letzten Wort einen Moment
weiterlaufen zu lassen behebt die Ursache.

Bleibt trotzdem etwas stehen, schneidet die Qualitätskontrolle es weg. Nach
Lautstärke ginge das nicht — der Vorspann ist Sprache. Über die Rückschrift
schon: sie sagt, ab welchem Wort der gewünschte Text beginnt, und die Wortzeiten
sagen, wann. Geschnitten wird nur, was sich sicher zuordnen lässt; im Zweifel
bleibt der Ton unangetastet. Was entfernt wurde, steht im Manifest und in der
Satzliste. Abschaltbar über `CLONEY_TRIM_REFERENCE_BLEED=false`.

Dafür muss die Qualitätskontrolle laufen — ohne `faster-whisper` gibt es keine
Rückschrift und damit keine Erkennung.

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

## Finetuning: der andere Mechanismus

Zero-Shot-Klonen *lernt* keine Stimme. Bei F5-TTS ist die Referenz ein Präfix,
das das Modell fortsetzt -- und dabei gilt aus dem Quelltext:

```python
if len(aseg) > 12000:
    aseg = aseg[:12000]  # alles über 12 s wird abgeschnitten
max_chars = int(len(ref_text.encode("utf-8")) / ref_sekunden * (22 - ref_sekunden) * speed)
```

Mehr Referenzmaterial hilft also nicht, es schadet: Überlänge wird gekappt, und
jede Sekunde Referenz verkleinert zugleich, was pro Durchgang erzeugt werden
kann. Wer Vielfalt über viele Sätze will, braucht den anderen Weg -- ein
Finetune, das die Gewichte selbst auf die Stimme zieht.

Der Weg dorthin in vier Schritten. **Die Schritte 1 bis 3 sind gebaut**, die
Verwaltung in der Oberfläche steht noch aus:

### 1. Datensatz — aus Aufnahmen wird Trainingsmaterial

```bash
cloney dataset build --audio lesungen/ --name anna
cloney dataset show anna
```

Lange Aufnahmen werden in Segmente von 3 bis 15 Sekunden zerlegt, mit Whisper
transkribiert und geprüft. Heraus kommt das Format, das F5-TTS erwartet: ein
`wavs`-Ordner und eine `metadata.csv` aus `pfad|text`.

Drei Entscheidungen tragen das:

**Geschnitten wird an Pausen, nie mitten im Klang.** Ein Segment, das mitten im
Wort beginnt, bringt dem Modell einen Anfang bei, den es nachher produziert. Ist
ein Bereich zu lang, wird er an seiner längsten inneren Pause geteilt -- dafür
genügt ein Atemzug, denn die Alternative wäre, zwanzig Sekunden brauchbare
Sprache wegzuwerfen. Findet sich gar keine, wird verworfen statt hart geschnitten.

**Was still ist, bestimmt die Aufnahme selbst.** Eine feste Schwelle wie
-40 dBFS setzt ein leises Zimmer voraus. Liegt der Raumton darüber -- bei einer
normalisierten Aufnahme keine Seltenheit --, ist plötzlich *nichts* mehr still,
und die ganze Lesung gilt als ein einziger Bereich ohne Schnittstelle. Cloney
misst deshalb den Grundpegel als leiseste *anhaltende* Stelle und setzt die
Schwelle zehn Dezibel darüber. An synthetischen Aufnahmen mit Raumton von -70
bis -30 dBFS findet das durchgehend genau die echten Pausen.

Zwei Anker, und es gewinnt der höhere: ein Mindestabstand **über dem
Grundpegel**, damit Rauschen nicht als Sprache zählt, und ein Mindestabstand
**unter dem Sprechpegel**, damit eine Pause auch dann erkannt wird, wenn in ihr
geatmet wird. Der zweite ist aus einer echten Lesung entstanden: Grundpegel
-63 dBFS, Sprechpegel -12, die Sprechpausen aber erst ab -50 aufwärts. Allein
am Grundpegel gemessen lag die Schwelle bei -53 und fand genau eine Pause auf
42 Sekunden.

Dazu zwei Fallen, in die beide Richtungen führen:

- **Exakt stille Stellen zählen nicht mit.** Eine halbe Sekunde harte Null am
  Dateianfang -- was jedes Schnittprogramm hinterlässt -- ergäbe einen
  Grundpegel von -240 dBFS und eine Schwelle von -230. Damit gälte jedes
  Rauschen als Sprache, und die Lesung wäre wieder ein Block ohne Pause. Solche
  Frames stammen vom Programm, nicht aus dem Zimmer, und bleiben deshalb bei der
  Schätzung außen vor.
- **Die Schwelle bleibt in einem sinnvollen Band** (-55 bis -20 dBFS), damit
  weder eine sehr leise noch eine durchweg laute Aufnahme sie ins Absurde zieht.

Was gemessen wurde, steht in der Ausgabe -- „Raumton -34 dBFS" oder
„geschnittene Stille". Das sagt mehr über die Aufnahme als jede Schwelle, die
wir vorgeben könnten.

**Der Text durchläuft dieselbe Normalisierung wie bei der Synthese.** Die
Spracherkennung schreibt „3. Mai", gesprochen wurde „dritten Mai". Trainiert
werden muss auf der Form, die später auch hineingeht -- sonst lernt das Modell,
Ziffern anders auszusprechen, als Cloney sie ihm vorlegt. Der Wortlaut der
Erkennung bleibt daneben im Manifest stehen.

**Was durchfällt, wird benannt.** Zu kurz, übersteuert, zu leise, keine
Rückschrift, oder ein Text, der nicht zur Länge passt (fast immer eine
Fehltranskription). Jeder verworfene Abschnitt steht mit Sekunde und Grund im
Manifest:

```
lesung.wav: 4 Abschnitte, 1 verworfen, Raumton -34 dBFS

Datensatz 'anna' in data/datasets/anna
  4 Segmente, 0.5 Minuten, 44100 Hz
  Median: 6.6s je Segment, 14.9 Zeichen/s
  Verworfen: 1 Abschnitte (0.4 Minuten)
       1x 22.8s ohne Pause zum Schneiden
```

Das Sprechtempo meint dabei den **Wortlaut**, nicht die Sprechfassung: die
Normalisierung bläht den Text auf -- aus „3. Mai 2024" werden 36 Zeichen --, und
auf ihr gerechnet sähe jede Aufnahme mit Ziffern zu schnell aus.

Ein Datensatz, der stillschweigend die Hälfte wegwirft, ist sonst nicht von
einem zu unterscheiden, bei dem die Aufnahme schlecht war.

### Wenn nichts durchkommt: nachsehen statt raten

Fällt eine Lesung durch, sind zwei Ursachen möglich -- eine Schwelle, die nicht
zur Aufnahme passt, oder eine Leseweise ohne Pausen. Von außen sehen beide
gleich aus:

```bash
cloney dataset probe --audio lesungen/normal-talk.wav
```

```
  38.8s bei 44100 Hz
  Grundpegel      -63 dBFS   (leiseste anhaltende Stelle)
  Sprechpegel     -23 dBFS   (95. Perzentil)
  Exakte Stille   0.0% der Aufnahme

  Schwelle   Pausen ab 180ms   ab 320ms   längste Stille   still
      -53               1          1          0.80s      5%  <- verwendet
      -45               1          1          0.80s      5%
      -35               1          1          0.80s      6%
      -25             107          1          0.87s     80%  <- über dem Sprechpegel

  Nur 1 Pause(n) auf 39s. Für Segmente von höchstens 15s bräuchte es mindestens 2.
  Die Schwelle ist nicht das Problem -- es wird zu lang am Stück gesprochen.
```

Die Tabelle spielt durch, was jede Schwelle fände. Bleibt die Zahl über alle
Schwellen gleich, liegt es nicht an der Einstellung. Gezählt werden dabei nur
Pausen **zwischen** der Sprache: Vorlauf und Ausklang sind fast immer still,
taugen aber nicht als Schnittstelle. Und eine Schwelle über dem Sprechpegel ist
markiert -- dort gilt die ganze Aufnahme als still, und die Zahl der „Pausen"
sagt nichts mehr aus.

Für Material, das schon aufgenommen ist und keine Pausen hat:

```bash
cloney dataset build --audio lesungen/ --name anna --force-split
```

Das trennt zu lange Bereiche an ihrer leisesten Stelle, auch ohne echte Pause --
kein guter Schnitt, aber besser, als den ganzen Bereich zu verlieren. Bewusst
nicht der Normalfall.

### 2. Vorbereiten

```bash
cloney finetune prepare anna
```

Ruft `prepare_csv_wavs.py` aus F5-TTS auf, das aus dem Datensatz `raw.arrow`,
`duration.json` und `vocab.txt` erzeugt. Cloney trainiert nicht selbst und baut
auch das Datenformat nicht nach -- das hieße, es bei jeder Änderung nachzuziehen.
Was Cloney beisteuert, ist alles davor, und genau dort liegen die Fallen:

**Das Eingabeformat weicht ab.** `prepare_csv_wavs.py` will eine CSV mit der
Kopfzeile `audio_file|text` und **absoluten** Pfaden und prüft ausdrücklich
darauf. Cloneys eigene `metadata.csv` führt relative Pfade ohne Kopfzeile, weil
das für alles andere handlicher ist. Übersetzt wird beim Vorbereiten.

Übergeben wird dabei die **Datei**, nicht ihr Ordner. Der Parameter heißt dort
`inp_dir`, das Skript prüft aber auf die Endung und liest die Tonpfade
unverändert aus der Tabelle -- dem Namen zu folgen endet in
`ValueError: input must be a .csv file`.

**Das Vokabular muss vom Pretrain stammen.** F5 kopiert im Finetune-Zweig sein
eigenes, fest eingetragenes Vokabular -- das des englisch-chinesischen
Basismodells. Beim Weitertrainieren eines *deutschen* Modells passt das nicht zu
den geladenen Gewichten: `text_num_embeds` ist die Vokabulargröße, und eine
andere Größe heißt eine andere Embedding-Matrix.

Schlimmer noch: derselbe Zweig prüft die Datei mit einem `assert`, und sie liegt
unter `<f5>/data/Emilia_ZH_EN_pinyin/vocab.txt` -- ein Pfad in die
Trainingsdaten, die bei einer Installation über pip nicht mitkommen:

```
AssertionError: pretrained vocab.txt not found:
  ...\site-packages\f5_tts\..\..\data\Emilia_ZH_EN_pinyin\vocab.txt
```

Cloney gibt deshalb `--pretrain` mit. Der Name führt in die Irre: das Flag
steuert im Skript ausschließlich, ob das Vokabular aus den eigenen Texten
erzeugt oder das fest eingetragene kopiert wird -- alles andere ist in beiden
Zweigen identisch. Anschließend wird die Datei durch die des deutschen Pretrains
ersetzt, das einzige Vokabular, das zu den Gewichten passt.

**Die Ordner liegen bei F5, nicht bei Cloney.** Der Datenlader sucht unter
`<f5>/data/<name>_<tokenizer>`, die Checkpoints landen unter `<f5>/ckpts/<name>`,
beides abgeleitet aus dem Ort des installierten Pakets. Wer F5 als Auscheckstand
statt über pip hat, gibt `--f5-dir` an.

### 3. Training

```bash
cloney finetune train anna              # startet den Lauf
cloney finetune train anna --dry-run    # zeigt nur, was liefe
```

```
Datensatz:   anna, 62.0 Minuten
Daten:       .../data/anna_custom
Checkpoints: .../ckpts/anna
Pretrain:    model_last.safetensors

1600 Frames je Schritt sind 17.1s Ton; eine Epoche braucht rund 218 Schritte,
100 Epochen also etwa 21800 Schritte.
```

Die Umrechnung ist der Punkt: `batch_size_per_gpu` zählt **Frames**, keine
Beispiele, und 1600 Frames sind bei 24 kHz und Hop 256 rund 17 Sekunden Ton je
Schritt. Erst damit lässt sich abschätzen, wie lange ein Lauf dauert -- und
sofort sehen, ob der Datensatz überhaupt trägt.

Zwei Voreinstellungen weichen bewusst von F5 ab, und beide **richten sich nach
der Länge des Laufs** statt fest zu sein:

- **Aufwärmen: höchstens 200 Schritte statt 20000, und nie mehr als ein Zehntel
  des Laufs.** F5s Standard ist für einen Lauf von Grund auf gedacht. Aber auch
  ein fester kleiner Wert geht schief: 0.6 Minuten Material ergeben rund 200
  Schritte, und mit 200 Aufwärmschritten wäre die Lernrate genau dann oben, wenn
  das Training endet -- der Lauf hätte nie bei der Ziellernrate trainiert.
- **Sichern: alle 1000 Schritte statt alle 50000, bei kurzen Läufen dichter.**
  Am Ende sichert F5 ohnehin einmal; interessant sind aber die Zwischenstände,
  denn an ihnen zeigt sich, ob längeres Training noch etwas bringt.

```
  0.6 min ->    200 Schritte, Aufwärmen   20, sichern alle   50
 60.0 min ->  21100 Schritte, Aufwärmen  200, sichern alle 1000
```

Was **nicht** gemessen ist: welche `batch_size_per_gpu` auf 16 GB durchläuft.
Der Vorschlag von 1600 ist die Hälfte von F5s Standard, der auf einer 24-GB-Karte
erprobt ist. Bei einem Speicherfehler `--batch-frames` halbieren.

Bei wenig Material sagt der Befehl es vorher:

```
Nur 0.7 Minuten Material. F5s eigene Angabe für diesen Fall lautet 10 bis 100 Stunden,
die dokumentierten Erfolge einzelner Sprecher liegen bei zwölf Stunden und darüber.
Für weniger gibt es keinen belegten Fall.
```

**Der Pretrain muss die EMA-Struktur tragen.** `Trainer.load_checkpoint` ruft
`self.ema_model.load_state_dict(...)` *bevor* der Zweig greift, der einen reinen
Inferenz-Export behandelt:

```python
if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
    checkpoint = {"ema_model_state_dict": load_file(...)}   # nackte Schlüssel
...
self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])   # scheitert hier
```

Der Wrapper erwartet `initted`, `step` und `ema_model.<...>`. Der deutsche
Finetune bringt einen Export mit nackten `transformer.<...>`-Schlüsseln mit --
das endet in einer seitenlangen Liste fehlender Schlüssel. Cloney legt deshalb
vor dem Training eine Fassung an, die diese Struktur trägt, und übergibt sie als
`--pretrain`. Trägt ein Checkpoint sie bereits, bleibt er unangetastet: ihn zu
kopieren kostete über ein Gigabyte für nichts.

### Das Ergebnis benutzen

Ein Trainingslauf hinterlässt `model_last.pt` und `model_<schritt>.pt` in F5s
Ordnern. Damit ist das Modell noch nicht *verwendbar*: die Engine liest ihren
Checkpoint aus der Konfiguration, und dort ist Platz für genau einen. Trainierte
Stände bekommen deshalb einen Namen:

```bash
cloney models list
cloney models add --name anna-4000 --ckpt <pfad>/model_4000.pt --note "Zwischenstand"
```

Den jüngsten Stand trägt `finetune train` nach einem erfolgreichen Lauf selbst
ein -- ein Checkpoint, der nirgends steht, lässt sich nicht anhören. Das
Vokabular gehört dazu: ein Finetune ist auf dem Vokabular seines Pretrains
trainiert, und mit einem anderen passt die Embedding-Matrix nicht zu den
Gewichten.

Damit wird gerendert und verglichen:

```bash
cloney render --text kapitel.txt --voice anna --model anna-ft

# Pretrain gegen Finetune, gleiche Probe, gleiche Seeds
cloney compare --text probe.txt --voice anna -m "" -m anna-ft -g speed=1.0
```

Ein leerer Modellname steht für den Pretrain. Mit mehreren Ständen wird das
Modell zu einer Achse des Vergleichs, und die Tabelle beantwortet die Frage, die
ein Finetune aufwirft: **ist er besser als das, was vorher da war?** -- an
Fehlerrate und Stimmähnlichkeit, statt am Eindruck.

### 4. In der Weboberfläche (geplant)### 4. In der Weboberfläche (geplant)

Datensätze ansehen, ein Training starten, den Verlauf verfolgen, Checkpoints
gegeneinander hören. Der Vergleichslauf ist dafür schon da: ein Checkpoint ist
nichts anderes als eine weitere Variante, die dieselbe Probe rendert.

### Was offen ist

**Wie viel Material nötig ist, weiß ich nicht belegt.** Die Angaben in der
Community reichen von Minuten bis Stunden, eine belastbare Zahl für Deutsch habe
ich nicht gefunden. Der Datensatzschritt liefert deshalb zuerst die Kennzahlen --
Gesamtdauer, Segmentlängen, Sprechtempo --, damit die Entscheidung auf Zahlen
steht statt auf einer Faustregel.

**Ob ein Finetune auf 16 GB durchläuft, ist ebenfalls unbelegt.** Hier gibt es
keine GPU, um es zu prüfen.

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
| `CLONEY_HIGGS_MODEL` | `bosonai/higgs-audio-v3-tts-4b` | muss dem `--model-path` des Servers entsprechen |
| `CLONEY_HIGGS_REFERENCE_MODE` | `base64` | `base64`/`auto`/`wsl`/`path` — wie der Server an die Referenz kommt |
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
Wiederaufnahme, Phasenmodell, Qualitätskontrolle mit Retry-Schleife,
Stimmähnlichkeit, Vergleichslauf, Zusammenbau mit Lautheitsangleichung,
Datensatzbau und Trainingsstart fürs Finetuning, CLI und Weboberfläche mit
Satz-Editor.

Als Nächstes: die Verwaltung des Finetunings in der Oberfläche. Danach
LLM-gestützte Textvorbereitung für das,
was Regeln nicht können (Fremdwörter, Eigennamen), und der Emotions-Director.
Der Vergleichslauf vergleicht bislang Reglerstellungen einer Engine; ihn über
mehrere Engines und über Checkpoints laufen zu lassen, ist der nächste Schritt
zum Benchmark-Harness.

Das Anfrageschema der Higgs-Engine folgt dem Kochbuch von SGLang-Omni und ist
gegen einen nachgebildeten Server geprüft (`tests/test_higgs.py`), aber nicht
gegen einen echten Higgs-Server gelaufen. Die Engine gibt den Fehler der
Gegenstelle unverändert weiter, statt ihn zu verschlucken. Der
Windows-Installer `install.ps1` konnte mangels PowerShell in der
Entwicklungsumgebung nicht ausgeführt werden — die gesamte Logik liegt deshalb in
`scripts/setup.py`, das geprüft ist; das Skript selbst beschränkt sich auf das
Anlegen der Umgebung.
