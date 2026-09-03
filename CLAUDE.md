# Cloney

Lokales Voice Cloning für deutsche Langform-Texte. Siehe README.md für Aufbau,
Installation und Bedienung.

## Arbeitsweise

- Entwickelt wird auf einem Branch, gemergt wird über einen Pull Request gegen `main`.
- **Ist die CI grün, wird der PR sofort gemergt** -- ohne Rückfrage.
- **Jeder neu geöffnete PR wird gleich beim Öffnen beobachtet**
  (`subscribe_pr_activity`). Dann kommt das Ergebnis der Prüfung von selbst an,
  statt von einer Nachfrage abzuhängen -- und eine rote CI fällt auf, auch wenn
  gerade niemand hinsieht. Gemergt wird danach von Hand.

  GitHubs Auto-Merge ist bewusst nicht im Einsatz. Er setzte zweierlei voraus:
  das Häkchen in den Repo-Einstellungen **und** einen Branch-Schutz auf `main`
  mit `test` als verlangter Prüfung -- ohne den gilt jeder PR sofort als
  mergebar, und GitHub lehnt Auto-Merge dafür ab. Beides kann nur der Besitzer
  des Repos setzen. Nicht bei jedem PR aufs Neue versuchen.
- Vor jedem Push laufen `pytest`, `ruff check .` und `ruff format --check .`.
  Die Testsuite kommt ohne GPU, ohne Modelldownload und ohne Netz aus.

## Worauf beim Ändern zu achten ist

Zwei Entwurfsentscheidungen tragen das Projekt. Wer sie aufweicht, nimmt Cloney
seinen Zweck:

1. **Normalisierung passiert vor dem Modell.** Ziffern, Symbole und Abkürzungen
   werden regelbasiert ausgeschrieben (`cloney/core/normalize.py`). Neue Regeln
   gehören mit einem Tabellentest in `tests/test_normalize.py` versehen.
2. **Jeder Chunk wird gegen eine unveränderte Referenzaufnahme konditioniert**,
   nie gegen das Ergebnis des Vorgängers. Daraus folgen Reproduzierbarkeit,
   Resume und das einzelne Neuwürfeln eines Satzes.

   Welche Aufnahme das ist, entscheidet die **Emotionslage** des Satzes. Sie
   steht wie der Seed im Manifest -- beide zusammen machen ihn reproduzierbar.
   Eine Lage zu wechseln verwirft den Ton und behält den Seed: nur so ist zu
   hören, was die Lage bewirkt, statt zugleich den Zufall zu bewegen. Nicht
   gewählt heißt neutral, und das ist die Hauptaufnahme der Stimme.

   Der Vergleichslauf (`cloney/core/compare.py`) zieht daraus die Konsequenz:
   alle Varianten teilen sich die aus der Vergleichskennung abgeleiteten Seeds
   und laufen ohne Wiederholungsversuche. Wer daran rührt, misst wieder Regler
   und Zufall zugleich -- und die Tabelle beantwortet nicht mehr die gestellte
   Frage.

   Seine Achsen sind Regler, Modell und Lage; die Lage gehört dabei zur
   Variante und nicht zum einzelnen Satz. Ein Zuschnitt unter zwei Varianten
   wird abgelehnt (`pruefe_raster`), und beim Ändern behält nur eine Zeile ihr
   Ergebnis, deren Regler, Modell und Lage gleich bleiben -- ein Wechsel von
   Probe, Stimme oder Engine verwirft alles.

Dazu zwei Regeln, die aus der VRAM-Beschränkung folgen:

- Es liegt **nie mehr als ein großes Modell gleichzeitig** im Speicher. Modelle
  werden ausschließlich über `cloney/vram.py` (`model_slot`) erzeugt und
  freigegeben.
- Hat eine Engine eine Grenze je Generierung, gehört sie als Daten in
  `EngineInfo` -- nicht als Sonderfall in die Pipeline.

## Neue Engine hinzufügen

`cloney/engines/base.py` beschreibt die Schnittstelle. Die Unterschiede zwischen
Engines sind Daten in `EngineInfo`: Lizenz der Gewichte, VRAM-Bedarf,
unterstützte Inline-Tags, ob ein Referenztext nötig ist, Grenze je Generierung.
Die Pipeline behandelt Engines allein darüber, ohne sie zu kennen.

Neue Engines werden in `cloney/engines/registry.py` eingetragen und sind gegen
ein eingeschleustes Ersatzmodul testbar (siehe `tests/test_f5_german.py`) --
das echte Modell muss dafür nicht vorliegen.
