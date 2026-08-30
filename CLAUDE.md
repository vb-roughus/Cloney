# Cloney

Lokales Voice Cloning für deutsche Langform-Texte. Siehe README.md für Aufbau,
Installation und Bedienung.

## Arbeitsweise

- Entwickelt wird auf einem Branch, gemergt wird über einen Pull Request gegen `main`.
- **Ist die CI grün, wird der PR sofort gemergt** -- ohne Rückfrage. Beim Öffnen
  wird dafür GitHubs Auto-Merge eingeschaltet: sonst bleibt ein PR liegen, wenn
  die Prüfung länger dauert als die Antwort, in der er angekündigt wurde.
- Vor jedem Push laufen `pytest`, `ruff check .` und `ruff format --check .`.
  Die Testsuite kommt ohne GPU, ohne Modelldownload und ohne Netz aus.

## Worauf beim Ändern zu achten ist

Zwei Entwurfsentscheidungen tragen das Projekt. Wer sie aufweicht, nimmt Cloney
seinen Zweck:

1. **Normalisierung passiert vor dem Modell.** Ziffern, Symbole und Abkürzungen
   werden regelbasiert ausgeschrieben (`cloney/core/normalize.py`). Neue Regeln
   gehören mit einem Tabellentest in `tests/test_normalize.py` versehen.
2. **Jeder Chunk wird gegen dieselbe unveränderte Referenz konditioniert**, nie
   gegen das Ergebnis des Vorgängers. Daraus folgen Reproduzierbarkeit, Resume
   und das einzelne Neuwürfeln eines Satzes.

   Der Vergleichslauf (`cloney/core/compare.py`) zieht daraus die Konsequenz:
   alle Varianten teilen sich die aus der Vergleichskennung abgeleiteten Seeds
   und laufen ohne Wiederholungsversuche. Wer daran rührt, misst wieder Regler
   und Zufall zugleich -- und die Tabelle beantwortet nicht mehr die gestellte
   Frage.

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
