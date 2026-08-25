## Worum geht es

<!-- Was ändert sich und warum. Wenn die Änderung eine der beiden Kernthesen
     berührt -- Normalisierung vor dem Modell, feste Referenz gegen Voice-Drift --
     bitte hier begründen. -->

## Wie geprüft

<!-- Was tatsächlich ausgeführt wurde, nicht was ausgeführt werden könnte. -->

- [ ] `pytest` (läuft ohne GPU gegen DummyEngine/DummyASR)
- [ ] `ruff check .` und `ruff format --check .`
- [ ] Bei Änderungen an der Normalisierung: Tabellentests in `tests/test_normalize.py` ergänzt
- [ ] Bei Änderungen an einer Engine: gegen echte Hardware geprüft (welche Karte, welches Modell)

## Auswirkung auf den VRAM-Bedarf

<!-- Lädt die Änderung ein weiteres Modell? Wenn ja: bleibt das Phasenmodell
     eingehalten, also nie zwei große Modelle gleichzeitig resident? -->
