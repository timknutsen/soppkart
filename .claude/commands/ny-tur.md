---
description: Logg funn fra en sopptur og bygg kartet på nytt
---

Brukeren har vært på sopptur. Turen: $ARGUMENTS

1. Les `mine_funn.csv` og `CLAUDE.md`.
2. Gjør om beskrivelsen til rader i `mine_funn.csv`. Spør om koordinater hvis
   de mangler, ikke gjett. Husk blanke søk (`funnet=0`) hvis brukeren nevner
   steder der det ikke var noe.
3. Sjekk at arten finnes i `SPECIES` i `soppkart_as.py`. Hvis ikke, foreslå å
   legge den til med vitenskapelig navn, farge og sesongmåneder.
4. Kjør `python soppkart_as.py` og rapporter per art: antall egne funn og
   blanke, CV-AUC, og de tre viktigste variablene.
5. Si fra hvis AUC faller merkbart fra forrige kjøring.
