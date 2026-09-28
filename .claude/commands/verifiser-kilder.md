---
description: Sjekk at alle eksterne datakilder svarer som forventet
---

1. Kjør `python soppkart_as.py --probe`.
2. Høydedata: kom det et tall ut? Hvis ikke, sjekk parameterformatet mot
   https://ws.geonorge.no/hoydedata/v1/
3. SR16: skriv ut råsvaret fra GetFeatureInfo. Finn de faktiske feltnavnene for
   treslag, bonitet og alder, og oppdater `SR16_FIELDS` og regexen i
   `sr16_featureinfo` slik at parsingen treffer. Dette er den delen som er
   minst verifisert.
4. GBIF: hent ett kjent taxonKey og bekreft at søket returnerer funn i Ås-området.
5. Oppsummer status per kilde i en kort tabell, og oppdater "Kjente svakheter"
   i `CLAUDE.md` hvis noe er løst.
