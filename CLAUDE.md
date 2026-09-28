# soppkart

Personlig sopp-habitatmodell rundt Ås. Bygger et interaktivt Leaflet-kart som
kombinerer offentlige funndata med Tims egne funn, og trener en modell per art.

## Kjør

```bash
python soppkart_as.py --probe                 # sjekk at eksterne API-er svarer
python soppkart_as.py                         # de fire høstartene
python soppkart_as.py --all-species --grid-m 100
```

Førstegangs kjøring fyller `covariates_cache.json` og tar minutter. Senere
kjøringer er sekunder. Slett cachen bare hvis rutenettet (senter, radius,
`--grid-m`) endres.

Kartet skrives til `docs/index.html`, som GitHub Pages publiserer
(`timknutsen.no/soppkart/`). Sannsynlighetsflatene havner i `modell/`, som ikke
sjekkes inn.

## Datakilder (alle åpne, ingen nøkler)

| Kilde | Bruk | Endepunkt |
|---|---|---|
| GBIF | funn av åtte matsopparter | `api.gbif.org/v1` |
| Kartverket høydedata | høyde per rute, gir helning og nordvendthet | `ws.geonorge.no/hoydedata/v1/punkt` |
| NIBIO SR16 | treslag, bonitet, kronedekning, skoghøyde per rute | `wms.nibio.no/cgi-bin/sr16` |
| NIBIO AR50 | treslag som bakgrunnslag | `wms.nibio.no/cgi-bin/ar50_2` |
| Kartverket topo | bakgrunnskart | `cache.kartverket.no/v1/wmts` |

## Egne funn

`mine_funn.csv`: `art,lat,lon,dato,funnet,mengde,notat`

`funnet=0` betyr at vi lette der og fant ingenting. Disse radene er de eneste
ekte fraværsdataene modellen har, og de er like viktige som funnene.

I kartet (også på mobil) logges funn med posisjonsknappen, «Logg her» eller et
trykk i kartet. Loggede funn lagres i nettleserens localStorage, og «Mine» gir
CSV-linjer som limes inn i `mine_funn.csv`. Kartet er statisk, så nye funn
påvirker modellen først etter ny bygging og push.

## Modell

Per art: HistGradientBoostingClassifier på rutenivå. Positive er GBIF-funn
(vekt 1) og egne funn (vekt `OWN_WEIGHT`, nå 3). Negative er egne blanke søk
(samme vekt) og tilfeldige bakgrunnsruter. Klassene balanseres på vekt.
Kryssvalidert AUC og permutasjonsviktighet skrives til terminal, og
sannsynlighetsflaten lagres som `.npy` per art.

## Konvensjoner

- Python, standard bibliotek pluss tabellen i `requirements.txt`. Ikke dra inn
  geopandas eller rasterio uten grunn, hele poenget er at dette kjører hvor som helst.
- Aldri em-strek i kode, kommentarer eller output.
- Norsk i brukervendt tekst og kolonnenavn, engelsk i kode.
- Ekstern I/O skal feile mykt: logg til stderr og fortsett, ikke krasj hele kjøringen.
- Nye arter legges i `SPECIES` med vitenskapelig navn, farge og sesongmåneder.

## Kjente svakheter

- SR16 GetFeatureInfo svarer bare med verdier i `text/html`, og bare for ett lag
  per kall. Lagene står i `SR16_LAYERS`. Det finnes ikke noe alderslag, så
  skoghøyde (`SRRHOYDEM`, dm) er stedfortreder. `--probe` viser råsvaret.
- OpenStreetMap-fliser blokkeres når fila åpnes fra disk (ingen referrer).
  Kartverket topo er standard.
- `docs/index.html` er offentlig. Egne funn i `mine_funn.csv` havner i kartet
  og i repoet, så hemmelige steder blir synlige hvis repoet er offentlig.
- GBIF-funn er dugnadsdata og klumper seg rundt stier og tettsteder. Modellen
  lærer delvis hvor folk går. Egne blanke søk er motgiften.
- 250 m rutenett er grovt for sopp. 100 m er bedre når cachen er varm.
- Ingen værvariabler ennå. Nedbør og jordtemperatur de siste ukene styrer
  faktisk fruktifisering, kartet viser bare habitat.
