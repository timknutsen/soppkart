# soppkart

Interaktivt sopp-kart rundt Ås som lærer av egne funn. Publiseres med GitHub
Pages fra `docs/`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python soppkart_as.py --probe
.venv/bin/python soppkart_as.py        # skriver docs/index.html
git add -A && git commit -m "Oppdater kart" && git push
```

## På mobilen

- Logg inn med **Logg her** eller **Mine**. Første gang trykker du **Opprett konto**
  og bekrefter e-posten.
- Posisjonsknappen (øverst til venstre) viser hvor du står. Første gang den finner
  deg, åpnes loggeruta der. Du kan også trykke hvor som helst i kartet.
- Velg art, fyll eventuelt inn mengde og notat, og trykk **Funnet** eller **Ingen funn**.
- Registreringene lagres i Supabase (tabell `soppfunn`). Radene er bare synlige for
  kontoen som la dem inn, så de er private selv om kartet og repoet er offentlige.
- Uten nett legges registreringene i en kø på telefonen (gul ring) og sendes
  automatisk når nettet er tilbake.
- **Mine** viser registreringene, med **Last ned CSV** og **Logg ut**. Trykk på en
  prikk for å slette den.

## Modellen bruker funnene

Kopier `.env.example` til `.env` og fyll inn samme e-post og passord som på
mobilen. `soppkart_as.py` henter da funnene fra Supabase når kartet bygges.
`mine_funn.csv` leses også hvis den finnes, men sjekkes ikke inn.

Posisjon krever https, så den virker på GitHub Pages og `localhost`, men ikke
når fila åpnes direkte fra disk.

Se `CLAUDE.md` for datakilder, modell og kjente svakheter.
