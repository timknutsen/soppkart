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

- Posisjonsknappen (øverst til venstre) viser hvor du står. Første gang den finner
  deg, åpnes loggeruta der.
- **Logg her** åpner loggeruta på din posisjon. Du kan også trykke hvor som helst
  i kartet. Velg art, skriv mengde eller notat, og trykk **Funnet** eller
  **Ingen funn**.
- Funnene lagres i nettleseren på telefonen og vises som grønne (funnet) eller
  grå (ingen funn) prikker.
- **Mine** viser CSV-linjene. **Del / kopier** sender dem videre. Lim dem inn i
  `mine_funn.csv`, og bygg og push kartet på nytt.

Posisjon krever https, så den virker på GitHub Pages og `localhost`, men ikke
når fila åpnes direkte fra disk.

Se `CLAUDE.md` for datakilder, modell og kjente svakheter.
