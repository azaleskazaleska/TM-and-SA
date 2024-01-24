# modelowanie-tematyczne-i-analiza-sentymentu
Modelowanie tematyczne i analiza sentymentu pojazdów elektrycznych na podstawie danych z Reddit.

Dane wykorzystane w tym projekcie pochodzą ze strony: https://the-eye.eu/redarcs/. Wykorzystano dwa subreddity: r/Cars oraz r/cartalk i pobrano pliki dla postów oraz komentarzy. Poniżej linki umożliwiające pobranie plików JSON z tymi danymi:
- https://the-eye.eu/redarcs/files/cars_submissions.zst
- https://the-eye.eu/redarcs/files/cars_comments.zst
- https://the-eye.eu/redarcs/files/Cartalk_submissions.zst
- https://the-eye.eu/redarcs/files/Cartalk_comments.zst

Z powyższych zbiorów wydrębniono posty związane z pojazdami elektrycznymi zamieszczone w okresie 01/01/2012 - 31/12/2022 i powiązane z tymi postami komentarze. Tak przygotowane dane są dostępne w tym repozytorium.

Do przypisania wartości sentymentu dla postów i komentarzy wykorzystano SentimentIntensityAnalyzer z NLTK, a w procesie modelowania tematycznego użyto utajonej alokazji Dirichleta i zidentyfikowano 3 główne grupy tematyczne, dla różnych podzbiorów.
