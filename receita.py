while(True):
    lingua_escolhida = None

    while(lingua_escolhida is None):
        lingua = input("Escreva a lingua desejada: pt ou en\n")

        if lingua not in ("pt", "en"):
            print("Lingua não suportada. Escolha 'pt' ou 'en'.\n")
        
        if lingua == "q":
            print("Ok. Saindo!")
            exit(0)

        lingua_escolhida = lingua

    if lingua_escolhida == 'en':
        print("It is in english")

    if lingua_escolhida == 'pt':
        print("Em português")
