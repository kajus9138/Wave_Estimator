import os

def renomear_arquivos(diretorio):
    try:
        # Iterar sobre os arquivos no diretório
        for nome_arquivo in os.listdir(diretorio):
            caminho_antigo = os.path.join(diretorio, nome_arquivo)
            
            # Verifica se é um arquivo
            if os.path.isfile(caminho_antigo):
                novo_nome = "L1_" + nome_arquivo
                caminho_novo = os.path.join(diretorio, novo_nome)
                
                # Renomear o arquivo
                os.rename(caminho_antigo, caminho_novo)
                print(f"Renomeado: {nome_arquivo} -> {novo_nome}")
    except Exception as e:
        print(f"Erro ao renomear arquivos: {e}")

# Caminho do diretório a ser renomeado
diretorio = r"C:\Users\ksilva\Documents\Wave_Estimator\dados\processados\temp"

renomear_arquivos(diretorio)
