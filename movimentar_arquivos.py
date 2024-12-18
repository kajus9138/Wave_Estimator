
import os
import shutil

# Lista de números fornecidos
numeros = [
    5, 12, 14, 16, 18, 19, 20, 21, 22, 25, 26, 27, 33, 34, 35, 38, 41, 46, 53, 55, 
    58, 67, 68, 75, 79, 88, 95, 96, 97, 102, 103, 108, 117, 118, 119, 120, 128, 130, 
    135, 137, 139, 140, 144, 155, 157, 160, 171, 173, 177, 178, 183, 188, 190, 194, 
    211, 216, 217, 221, 223, 228, 230, 232, 234, 251, 255, 257, 261, 264, 271, 274, 
    275, 277, 284, 292, 293, 296, 300, 301, 304, 308, 321, 325, 336, 342, 344, 350, 
    352, 354, 356, 358, 362, 369, 373, 374, 380, 381, 386, 391, 408, 413, 414, 419, 
    423, 425, 431, 433, 434, 442, 447, 451, 457, 459, 461, 466, 474, 476, 481, 485, 
    486, 490, 498, 499, 500, 501, 504, 510, 512, 516, 517, 523, 524, 526, 537, 540, 
    541, 555, 561, 562, 572, 573, 575, 583, 584, 590, 594, 600, 603, 611, 613, 614, 
    615, 616, 623, 624, 625, 630, 631, 632, 638, 640, 649, 656, 666, 669, 673, 675, 
    676, 677, 680, 683, 690, 691, 699, 701, 702, 704, 709, 711, 713, 716, 717, 722, 
    732, 737, 741, 743, 745, 747, 753, 754, 758, 766, 782, 791, 793, 798, 800, 806, 
    818, 832, 833, 837, 838, 839, 848, 854, 858, 864, 866, 868, 872, 875, 879, 881, 
    883, 885, 889, 891, 893, 901, 903, 905, 906, 911, 913, 915, 922, 924, 925, 929, 
    931, 932, 933, 935, 938, 949, 951, 952, 954, 955, 957, 958, 965, 968, 976, 978, 
    983, 985, 989, 991
]



# Pastas de origem e destino
pasta_origem = r"C:\Users\ksilva\Documents\Wave_Estimator\dados\brutos\Ondas_PMG"
pasta_destino = r"C:\Users\ksilva\Documents\Wave_Estimator\dados\processados\temp"

# Garantir que a pasta de destino exista
os.makedirs(pasta_destino, exist_ok=True)

# Função para verificar se o número no final do arquivo está na lista
def contem_numero_exato(arquivo, numeros):
    # Extrair o número no final do arquivo (ex.: 'teste0001' -> 1)
    nome_arquivo, _ = os.path.splitext(arquivo)
    if nome_arquivo.startswith("teste") and nome_arquivo[5:].isdigit():
        numero_arquivo = int(nome_arquivo[5:])  # Converter o número do final para inteiro
        return numero_arquivo in numeros
    return False

# Percorrer os arquivos na pasta de origem
for arquivo in os.listdir(pasta_origem):
    if contem_numero_exato(arquivo, numeros):
        # Mover o arquivo para a pasta de destino
        caminho_origem = os.path.join(pasta_origem, arquivo)
        caminho_destino = os.path.join(pasta_destino, arquivo)
        shutil.move(caminho_origem, caminho_destino)
        print(f"Arquivo {arquivo} movido para {pasta_destino}")

print("Arquivos selecionados foram movidos com sucesso.")

