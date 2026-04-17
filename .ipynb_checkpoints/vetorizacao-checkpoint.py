import os
import io
import time
import gspread
import torch
import clip
from PIL import Image
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from supabase import create_client, Client

# ─────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────
load_dotenv()

SPREADSHEET_ID    = "1J2K3XzSAJd7GZP65EpK0NGfduV7g-9PTViM_5lmG8sg"
ABA_PRODUTOS      = "PRODUTOS"
COLUNA_VETORIZADA = "FotoVetorizada"
LOTE_SIZE         = 10
SLEEP_ENTRE_LOTES = 2

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

# ─── Mapeamento Sheets → Supabase ───────
COL_PRODUTO_ID    = "ProdutoID"
COL_COD_BARRAS    = "CodigoBarras"
COL_DESCRICAO     = "Descricao"
COL_TAMANHO       = "Tamanho"
COL_COR           = "Cor"
COL_CATEGORIA     = "Categoria"
COL_FOTO_PATH     = "Foto"
COL_ATIVO         = "Ativo"

# ─────────────────────────────────────────
# AUTENTICAÇÃO
# ─────────────────────────────────────────
def autenticar():
    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    gc    = gspread.authorize(creds)
    drive = build("drive", "v3", credentials=creds)
    return gc, drive

# ─────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────
def carregar_produtos(gc):
    sh    = gc.open_by_key(SPREADSHEET_ID)
    aba   = sh.worksheet(ABA_PRODUTOS)
    dados = aba.get_all_records()
    return aba, dados

def marcar_vetorizado(aba, linha_index, headers):
    if COLUNA_VETORIZADA not in headers:
        print(f"    ⚠️  Coluna '{COLUNA_VETORIZADA}' não encontrada no Sheets.")
        return
    col_num = headers.index(COLUNA_VETORIZADA) + 1
    aba.update_cell(linha_index + 2, col_num, True)

# ─────────────────────────────────────────
# GOOGLE DRIVE — busca por ProdutoID no nome
# ─────────────────────────────────────────
def buscar_arquivo_por_path(drive, foto_path: str):
    """
    foto_path ex: 'PRODUTOS_Images/P311400800.Foto.234906.jpg'
    Extrai o ProdutoID (parte antes de '.Foto.') e busca no Drive.
    Ex: 'P311400800.Foto.234906.jpg' → busca por 'P311400800.Foto'
    """
    nome_arquivo = foto_path.split("/")[-1]          # 'P311400800.Foto.234906.jpg'
    produto_id   = nome_arquivo.split(".Foto.")[0]   # 'P311400800'

    query = f"name contains '{produto_id}.Foto' and trashed = false"

    resultado = drive.files().list(
        q=query, spaces="drive",
        fields="files(id, name)", pageSize=1
    ).execute()

    arquivos = resultado.get("files", [])
    if arquivos:
        print(f"    🔎 Arquivo encontrado: {arquivos[0]['name']}")
    return arquivos[0]["id"] if arquivos else None

def baixar_imagem_drive(drive, file_id) -> Image.Image:
    request    = drive.files().get_media(fileId=file_id)
    buffer     = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# ─────────────────────────────────────────
# CLIP — embedding (OpenAI local)
# ─────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ViT-B-32.pt")

print("⏳ Carregando modelo CLIP...")
modelo_clip, preprocess_clip = clip.load(MODEL_PATH, device=DEVICE)
print(f"✅ Modelo CLIP carregado! (device: {DEVICE})\n")

def gerar_embedding(imagem: Image.Image) -> list:
    tensor = preprocess_clip(imagem).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = modelo_clip.encode_image(tensor)
    return features[0].cpu().numpy().tolist()

# ─────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────
def conectar_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def salvar_no_supabase(supabase: Client, produto: dict, embedding: list):
    supabase.rpc("upsert_produto", {
        "p_produto_id":    produto.get(COL_PRODUTO_ID),
        "p_codigo_barras": produto.get(COL_COD_BARRAS),
        "p_descricao":     produto.get(COL_DESCRICAO),
        "p_tamanho":       str(produto.get(COL_TAMANHO, "")),
        "p_cor":           produto.get(COL_COR),
        "p_categoria":     produto.get(COL_CATEGORIA),
        "p_foto_path":     produto.get(COL_FOTO_PATH),
        "p_embedding":     embedding,
    }).execute()

# ─────────────────────────────────────────
# FLUXO PRINCIPAL
# ─────────────────────────────────────────
def main():
    print("🔐 Autenticando...")
    gc, drive = autenticar()
    supabase  = conectar_supabase()

    print("📋 Carregando planilha PRODUTOS...")
    aba, dados = carregar_produtos(gc)
    headers    = list(dados[0].keys()) if dados else []

    pendentes = [
        (i, p) for i, p in enumerate(dados)
        if str(p.get(COL_ATIVO, "")).upper() == "TRUE"
        and str(p.get(COLUNA_VETORIZADA, "")).upper() != "TRUE"
    ]

    total = len(pendentes)
    print(f"🔍 {total} produto(s) pendente(s) de vetorização.\n")

    if total == 0:
        print("✅ Nenhum produto pendente. Tudo já vetorizado!")
        return

    erros = []

    for lote_inicio in range(0, total, LOTE_SIZE):
        lote = pendentes[lote_inicio : lote_inicio + LOTE_SIZE]
        print(f"📦 Lote {lote_inicio // LOTE_SIZE + 1} — {len(lote)} produto(s)...")

        for i, (idx, produto) in enumerate(lote):
            sku       = produto.get(COL_PRODUTO_ID, "???")
            foto_path = produto.get(COL_FOTO_PATH, "")

            print(f"  [{lote_inicio + i + 1}/{total}] {sku} — {foto_path}")

            try:
                file_id = buscar_arquivo_por_path(drive, foto_path)
                if not file_id:
                    msg = f"Arquivo '{foto_path}' não encontrado no Drive"
                    print(f"    ⚠️  {msg}")
                    erros.append({"sku": sku, "motivo": msg})
                    continue

                imagem    = baixar_imagem_drive(drive, file_id)
                embedding = gerar_embedding(imagem)
                salvar_no_supabase(supabase, produto, embedding)
                marcar_vetorizado(aba, idx, headers)
                print(f"    ✅ OK!")

            except Exception as e:
                print(f"    ❌ Erro: {e}")
                erros.append({"sku": sku, "motivo": str(e)})

        if lote_inicio + LOTE_SIZE < total:
            print(f"  ⏸️  Aguardando {SLEEP_ENTRE_LOTES}s...\n")
            time.sleep(SLEEP_ENTRE_LOTES)

    print("\n" + "═" * 45)
    print(f"✅ {total - len(erros)}/{total} vetorizados com sucesso.")
    if erros:
        print(f"❌ {len(erros)} erro(s):")
        for e in erros:
            print(f"   • {e['sku']}: {e['motivo']}")
    print("═" * 45)

if __name__ == "__main__":
    main()