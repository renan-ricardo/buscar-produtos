import os, sys, json, requests, io, torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ─── Pré-processamento (MESMO do buscar.py) ────────────────────
def preprocessar(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.convert("RGB")
    return image

# ─── Carrega modelo ────────────────────────────────────────────
print("Carregando modelo CLIP...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sb        = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
print("Modelo carregado!\n")

# ─── Busca produtos sem embedding ou para revetorizar ──────────
revetorizar = len(sys.argv) > 1 and sys.argv[1] == "revetorizar"

if revetorizar:
    print("⚠️  Modo REVETORIZAR — atualizando todos os embeddings...")
    produtos = sb.table("produtos").select("produto_id, foto_path").execute().data
else:
    print("Buscando produtos sem embedding...")
    produtos = sb.table("produtos").select("produto_id, foto_path").is_("embedding", "null").execute().data

total = len(produtos)
print(f"Total a processar: {total}\n")

erros = []
ok    = 0

for i, produto in enumerate(produtos, 1):
    produto_id = produto["produto_id"]
    foto_path  = produto.get("foto_path")

    print(f"[{i}/{total}] {produto_id} ... ", end="", flush=True)

    if not foto_path:
        print("⚠️  sem foto_path, pulando")
        erros.append({"produto_id": produto_id, "erro": "sem foto_path"})
        continue

    try:
        # ─── URL assinada (bucket privado) ─────────────────────
        signed   = sb.storage.from_("produtos").create_signed_url(foto_path, 60)
        foto_url = signed["signedURL"]

        response = requests.get(foto_url, timeout=15)
        response.raise_for_status()

        image  = Image.open(io.BytesIO(response.content)).convert("RGB")
        image  = preprocessar(image)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output    = model.get_image_features(**inputs)
            embedding = output if isinstance(output, torch.Tensor) else output.pooler_output
            embedding = F.normalize(embedding, dim=-1)
            vetor     = embedding[0].tolist()

        sb.table("produtos").update({"embedding": vetor}).eq("produto_id", produto_id).execute()

        print("✅")
        ok += 1

    except Exception as e:
        print(f"❌ {e}")
        erros.append({"produto_id": produto_id, "erro": str(e)})

# ─── Resumo ────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"✅ Sucesso : {ok}/{total}")
print(f"❌ Erros   : {len(erros)}/{total}")
if erros:
    print("\nProdutos com erro:")
    for e in erros:
        print(f"  - {e['produto_id']}: {e['erro']}")