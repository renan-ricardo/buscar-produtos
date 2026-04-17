import sys, os, json, requests, io, torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# ─── Argumentos via N8N Execute Command ────────────────────────
# python3 buscar.py <foto_url> <quantidade> <categoria> <cor> <descricao>
foto_url  = sys.argv[1].strip()
quantidade = int(sys.argv[2])              if len(sys.argv) > 2 else 5
categoria  = sys.argv[3].strip() or None   if len(sys.argv) > 3 else None
cor        = sys.argv[4].strip() or None   if len(sys.argv) > 4 else None
descricao  = sys.argv[5].strip() or None   if len(sys.argv) > 5 else None

if not foto_url:
    print(json.dumps({"erro": "foto_obrigatoria", "resultados": []}, ensure_ascii=False))
    sys.exit(1)

# ─── Pré-processamento de imagem ───────────────────────────────
def preprocessar(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.convert("RGB")
    return image

# ─── Carrega modelo ────────────────────────────────────────────
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sb        = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ─── Vetoriza a foto ───────────────────────────────────────────
response = requests.get(foto_url, timeout=15)
image    = Image.open(io.BytesIO(response.content)).convert("RGB")
image    = preprocessar(image)
inputs   = processor(images=image, return_tensors="pt")

with torch.no_grad():
    output    = model.get_image_features(**inputs)
    embedding = output if isinstance(output, torch.Tensor) else output.pooler_output
    embedding = F.normalize(embedding, dim=-1)
    vetor     = embedding[0].tolist()

# ─── Busca visual no Supabase ──────────────────────────────────
buscar_qtd = quantidade * 4

result     = sb.rpc("buscar_produto_similar", {
    "query_embedding": vetor,
    "match_count":     buscar_qtd
}).execute()

candidatos = result.data or []

# ─── Score híbrido ─────────────────────────────────────────────
PESO_VISUAL  = 0.6
PESO_TEXTUAL = 0.4

def score_textual(produto):
    pontos = 0
    total  = 0

    if categoria:
        total += 1
        cat_prod = (produto.get("categoria") or "").lower().strip()
        if categoria.lower() in cat_prod or cat_prod in categoria.lower():
            pontos += 1

    if cor:
        total += 1
        cor_prod = (produto.get("cor") or "").lower().strip()
        if cor.lower() in cor_prod or cor_prod in cor.lower():
            pontos += 1

    if descricao:
        total += 1
        desc_prod = (produto.get("descricao") or "").lower().strip()
        palavras  = [p for p in descricao.lower().split() if len(p) > 2]
        if any(p in desc_prod for p in palavras):
            pontos += 1

    return pontos / total if total > 0 else 0.0

def score_final(produto):
    s_visual  = float(produto.get("similaridade", 0))
    s_textual = score_textual(produto)

    if not any([categoria, cor, descricao]):
        return s_visual

    return (s_visual * PESO_VISUAL) + (s_textual * PESO_TEXTUAL)

for p in candidatos:
    p["score_visual"]  = round(float(p.get("similaridade", 0)), 4)
    p["score_textual"] = round(score_textual(p), 4)
    p["score_final"]   = round(score_final(p), 4)

candidatos.sort(key=lambda x: x["score_final"], reverse=True)
resultados = candidatos[:quantidade]

# ─── Output ────────────────────────────────────────────────────
print(json.dumps({
    "resultados": resultados
}, ensure_ascii=False))