"""
AI RAG Microservice for Knowledge Management System
===================================================
Specialized microservice handling:
1. Google Gemini Embeddings (gemini-embedding-001)
2. ChromaDB Vector Store Persistence
3. Retrieval-Augmented Generation (RAG)
4. Google Gemini LLM Chat Inference
"""

import os
import io
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
import uvicorn

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PORT = int(os.getenv("PORT", "22004"))
CHROMA_PATH = BASE_DIR / os.getenv("CHROMA_PATH", "chroma_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# ---------------------------------------------------------
# INITIALIZE GEMINI CLIENT
# ---------------------------------------------------------
gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[AI Microservice] Error init Gemini Client: {e}")
        gemini_client = None

chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

app = FastAPI(
    title="AI RAG Microservice - Knowledge Management System",
    version="1.0.0",
    description="Dedicated microservice for Gemini LLM Inference, ChromaDB Vector Retrieval, and RAG search."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------
class ReportEmbedRequest(BaseModel):
    report_id: str
    title: str
    document_number: str
    document_datetime: str
    document_type: str
    status: str
    officer: str
    location: Optional[str] = ""
    uraian: Optional[str] = ""
    pemeriksaan: Optional[str] = ""
    tindakan: Optional[str] = ""
    hasil: Optional[str] = ""
    rekomendasi: Optional[str] = ""
    catatan: Optional[str] = ""
    pdf_url: Optional[str] = ""
    attachment_urls: Optional[str] = ""

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ---------------------------------------------------------
# DATABASE MEDIA HELPER (FALLBACK FOR REPORT ATTACHMENTS & PDFS)
# ---------------------------------------------------------
def get_report_media_from_db(report_id: str) -> dict:
    try:
        import pyodbc
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};SERVER=62.171.169.192,1433;DATABASE=sistem-knowladge;UID=sa;PWD=Tijar200#MSSQL!;",
            timeout=4
        )
        cur = conn.cursor()
        cur.execute("SELECT pdf_url FROM dbo.reports WHERE report_id = ?", (report_id,))
        rep_row = cur.fetchone()
        pdf_url = rep_row[0] if (rep_row and rep_row[0]) else f"/documents/detail/{report_id}"
        
        cur.execute("SELECT url FROM dbo.attachments WHERE report_id = ? ORDER BY sort_order, id", (report_id,))
        att_rows = cur.fetchall()
        att_urls = [r[0] for r in att_rows if r and r[0]]
        conn.close()
        return {
            "pdf_url": pdf_url,
            "attachment_urls": "|".join(att_urls)
        }
    except Exception as e:
        print(f"[AI Microservice] DB Media Lookup Warning: {e}")
        return {"pdf_url": f"/documents/detail/{report_id}", "attachment_urls": ""}

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_text(val: Optional[str]) -> str:
    if not val:
        return ""
    return str(val).strip()

def is_pure_conversational(text: str) -> bool:
    """Mendeteksi apakah pertanyaan adalah basa-basi, sapaan, identitas, atau ucapan terima kasih murni."""
    import re
    t = text.lower().strip()
    cleaned = re.sub(r'[^\w\s]', ' ', t).strip()
    words = cleaned.split()
    if not words:
        return True

    # Kata kunci teknis & arsip — jika ada kata ini, BUKAN percakapan murni
    technical_keywords = {
        "laporan", "dokumen", "arsip", "insiden", "kerusakan", "perbaikan",
        "pemeliharaan", "ups", "switch", "freon", "pac", "baterai", "tegangan",
        "suhu", "ac", "sfp", "server", "router", "kabel", "fiber", "network",
        "jaringan", "data center", "petugas", "foto", "lampiran", "berkas",
        "rekomendasi", "pemeriksaan", "tindakan", "hasil", "solusi",
        "masalah", "kendala", "penanganan", "catatan", "fluktuasi", "bocor",
        "kebocoran", "throughout", "throughput", "optik", "optic"
    }

    if any(w in technical_keywords for w in words):
        return False
    if "inc-" in cleaned or "doc-" in cleaned:
        return False

    # Pola sapaan, ucapan terima kasih, dan identitas murni
    chitchat_exact = {
        "siapa anda", "siapa kamu", "kamu siapa", "anda siapa", "siapakah anda",
        "siapa diri anda", "siapa nama anda", "siapa namamu",
        "halo", "hai", "hei", "hello", "hi", "hey",
        "apa kabar", "gimana kabar", "bagaimana kabar",
        "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
        "terima kasih", "makasih", "terimakasih", "thanks", "thank you",
        "terima kasih banyak", "makasih banyak", "matur nuwun",
        "bye", "dadah", "selamat tinggal", "sampai jumpa",
        "ok", "oke", "okay", "siap", "baik", "baiklah", "sip",
        "tes", "test", "testing", "ping",
        "kamu hebat", "kamu pintar", "mantap", "keren", "hebat", "good job", "bagus"
    }

    if cleaned in chitchat_exact:
        return True

    # Jika diawali frasa sapaan dan kalimatnya pendek (<= 7 kata)
    for p in chitchat_exact:
        if cleaned.startswith(p) and len(words) <= 7:
            return True

    return False

def is_catalog_query(text: str) -> bool:
    """Mendeteksi apakah pengguna menanyakan katalog atau daftar seluruh dokumen laporan yang ada di sistem."""
    import re
    t = text.lower().strip()
    cleaned = re.sub(r'[^\w\s]', ' ', t).strip()

    # Jika menyebutkan nomor dokumen spesifik (misal INC-20261027-001), BUKAN katalog!
    if "inc-" in cleaned or "doc-" in cleaned or "inc " in cleaned or "doc " in cleaned:
        return False

    catalog_phrases = [
        "dokumen apa aja", "dokumen apa saja", "laporan apa aja", "laporan apa saja",
        "daftar dokumen", "daftar laporan", "list dokumen", "list laporan",
        "apa saja dokumen", "apa aja dokumen", "apa saja laporan", "apa aja laporan",
        "ada apa aja", "ada apa saja", "ada dokumen apa", "ada laporan apa",
        "ada arsip apa", "sebutkan dokumen", "sebutkan laporan", "sebutkan semua dokumen",
        "sebutkan seluruh dokumen", "katalog dokumen", "katalog laporan",
        "seluruh dokumen", "seluruh laporan", "semua dokumen", "semua laporan",
        "apa yang tercatat", "apa saja yang tercatat", "apa saja yang ada", "apa aja yang ada",
        "dokumen yang ada", "laporan yang ada", "isi arsip", "koleksi dokumen", "koleksi laporan",
        "arsip apa saja", "arsip apa aja"
    ]

    for p in catalog_phrases:
        if p in cleaned:
            return True
    return False

def get_embedding(text: str) -> Optional[List[float]]:
    """Menghasilkan vector embedding menggunakan Gemini embedding model."""
    global gemini_client
    # Coba muat gemini_client jika belum ada
    current_key = os.getenv("GEMINI_API_KEY", "")
    if current_key and not gemini_client:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=current_key)
        except Exception:
            pass

    if gemini_client:
        candidate_embed_models = [EMBEDDING_MODEL, "gemini-embedding-001", "gemini-embedding-2"]
        for em in candidate_embed_models:
            if not em:
                continue
            try:
                response = gemini_client.models.embed_content(
                    model=em,
                    contents=text
                )
                if response and response.embeddings:
                    return response.embeddings[0].values
            except Exception as e:
                print(f"[AI Microservice] Gemini Embed Error ({em}): {e}")
                continue
        return None
    return None

def build_report_chunks(data: ReportEmbedRequest) -> List[Dict[str, Any]]:
    chunks = []
    
    header_parts = [
        f"Judul Laporan: {data.title}",
        f"Nomor Dokumen: {data.document_number}",
        f"Tanggal/Waktu: {data.document_datetime}",
        f"Jenis Dokumen: {data.document_type}",
        f"Status: {data.status}",
        f"Petugas Penanggung Jawab: {data.officer}"
    ]
    if data.location:
        header_parts.append(f"Lokasi: {data.location}")
    if data.attachment_urls:
        att_count = len([u for u in data.attachment_urls.split("|") if u])
        header_parts.append(f"Dokumentasi Lampiran Foto: Tersedia ({att_count} foto dokumentasi teknis terlampir)")
        
    chunks.append({
        "chunk_id": f"{data.report_id}_chunk_0",
        "text": "\n".join(header_parts),
        "section": "identitas_dokumen"
    })
    
    sections = [
        ("uraian", "1. Uraian Masalah", data.uraian),
        ("pemeriksaan", "2. Hasil Pemeriksaan & Analisis Teknis", data.pemeriksaan),
        ("tindakan", "3. Tindakan / Langkah Solusi Perbaikan", data.tindakan),
        ("hasil", "4. Hasil Akhir Pasca Perbaikan", data.hasil),
        ("rekomendasi", "5. Rekomendasi Pencegahan & Panduan", data.rekomendasi),
        ("catatan", "6. Catatan Tambahan Teknisi & PIC", data.catatan)
    ]
    
    for idx, (sec_key, sec_title, sec_val) in enumerate(sections, start=1):
        clean_val = clean_text(sec_val)
        if clean_val:
            chunk_text = (
                f"Dokumen: {data.title} ({data.document_number})\n"
                f"{sec_title}:\n{clean_val}"
            )
            chunks.append({
                "chunk_id": f"{data.report_id}_chunk_{idx}",
                "text": chunk_text,
                "section": sec_key
            })
            
    return chunks

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI RAG Microservice",
        "gemini_active": gemini_client is not None,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "collection": COLLECTION_NAME,
        "vector_count": collection.count()
    }

@app.post("/ai/embed-report")
async def embed_report(req: ReportEmbedRequest):
    try:
        try:
            collection.delete(where={"report_id": req.report_id})
        except Exception:
            pass
            
        chunks = build_report_chunks(req)
        if not chunks:
            return {"success": False, "message": "Tidak ada konten untuk di-embed"}
            
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        has_embeddings = False
        for item in chunks:
            vec = await get_embedding(item["text"])
            ids.append(item["chunk_id"])
            documents.append(item["text"])
            metadatas.append({
                "report_id": req.report_id,
                "document_number": req.document_number,
                "title": req.title,
                "section": item["section"],
                "officer": req.officer,
                "status": req.status,
                "pdf_url": req.pdf_url or "",
                "attachment_urls": req.attachment_urls or ""
            })
            if vec is not None:
                embeddings.append(vec)
                has_embeddings = True
                
        if has_embeddings and len(embeddings) == len(ids):
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # Upsert dokumen dengan default embedding jika Gemini belum aktif
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        return {
            "success": True,
            "report_id": req.report_id,
            "chunks_count": len(chunks),
            "message": f"Berhasil menyimpan {len(chunks)} chunks ke ChromaDB"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses embedding: {str(e)}"
        )

@app.post("/search")
async def raw_vector_search(req: SearchRequest):
    try:
        total = collection.count()
        if total == 0:
            return {"query": req.query, "top_k": 0, "results": {"documents": [[]], "distances": [[]], "metadatas": [[]]}}
        actual_top_k = min(max(req.top_k, 1), total)
        query_vec = await get_embedding(req.query)
        if query_vec:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=actual_top_k
            )
        else:
            results = collection.query(
                query_texts=[req.query],
                n_results=actual_top_k
            )
        return {
            "query": req.query,
            "top_k": actual_top_k,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan pencarian ChromaDB: {str(e)}")

@app.post("/ai/search")
def semantic_search(req: SearchRequest):
    try:
        total = collection.count()
        if total == 0:
            return {"success": True, "query": req.query, "total_matches": 0, "results": []}
            
        query_vec = get_embedding(req.query)
        if query_vec:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=min(req.top_k, total)
            )
        else:
            results = collection.query(
                query_texts=[req.query],
                n_results=min(req.top_k, total)
            )
        
        items = []
        if results and "documents" in results and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else []
            distances = results["distances"][0] if "distances" in results else []
            
            for doc, meta, dist in zip(docs, metas, distances):
                items.append({
                    "snippet": doc,
                    "metadata": meta,
                    "distance": dist
                })
                
        return {
            "success": True,
            "query": req.query,
            "total_matches": len(items),
            "results": items
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal melakukan semantic search: {str(e)}"
        )

@app.post("/chat")
@app.post("/ai/chat")
def rag_chat(req: ChatRequest):
    # Threshold relevansi cosine distance: semakin kecil = semakin mirip.
    # Hanya dokumen di bawah threshold yang dianggap relevan.
    RELEVANCE_THRESHOLD = 0.385

    try:
        q_text = req.question.strip()
        if not q_text:
            raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong")
            
        total_vectors = collection.count()
        context_text = ""
        sources = []
        
        # 1. Cek apakah ini sapaan, identitas, atau percakapan murni
        if is_pure_conversational(q_text):
            has_relevant_context = False
            sources = []
            context_text = ""
        # 2. Cek apakah ini pertanyaan katalog / daftar seluruh dokumen
        elif is_catalog_query(q_text):
            cat_res = collection.get(where={"section": "identitas_dokumen"})
            context_blocks = []
            media_cache = {}
            for idx, (doc_id, meta, doc) in enumerate(zip(cat_res.get("ids", []), cat_res.get("metadatas", []), cat_res.get("documents", [])), start=1):
                doc_num = meta.get("document_number", "-")
                title = meta.get("title", "-")
                sec = meta.get("section", "-")
                rep_id = meta.get("report_id", "")
                att_urls = meta.get("attachment_urls") or ""
                pdf_u = meta.get("pdf_url") or ""

                if rep_id and (not att_urls or not pdf_u):
                    if rep_id not in media_cache:
                        media_cache[rep_id] = get_report_media_from_db(rep_id)
                    db_m = media_cache[rep_id]
                    if not att_urls:
                        att_urls = db_m.get("attachment_urls", "")
                    if not pdf_u:
                        pdf_u = db_m.get("pdf_url", "")

                context_blocks.append(f"[DOKUMEN {idx}: {title} | No: {doc_num} | Bagian: {sec}]\n{doc}")
                sources.append({
                    "reportId": rep_id,
                    "documentNumber": doc_num,
                    "title": title,
                    "section": sec,
                    "snippet": doc[:250] + ("..." if len(doc) > 250 else ""),
                    "distance": 0.1,
                    "pdf_url": pdf_u,
                    "attachment_urls": att_urls,
                    "metadata": {
                        **meta,
                        "report_id": rep_id,
                        "document_number": doc_num,
                        "title": title,
                        "pdf_url": pdf_u,
                        "attachment_urls": att_urls
                    },
                    "rerank_score": 0.95
                })
            context_text = "\n\n".join(context_blocks)
            has_relevant_context = len(sources) > 0
        # 3. Pencarian semantik spesifik / umum
        elif total_vectors > 0:
            import re
            doc_code_match = re.search(r'\b(INC-\d{8}-\d{3}|DOC-\d{8}-\d{4})\b', q_text, re.IGNORECASE)
            where_filter = None
            if doc_code_match:
                where_filter = {"document_number": doc_code_match.group(1).upper()}

            q_vec = get_embedding(q_text)
            candidate_k = min(20, total_vectors)

            query_kwargs = {"n_results": candidate_k}
            if where_filter:
                query_kwargs["where"] = where_filter

            if q_vec:
                query_results = collection.query(
                    query_embeddings=[q_vec],
                    **query_kwargs
                )
            else:
                query_results = collection.query(
                    query_texts=[q_text],
                    **query_kwargs
                )
            
            if query_results and "documents" in query_results and len(query_results["documents"]) > 0:
                docs = query_results["documents"][0]
                metas = query_results["metadatas"][0] if "metadatas" in query_results else []
                dists = query_results["distances"][0] if "distances" in query_results else []
                
                # Kumpulkan kandidat yang memenuhi threshold & lakukan hybrid keyword bonus
                stopwords = {
                    "apa", "yang", "dan", "di", "ke", "dari", "pada", "ini", "itu", "ada",
                    "adalah", "terjadi", "laporan", "dokumen", "apakah", "insiden", "kendala",
                    "masalah", "penanganan", "perbaikan", "kerusakan"
                }
                q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_text) if w.lower() not in stopwords]

                valid_candidates = []
                is_officer_query = any(w in q_words for w in ["petugas", "teknisi", "pic", "penanggung", "jawab"])
                for doc, meta, dist in zip(docs, metas, dists):
                    if dist is not None and dist <= RELEVANCE_THRESHOLD:
                        doc_lower = doc.lower()
                        keyword_hits = sum(1 for w in q_words if w in doc_lower)
                        
                        # Jika menanyakan petugas/teknisi/pic dan chunk ini adalah identitas dokumen
                        if is_officer_query and meta.get("section") == "identitas_dokumen":
                            keyword_hits += 2

                        effective_dist = dist - (0.04 * min(keyword_hits, 3))
                        
                        # Beri prioritas pada identitas_dokumen agar metadata kunci (petugas, lokasi, foto) selalu tersedia
                        if meta.get("section") == "identitas_dokumen":
                            effective_dist -= 0.015
                            
                        valid_candidates.append((doc, meta, dist, effective_dist))

                if valid_candidates:
                    # Urutkan berdasarkan effective_dist (paling relevan di depan)
                    valid_candidates.sort(key=lambda c: c[3])
                    best_effective = valid_candidates[0][3]
                    best_rep_id = valid_candidates[0][1].get("report_id", "")

                    doc_chunk_count = {}
                    context_blocks = []
                    media_cache = {}
                    relevant_idx = 0

                    for doc, meta, dist, eff_dist in valid_candidates:
                        rep_id = meta.get("report_id", "")
                        
                        # Anti-leakage: Jika pertanyaan mengarah ke 1 dokumen spesifik,
                        # dokumen lain yang terpaut (> 0.030 dari best_effective) jangan diikutsertakan
                        if rep_id != best_rep_id and (eff_dist - best_effective) > 0.030:
                            continue

                        # Batasi maksimal 3 chunks per dokumen agar konteks proporsional
                        doc_chunk_count[rep_id] = doc_chunk_count.get(rep_id, 0) + 1
                        if doc_chunk_count[rep_id] > 3:
                            continue

                        relevant_idx += 1
                        doc_num = meta.get("document_number", "-")
                        title = meta.get("title", "-")
                        sec = meta.get("section", "-")

                        att_urls = meta.get("attachment_urls") or ""
                        pdf_u = meta.get("pdf_url") or ""

                        # Fallback ke database jika metadata di Chroma belum memiliki media URLs
                        if rep_id and (not att_urls or not pdf_u):
                            if rep_id not in media_cache:
                                media_cache[rep_id] = get_report_media_from_db(rep_id)
                            db_m = media_cache[rep_id]
                            if not att_urls:
                                att_urls = db_m.get("attachment_urls", "")
                            if not pdf_u:
                                pdf_u = db_m.get("pdf_url", "")

                        context_blocks.append(f"[DOKUMEN {relevant_idx}: {title} | No: {doc_num} | Bagian: {sec}]\n{doc}")
                        
                        sources.append({
                            "reportId": rep_id,
                            "documentNumber": doc_num,
                            "title": title,
                            "section": sec,
                            "snippet": doc[:250] + ("..." if len(doc) > 250 else ""),
                            "distance": dist,
                            "pdf_url": pdf_u,
                            "attachment_urls": att_urls,
                            "metadata": {
                                **meta,
                                "report_id": rep_id,
                                "document_number": doc_num,
                                "title": title,
                                "pdf_url": pdf_u,
                                "attachment_urls": att_urls
                            },
                            "rerank_score": 1.0 - (dist if dist is not None else 0.5)
                        })
                        
                    context_text = "\n\n".join(context_blocks)

        has_relevant_context = len(sources) > 0 and bool(context_text.strip())

        global gemini_client
        current_key = os.getenv("GEMINI_API_KEY", "")
        if current_key and not gemini_client:
            try:
                from google import genai
                gemini_client = genai.Client(api_key=current_key)
            except Exception:
                pass

        if not gemini_client:
            # Fallback jawaban informatif jika API key Gemini belum diinput
            if has_relevant_context:
                answer = (
                    f"Berdasarkan arsip Knowledge Management System, ditemukan dokumen terkait:\n\n"
                    f"**{sources[0]['title']} ({sources[0]['documentNumber']})** pada bagian *{sources[0]['section']}*:\n\n"
                    f"{sources[0]['snippet']}\n\n"
                    f"*(Catatan: Masukkan GEMINI_API_KEY valid di file .env AiRagMicroservice untuk mengaktifkan pemrosesan narasi penuh dari Gemini LLM)*."
                )
            else:
                answer = "Informasi belum tercatat dalam arsip laporan teknis saat ini."
            return {"answer": answer, "sources": sources}

        # Prompt untuk Gemini LLM — dibedakan tergantung ada/tidaknya konteks relevan
        if has_relevant_context:
            system_instruction = (
                "Anda adalah Asisten Pengetahuan Teknis perusahaan yang profesional dan ramah. "
                "Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan DOKUMEN KONTEKS di bawah.\n\n"
                "Aturan:\n"
                "1. Jawab secara ringkas, terstruktur, dan akurat berdasarkan fakta dari DOKUMEN KONTEKS.\n"
                "2. Jika pengguna menanyakan daftar atau dokumen apa saja yang ada (misal: 'ada dokumen apa aja', 'daftar laporan', dll):\n"
                "   - Sebutkan SELURUH dokumen laporan yang tertera di DOKUMEN KONTEKS dalam bentuk daftar bernomor rapi.\n"
                "   - Format per butir: [Nomor Dokumen] - [Judul Laporan] ([Jenis Dokumen] / Status: [Status]).\n"
                "   - Berikan ringkasan 1 kalimat singkat mengenai fokus penanganan tiap dokumen.\n"
                "3. Sebutkan nomor dokumen (contoh: INC-20261027-001) dan judul laporan yang relevan saat menjawab pertanyaan teknis.\n"
                "4. Jika pengguna menanyakan tindakan atau rekomendasi, berikan langkah-langkah konkret bernomor.\n"
                "5. ATURAN PENANGANAN FOTO / GAMBAR:\n"
                "   - JIKA PENGGUNA TIDAK MENANYAKAN FOTO/GAMBAR (misal hanya bertanya: 'Carikan dokumen...', 'Apa isi dokumen...', 'Siapa teknisi...', 'Bagaimana penanganan...', dll):\n"
                "     * DILARANG KERAS membuat poin/butir 'Dokumentasi Foto' atau menuliskan 'kartu foto ditampilkan di bawah'.\n"
                "     * Fokuskan seluruh isi jawaban murni pada data teks dokumen yang ditanyakan (Nomor Dokumen, Judul, Tanggal, Petugas, Lokasi, Status, Uraian, Penanganan).\n"
                "   - JIKA DAN HANYA JIKA PENGGUNA EKSPLISIT MENANYAKAN FOTO, GAMBAR, LAMPIRAN, ATAU BUKTI VISUAL:\n"
                "     * Periksa apakah dokumen memiliki lampiran foto (tertera 'Dokumentasi Foto Lampiran: Tersedia').\n"
                "     * Jika foto TERSEDIA: Konfirmasi bahwa dokumentasi foto untuk laporan tersebut tersedia, jelaskan secara singkat temuan visual teknisnya, dan sebutkan bahwa kartu foto ditampilkan di bawah jawaban ini.\n"
                "     * Jika foto TIDAK TERSEDIA: Jelaskan secara jujur bahwa laporan tersebut tidak memiliki lampiran dokumentasi foto (hanya catatan laporan teks).\n"
                "     * Jika pengguna meminta HANYA gambar / BUKAN dokumen, fokuskan penjelasan pada foto visual tanpa mengarahkan ke link dokumen teks.\n"
                "   - JANGAN pernah mengatakan 'sistem AI tidak bisa menampilkan gambar' atau 'saya model teks yang tidak bisa melihat gambar', karena antarmuka web menampilkan kartu foto secara otomatis.\n"
                "6. JANGAN menambahkan informasi yang TIDAK ADA di dalam DOKUMEN KONTEKS. Jika fakta atau insiden yang ditanyakan pengguna tidak tertera sama sekali di DOKUMEN KONTEKS, sampaikan secara jujur bahwa informasi tersebut tidak ditemukan dalam laporan saat ini, jangan mengaitkannya ke laporan lain yang tidak cocok.\n"
                "7. Jawab singkat dan padat, hindari pengulangan atau basa-basi berlebihan."
            )

            full_prompt = f"""{system_instruction}

DOKUMEN KONTEKS:
{context_text}

PERTANYAAN PENGGUNA:
{q_text}

JAWABAN:"""
        else:
            # Tidak ada konteks relevan — jawab secara umum tanpa mengada-ada
            system_instruction = (
                "Anda adalah Asisten Pengetahuan Teknis perusahaan yang profesional dan ramah. "
                "Anda membantu pengguna dengan pertanyaan seputar arsip laporan teknis perusahaan.\n\n"
                "Aturan:\n"
                "1. Jika pengguna bertanya tentang identitas Anda (siapa anda, dll), jawab singkat: "
                "Anda adalah asisten pengetahuan teknis perusahaan yang membantu mencari informasi dari arsip laporan.\n"
                "2. Jika pengguna menanyakan hal di luar cakupan arsip laporan teknis, jawab dengan ramah "
                "bahwa informasi tersebut berada di luar cakupan arsip laporan perusahaan.\n"
                "3. Jangan menyebutkan nama dokumen apapun atau memfabrikasi informasi.\n"
                "4. Jangan menyebutkan foto, gambar, lampiran, atau file karena tidak ada konteks yang relevan.\n"
                "5. Jawab singkat dan padat, maksimal 2-3 kalimat saja."
            )

            full_prompt = f"""{system_instruction}

PERTANYAAN PENGGUNA:
{q_text}

JAWABAN:"""

        response = None
        candidate_models = [LLM_MODEL, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-3.5-flash-lite"]
        last_error = None
        import time
        for mod in candidate_models:
            if not mod:
                continue
            try:
                response = gemini_client.models.generate_content(
                    model=mod,
                    contents=full_prompt
                )
                if response and response.text:
                    break
            except Exception as ex_gen:
                last_error = ex_gen
                err_str = str(ex_gen)
                print(f"[AI Microservice] Generation attempt with {mod} failed: {err_str[:120]}")
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(2)
                continue

        if not response or not response.text:
            if last_error:
                print(f"[AI Microservice] All generation models failed: {last_error}")
            answer = "Maaf, sistem AI sedang mengalami kepadatan antrian (high demand) dari Google Gemini. Silakan coba sesaat lagi."
        else:
            answer = response.text.strip()
        
        # Jika jawaban secara eksplisit menyatakan bahwa informasi tidak ditemukan / di luar cakupan,
        # kosongkan sources agar UI tidak memunculkan kartu dokumen atau foto yang tidak relevan
        ans_lower = answer.lower()
        negative_indicators = [
            "tidak ditemukan dalam laporan",
            "tidak ditemukan dalam dokumen",
            "tidak terdapat dalam laporan",
            "tidak terdapat dalam dokumen",
            "tidak tercatat dalam arsip",
            "di luar cakupan arsip"
        ]
        if any(neg in ans_lower for neg in negative_indicators) and len(answer) < 300:
            sources = []
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        # Fallback graceful jika API key gagal / 401
        err_msg = str(e)
        if "UNAUTHENTICATED" in err_msg or "API key not valid" in err_msg:
            answer = (
                "Koneksi ke dokumen lokal berhasil, namun Google Gemini API Key pada file .env belum valid atau telah kedaluwarsa. "
                "Silakan perbarui GEMINI_API_KEY di file .env agar AI dapat memberikan jawaban naratif."
            )
            return {"answer": answer, "sources": sources}
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses RAG Chat: {err_msg}"
        )

@app.delete("/ai/reports/{report_id}")
async def delete_report_vectors(report_id: str):
    try:
        deleted_count = 0
        try:
            collection.delete(where={"report_id": report_id})
        except Exception:
            pass
            
        try:
            res = collection.get()
            matching_ids = [cid for cid in res.get("ids", []) if report_id in cid]
            if matching_ids:
                collection.delete(ids=matching_ids)
                deleted_count = len(matching_ids)
        except Exception:
            pass
            
        return {"success": True, "message": f"Vektor report {report_id} berhasil dihapus ({deleted_count} chunks dibersihkan)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/stats")
async def get_ai_stats():
    return {
        "status": "online",
        "provider": "Google Gemini",
        "gemini_ready": gemini_client is not None,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "total_vectors": collection.count(),
        "collection_name": COLLECTION_NAME
    }

@app.get("/api")
@app.get("/api/info")
async def get_system_info():
    return {
        "status": "online",
        "version": "1.0.0",
        "llm": {
            "provider": "Google Gemini",
            "model": LLM_MODEL,
            "ready": gemini_client is not None
        },
        "embedding": {
            "provider": "Google Gemini",
            "model": EMBEDDING_MODEL,
            "dimension": 768
        },
        "chroma": {
            "collection": COLLECTION_NAME,
            "count": collection.count(),
            "storage_path": str(CHROMA_PATH)
        }
    }

@app.get("/chroma/info")
async def get_chroma_info():
    return {
        "database": "ChromaDB (Local Persistent)",
        "collection": COLLECTION_NAME,
        "documents": collection.count(),
        "storage_path": str(CHROMA_PATH),
        "embedding_provider": "Google Gemini",
        "embedding_model": EMBEDDING_MODEL,
        "expected_embedding_dimension": 768,
        "llm_provider": "Google Gemini",
        "llm_model": LLM_MODEL
    }

@app.delete("/chroma/reset")
async def reset_chroma():
    global collection
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        return {"success": True, "message": "ChromaDB berhasil di-reset. Semua dokumen vektor telah dikosongkan."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_all_documents():
    try:
        res = collection.get()
        return {
            "count": len(res.get("ids", [])),
            "ids": res.get("ids", []),
            "documents": res.get("documents", []),
            "metadatas": res.get("metadatas", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{source}")
async def delete_document_by_source(source: str):
    try:
        collection.delete(where={"source": source})
        return {"success": True, "message": f"Dokumen '{source}' berhasil dihapus dari vector store."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_pdf_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file tidak ditemukan.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang didukung.")
    
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File PDF kosong.")
    
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for idx, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(f"[Halaman {idx}]\n{t.strip()}")
        
        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="Tidak ada teks yang dapat diekstrak dari PDF.")
        
        # Simple chunking by paragraph (~500-1000 chars)
        raw_paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 30]
        if not raw_paragraphs:
            raw_paragraphs = [full_text[:1000]]
            
        chunks = []
        ids = []
        metadatas = []
        
        base_name = Path(file.filename).stem
        for c_idx, chunk in enumerate(raw_paragraphs):
            c_id = f"pdf_{base_name}_{c_idx}"
            chunks.append(chunk)
            ids.append(c_id)
            metadatas.append({
                "source": file.filename,
                "document_title": file.filename,
                "section": f"Bagian {c_idx+1}",
                "page": c_idx + 1
            })
            
        embeddings = []
        for c in chunks:
            emb = await get_embedding(c)
            embeddings.append(emb)
            
        valid_embeddings = [e for e in embeddings if e is not None]
        if len(valid_embeddings) == len(chunks):
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
            
        return {
            "success": True,
            "message": "Upload & proses PDF berhasil!",
            "filename": file.filename,
            "pages": len(reader.pages),
            "characters": len(full_text),
            "chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses PDF: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)
