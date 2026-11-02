"""
Script untuk mempopulasi vektor awal ke ChromaDB dari database VPS 'sistem-knowladge'
Lengkap dengan lampiran foto (attachment_urls) dan file dokumen (pdf_url)
Jalankan: python seed_initial_vectors.py
"""
import os
import pyodbc
from google import genai
import chromadb
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=API_KEY)

CHROMA_PATH = BASE_DIR / "chroma_data"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="knowledge_documents", metadata={"hnsw:space": "cosine"})

VPS_HOST = '62.171.169.192,1433'
VPS_USER = 'sa'
VPS_PASS = 'Tijar200#MSSQL!'
DB_NAME = 'sistem-knowladge'

print("Menghubungkan ke database VPS [sistem-knowladge]...")
conn = pyodbc.connect(
    f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={VPS_HOST};DATABASE={DB_NAME};UID={VPS_USER};PWD={VPS_PASS}',
    timeout=15
)
cur = conn.cursor()

cur.execute("""
    SELECT report_id, title, document_number, document_datetime, document_type, status, officer, location,
           uraian, pemeriksaan, tindakan, hasil, rekomendasi, catatan, pdf_url
    FROM dbo.reports
""")
rows = cur.fetchall()
print(f"Ditemukan {len(rows)} laporan di database.")

# Ambil seluruh lampiran foto
cur.execute("""
    SELECT report_id, url
    FROM dbo.attachments
    ORDER BY sort_order, id
""")
att_map = {}
for rep_id, url in cur.fetchall():
    if rep_id not in att_map:
        att_map[rep_id] = []
    if url:
        att_map[rep_id].append(url)

print(f"Ditemukan lampiran foto untuk {len(att_map)} laporan.")

def get_embedding(text):
    res = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return res.embeddings[0].values

total_chunks = 0
for idx, r in enumerate(rows, start=1):
    report_id, title, doc_num, doc_dt, doc_type, status, officer, location, uraian, pemeriksaan, tindakan, hasil, rekomendasi, catatan, pdf_url = r
    print(f"[{idx}/{len(rows)}] Memproses: {title} ({doc_num})...")

    # Clear old chunks for this report
    try:
        collection.delete(where={"report_id": report_id})
    except Exception:
        pass

    att_list = att_map.get(report_id, [])
    att_urls_str = "|".join(att_list)
    doc_file_url = pdf_url if (pdf_url and pdf_url.strip()) else f"/documents/detail/{report_id}"

    chunks = []

    # Header Chunk (identitas_dokumen)
    h_text = f"Judul Laporan: {title}\nNomor Dokumen: {doc_num}\nTanggal: {doc_dt}\nJenis: {doc_type}\nStatus: {status}\nPetugas: {officer}"
    if location:
        h_text += f"\nLokasi: {location}"
    if att_list:
        h_text += f"\nDokumentasi Foto Lampiran: Tersedia ({len(att_list)} foto dokumentasi terlampir)"
    h_text += f"\nBerkas Dokumen: {doc_file_url}"

    chunks.append(("chunk_0", h_text, "identitas_dokumen"))

    # Sections
    sections = [
        ("chunk_1", f"Dokumen: {title} ({doc_num})\n1. Uraian Masalah:\n{uraian}", "uraian") if uraian else None,
        ("chunk_2", f"Dokumen: {title} ({doc_num})\n2. Pemeriksaan Teknis:\n{pemeriksaan}", "pemeriksaan") if pemeriksaan else None,
        ("chunk_3", f"Dokumen: {title} ({doc_num})\n3. Tindakan Solusi:\n{tindakan}", "tindakan") if tindakan else None,
        ("chunk_4", f"Dokumen: {title} ({doc_num})\n4. Hasil Akhir:\n{hasil}", "hasil") if hasil else None,
        ("chunk_5", f"Dokumen: {title} ({doc_num})\n5. Rekomendasi Pencegahan:\n{rekomendasi}", "rekomendasi") if rekomendasi else None,
        ("chunk_6", f"Dokumen: {title} ({doc_num})\n6. Catatan Tambahan:\n{catatan}", "catatan") if catatan else None,
    ]

    for s in sections:
        if s:
            chunks.append(s)

    c_ids = []
    c_docs = []
    c_metas = []
    c_embeddings = []

    for c_id_suffix, text, sec_key in chunks:
        vec = get_embedding(text)
        c_ids.append(f"{report_id}_{c_id_suffix}")
        c_docs.append(text)
        c_metas.append({
            "report_id": report_id,
            "document_number": doc_num,
            "title": title,
            "section": sec_key,
            "officer": officer,
            "status": status,
            "pdf_url": doc_file_url,
            "attachment_urls": att_urls_str
        })
        c_embeddings.append(vec)

    collection.upsert(ids=c_ids, documents=c_docs, metadatas=c_metas, embeddings=c_embeddings)
    total_chunks += len(c_ids)

conn.close()
print(f"\nSelesai! Total {total_chunks} chunk vektor tersimpan di ChromaDB.")
print(f"Total vektor di koleksi saat ini: {collection.count()}")
