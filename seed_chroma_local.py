import pyodbc
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
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
           uraian, pemeriksaan, tindakan, hasil, rekomendasi, catatan
    FROM dbo.reports
""")
rows = cur.fetchall()
print(f"Membaca {len(rows)} laporan...")

total_chunks = 0
for idx, r in enumerate(rows, start=1):
    report_id, title, doc_num, doc_dt, doc_type, status, officer, location, uraian, pemeriksaan, tindakan, hasil, rekomendasi, catatan = r

    try:
        collection.delete(where={"report_id": report_id})
    except Exception:
        pass

    chunks = []
    h_text = f"Judul Laporan: {title}\nNomor Dokumen: {doc_num}\nTanggal: {doc_dt}\nJenis: {doc_type}\nStatus: {status}\nPetugas: {officer}"
    if location:
        h_text += f"\nLokasi: {location}"
    chunks.append(("chunk_0", h_text, "identitas_dokumen"))

    sections = [
        ("chunk_1", f"Dokumen: {title}\n1. Uraian:\n{uraian}", "uraian") if uraian else None,
        ("chunk_2", f"Dokumen: {title}\n2. Pemeriksaan:\n{pemeriksaan}", "pemeriksaan") if pemeriksaan else None,
        ("chunk_3", f"Dokumen: {title}\n3. Tindakan:\n{tindakan}", "tindakan") if tindakan else None,
        ("chunk_4", f"Dokumen: {title}\n4. Hasil:\n{hasil}", "hasil") if hasil else None,
        ("chunk_5", f"Dokumen: {title}\n5. Rekomendasi:\n{rekomendasi}", "rekomendasi") if rekomendasi else None,
        ("chunk_6", f"Dokumen: {title}\n6. Catatan:\n{catatan}", "catatan") if catatan else None,
    ]

    for s in sections:
        if s:
            chunks.append(s)

    c_ids = [f"{report_id}_{c[0]}" for c in chunks]
    c_docs = [c[1] for c in chunks]
    c_metas = [{
        "report_id": report_id,
        "document_number": doc_num,
        "title": title,
        "section": c[2],
        "officer": officer,
        "status": status
    } for c in chunks]

    collection.upsert(ids=c_ids, documents=c_docs, metadatas=c_metas)
    total_chunks += len(c_ids)

conn.close()
print(f"Berhasil mengindeks {total_chunks} chunks dari {len(rows)} dokumen ke ChromaDB!")
print(f"Total item di ChromaDB sekarang: {collection.count()}")
