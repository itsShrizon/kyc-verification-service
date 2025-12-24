# IdentityGuard – KYC & Identity Verification API

FastAPI-based backend for OCR, face verification, and liveness detection with Java client support.

---

## Requirements

* Docker **or**
* Python 3.9+
* Tesseract OCR
* Java 11+ (optional client)

---

## Run with Docker (Recommended)

```bash
docker build -t identityguard .
docker run -p 8000:8000 identityguard
```

API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Run Locally (Without Docker)

```bash
git clone https://github.com/YOUR_USERNAME/IdentityGuard-KYC.git
cd IdentityGuard-KYC
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set Tesseract path if needed:

```bash
setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## API Endpoints

| Method | Endpoint          |
| ------ | ----------------- |
| POST   | /extract-id-data/ |
| POST   | /verify-identity/ |
| POST   | /check-liveness/  |

---

## Java Client

```bash
cd java-client
javac IdentityClient.java
java IdentityClient
```

---

## Notes

* OCR uses OpenCV + Tesseract
* Face matching uses deep embeddings
* Docker image includes all dependencies

---

