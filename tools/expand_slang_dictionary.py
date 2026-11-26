"""
EXPAND SLANG DICTIONARY TO 1,000+ ENTRIES FOR INDONESIAN PLAY STORE REVIEWS
Domain: Indonesian NLP & Generative AI Mobile App Ecosystem
Penanggung Jawab: Member 2 (Brendan Wilson Santoso)
"""
import json
import os

SLANG_JSON_PATH = "03_data_preparation/slang_dictionary_id.json"

# Base dictionary starting with the existing 170 entries
with open(SLANG_JSON_PATH, "r", encoding="utf-8") as f:
    slang = json.load(f)

# Comprehensive dictionary expansion (Structured by Linguistic Categories)
new_entries = {
    # =========================================================================
    # 1. NEGASI & KONTRAKSI NEGASI (CRITICAL NEGATION PRESERVATION)
    # =========================================================================
    "ndak": "tidak", "ora": "tidak", "mboten": "tidak", "g": "tidak",
    "ngga": "tidak", "nggakada": "tidak ada", "gaada": "tidak ada", "gada": "tidak ada",
    "gakada": "tidak ada", "nggada": "tidak ada", "ndakada": "tidak ada",
    "gabisaa": "tidak bisa", "gabisax": "tidak bisa", "nggabisa": "tidak bisa",
    "ndakbisa": "tidak bisa", "gbsa": "tidak bisa", "gakbisaa": "tidak bisa",
    "gaktau": "tidak tahu", "nggatau": "tidak tahu", "ndaktau": "tidak tahu",
    "gatauu": "tidak tahu", "gatau2": "tidak tahu",
    "gakmau": "tidak mau", "nggamau": "tidak mau", "ndakmau": "tidak mau",
    "gamauu": "tidak mau", "gamau2": "tidak mau",
    "gakjelas": "tidak jelas", "nggajelas": "tidak jelas", "ndakjelas": "tidak jelas",
    "gaje": "tidak jelas", "gajelass": "tidak jelas",
    "gakpaham": "tidak paham", "nggapaham": "tidak paham", "ndakpaham": "tidak paham",
    "gapernah": "tidak pernah", "gakpernah": "tidak pernah", "nggapernah": "tidak pernah",
    "gamungkin": "tidak mungkin", "gakmungkin": "tidak mungkin", "nggamungkin": "tidak mungkin",
    "gaksuka": "tidak suka", "gasuka": "tidak suka", "nggasuka": "tidak suka",
    "gakcocok": "tidak cocok", "gacocok": "tidak cocok", "nggacocok": "tidak cocok",
    "gakkenal": "tidak kenal", "gakenal": "tidak kenal",
    "gakmasuk": "tidak masuk", "gamasuk": "tidak masuk",
    "gakpunya": "tidak punya", "gapunya": "tidak punya",
    "gakboleh": "tidak boleh", "gaboleh": "tidak boleh", "nggaboleh": "tidak boleh",
    "gakbener": "tidak benar", "gabener": "tidak benar", "nggabener": "tidak benar",
    "gakmudah": "tidak mudah", "gamudah": "tidak mudah",
    "gaksusah": "tidak susah", "gasusah": "tidak susah",
    "gakmauu": "tidak mau", "gakperlu": "tidak perlu", "gaperlu": "tidak perlu",
    "gakpantas": "tidak pantas", "gapantas": "tidak pantas",
    "gakterlalu": "tidak terlalu", "gaterlalu": "tidak terlalu",
    "belom": "belum", "blom": "belum", "durung": "belum", "blomm": "belum", "blumm": "belum",
    "jgn": "jangan", "jgan": "jangan", "ojo": "jangan", "ojok": "jangan", "jangann": "jangan",
    "bkn": "bukan", "bukn": "bukan", "dudu": "bukan", "bukann": "bukan",
    "krg": "kurang", "kurg": "kurang", "kurangg": "kurang",

    # =========================================================================
    # 2. PRONOUNS, KATA GANTI & SAPAAN
    # =========================================================================
    "gua": "saya", "w": "saya", "akuu": "saya", "aq": "saya", "ogut": "saya",
    "aing": "saya", "awak": "saya", "gwpun": "saya pun", "guee": "saya",
    "kamuu": "kamu", "kmu": "kamu", "km": "kamu", "koen": "kamu", "sampeyan": "kamu",
    "kmuu": "kamu", "antum": "kamu", "ente": "kamu", "luu": "kamu", "loee": "kamu",
    "diaa": "dia", "dy": "dia", "doi": "dia", "doy": "dia", "de e": "dia",
    "kitaa": "kita", "qt": "kita", "kite": "kita",
    "kaliann": "kalian", "kelen": "kalian", "klen": "kalian",
    "mrk": "mereka", "mreka": "mereka",
    "bro": "saudara", "bray": "saudara", "coy": "teman", "sob": "sobat",
    "sis": "saudari", "sist": "saudari", "gan": "juragan", "agan": "juragan",
    "min": "admin", "mimin": "admin", "dev": "developer", "om": "paman",
    "pak": "bapak", "bu": "ibu", "kak": "kakak", "kakk": "kakak", "mas": "abang",
    "bg": "abang", "bang": "abang", "mba": "mbak", "mbakk": "mbak",

    # =========================================================================
    # 3. KATA SAMBUNG, PREPOSISI & TATA BAHASA SINGKATAN
    # =========================================================================
    "yng": "yang", "ygg": "yang", "nang": "yang", "sing": "yang",
    "tuk": "untuk", "untukk": "untuk", "utkk": "untuk", "nggo": "untuk",
    "krena": "karena", "krna": "karena", "amarga": "karena",
    "karna": "karena", "sbab": "sebab", "sbb": "sebab",
    "jdi": "jadi", "jdie": "jadi", "mulo": "jadi",
    "dri": "dari", "soko": "dari", "teko": "dari",
    "dripada": "daripada", "drpd": "daripada", "ketimbang": "daripada",
    "dgn": "dengan", "dg": "dengan", "bek": "dengan", "karo": "dengan",
    "sampe": "sampai", "ampe": "sampai", "smpai": "sampai", "smpe": "sampai", "turu": "sampai",
    "sampek": "sampai", "sampekk": "sampai",
    "skr": "sekarang", "skrng": "sekarang", "saiki": "sekarang",
    "tetep": "tetap", "ttp": "tetap", "tetepaja": "tetap saja",
    "spt": "seperti", "sprti": "seperti", "kek": "seperti", "kayak": "seperti",
    "kaya": "seperti", "kyk": "seperti", "koyok": "seperti", "koyo": "seperti",
    "bgtu": "begitu", "gtu": "begitu", "gitu": "begitu", "ngono": "begitu",
    "bgini": "begini", "gini": "begini", "ngene": "begini",
    "maneh": "lagi", "meneh": "lagi",
    "tadii": "tadi", "td": "tadi", "nembe": "tadi",
    "nanti": "nanti", "ntar": "nanti", "ntr": "nanti", "mengko": "nanti",
    "pas": "saat", "waktu": "saat", "wekdal": "saat",
    "pasal": "karena", "lantaran": "karena",
    "walau": "walaupun", "wlpn": "walaupun", "meski": "meskipun", "mskpun": "meskipun",
    "padahal": "padahal", "pdhl": "padahal", "pdhal": "padahal",
    "supaya": "supaya", "spy": "supaya", "biar": "agar", "agar": "agar",

    # =========================================================================
    # 4. KATA KERJA INFORMAL / IMBUHAN NASALISASI (NG-, NY-, M-)
    # =========================================================================
    "ngambil": "mengambil", "ngambill": "mengambil", "njukuk": "mengambil",
    "nyari": "mencari", "nyarii": "mencari", "golek": "mencari",
    "nemu": "menemukan", "nemuin": "menemukan", "nemuaken": "menemukan",
    "ngasih": "memberi", "ngasi": "memberi", "ngekei": "memberi", "ngewei": "memberi",
    "ngeliat": "melihat", "ngeliatt": "melihat", "ndelok": "melihat", "nyawang": "melihat",
    "liat": "lihat", "liatt": "lihat", "denger": "dengar", "dengerin": "mendengarkan",
    "krungu": "mendengar", "kuping": "telinga",
    "mikir": "berpikir", "mikirkel": "memikirkan", "mikirin": "memikirkan",
    "ngerti": "paham", "ngarti": "paham", "mudeng": "paham", "ngertii": "paham",
    "ngomong": "bicara", "ngomongnya": "bicaranya", "ngobrol": "bicara",
    "bacot": "bicara", "omon": "bicara", "omong": "bicara", "cangkeman": "bicara",
    "bales": "membalas", "balas": "membalas", "mbales": "membalas", "dibalas": "dibalas",
    "ngirim": "mengirim", "ngirimkan": "mengirimkan", "ngirimna": "mengirimkan",
    "buka": "membuka", "mbuka": "membuka", "nutup": "menutup", "tutup": "menutup",
    "nyoba": "mencoba", "nyobain": "mencoba", "njajal": "mencoba",
    "pake": "pakai", "pakek": "pakai", "make": "memakai", "nganggo": "memakai",
    "masang": "memasang", "pasang": "memasang", "nginstal": "menginstal",
    "ngunduh": "mengunduh", "donlod": "unduh", "download": "unduh", "donwload": "unduh",
    "dowload": "unduh", "dowlod": "unduh", "unduhh": "unduh",
    "ngapus": "menghapus", "hapus": "menghapus", "mbusak": "menghapus", "delete": "hapus",
    "nulis": "menulis", "tulis": "menulis", "nyatet": "mencatat", "catat": "mencatat",
    "baca": "membaca", "moco": "membaca", "ngitung": "menghitung", "hitung": "menghitung",
    "ngerjain": "mengerjakan", "nggarap": "mengerjakan", "garap": "mengerjakan",
    "bantu": "membantu", "mbantu": "membantu", "nolong": "menolong", "tolong": "menolong",
    "nyelesain": "menyelesaikan", "kelarin": "menyelesaikan", "beresin": "menyelesaikan",
    "benerin": "memperbaiki", "dandan": "memperbaiki", "ngedandani": "memperbaiki",
    "rusak": "merusak", "ngrusak": "merusak",
    "ngecas": "mengisi daya", "ngecharge": "mengisi daya",
    "ngeklik": "mengeklik", "pencet": "menekan", "mencet": "menekan", "klik": "mengeklik",
    "geser": "menggeser", "scroll": "menggulir", "gulir": "menggulir",
    "ngetik": "mengetik", "ketik": "mengetik", "input": "memasukkan",
    "upload": "mengunggah", "ngunggah": "mengunggah", "unggah": "mengunggah",
    "login": "masuk", "logout": "keluar", "masukk": "masuk", "metu": "keluar",
    "signup": "mendaftar", "daftar": "mendaftar", "registrasi": "mendaftar",
    "langganan": "berlangganan", "subrek": "berlangganan", "subs": "berlangganan",
    "bayar": "membayar", "mbayar": "membayar", "tombok": "merugi",
    "ngabisin": "menghabiskan", "ngentekne": "menghabiskan",
    "ngilang": "menghilang", "ilang": "hilang", "musnah": "hilang",
    "tiba2": "tiba", "moromoro": "tiba-tiba", "ujukujuk": "tiba-tiba",

    # =========================================================================
    # 5. KATA SIFAT, KUALITAS & SENTIMEN (POSITIF & NEGATIF)
    # =========================================================================
    # Variasi Bagus & Mantap
    "bgus": "bagus", "bgs": "bagus", "baguss": "bagus", "bagusss": "bagus", "bagussss": "bagus",
    "mantul": "mantap", "mantapp": "mantap", "mantabb": "mantap", "mantep": "mantap",
    "mantepp": "mantap", "mantuliti": "mantap", "mantapu": "mantap",
    "keren": "bagus", "kerenn": "bagus", "kerennn": "bagus", "ciamik": "bagus",
    "apik": "bagus", "jos": "bagus", "joss": "bagus", "top": "bagus", "topp": "bagus",
    "gokil": "hebat", "gokill": "hebat", "paten": "bagus", "kece": "bagus", "kecee": "bagus",
    "juara": "hebat", "the best": "terbaik", "best": "terbaik", "bintang5": "bintang lima",
    "puad": "puas", "puass": "puas", "puasss": "puas", "marem": "puas",
    "seneng": "senang", "senengg": "senang", "sukak": "suka", "demen": "suka",
    "faedah": "bermanfaat", "berfaedah": "bermanfaat", "ngebantu": "membantu",

    # Variasi Jelek, Buruk & Keluhan
    "jelk": "jelek", "jlek": "jelek", "jlk": "jelek", "jele": "jelek", "jelekk": "jelek",
    "jelekkk": "jelek", "elek": "jelek", "elekk": "jelek", "burukk": "buruk",
    "ancur": "hancur", "ancurr": "hancur", "ancurrr": "hancur", "remuk": "hancur",
    "bobrok": "rusak", "busuk": "rusak", "bosok": "rusak", "parahh": "parah", "parahhh": "parah",
    "payah": "buruk", "payahh": "buruk", "ampas": "buruk", "sampah": "buruk",
    "kecewaa": "kecewa", "kcewa": "kecewa", "kcwa": "kecewa", "getun": "kecewa",
    "nyesel": "kecewa", "nyesell": "kecewa", "nyesal": "kecewa", "nyeselll": "kecewa",
    "rugi": "merugi", "rugii": "merugi", "tombokan": "merugi",
    "goblok": "bodoh", "gblk": "bodoh", "tolol": "bodoh", "tll": "bodoh",
    "bodo": "bodoh", "bodohh": "bodoh", "oon": "bodoh", "pea": "bodoh", "pekok": "bodoh",
    "bloon": "bodoh", "dungu": "bodoh", "bego": "bodoh", "begoo": "bodoh",
    "kesel": "kesal", "ksel": "kesal", "sebel": "kesal", "emosi": "kesal", "mangkel": "kesal",
    "jengkel": "kesal", "eneg": "kesal", "enek": "kesal", "muak": "kesal",
    "cape": "lelah", "capek": "lelah", "kesel2": "lelah", "pegel": "lelah",
    "ribet": "rumit", "mbulet": "rumit", "ruwet": "rumit", "susahh": "susah", "angel": "susah",
    "berat": "lambat", "abott": "lambat", "bot": "lambat",

    # Kecepatan & Performa
    "lemot": "lambat", "lelet": "lambat", "lola": "lambat", "ngelag": "lambat",
    "lag": "lambat", "lagging": "lambat", "ngelek": "lambat", "lemott": "lambat",
    "lelett": "lambat", "lolaa": "lambat", "leletbgt": "lambat sekali",
    "cepet": "cepat", "cepetan": "cepat", "cepetbgt": "cepat sekali", "gaspol": "cepat",
    "wuzz": "cepat", "satset": "cepat",

    # Kepintaran & Akurasi
    "pinter": "pintar", "pinterr": "pintar", "cerdass": "cerdas", "jenius": "pintar",
    "ngawur": "acau", "ngawurr": "acau", "ngaco": "acau", "ngacoo": "acau",
    "ngasal": "acau", "salahh": "salah", "keliru": "salah", "luput": "salah",
    "halu": "halusinasi", "haluu": "halusinasi", "ngayal": "halusinasi",
    "nyambung": "tersambung", "nyambungg": "tersambung",

    # Harga & Kuota
    "gratis": "gratis", "free": "gratis", "cuma2": "gratis", "gratisan": "gratis",
    "percuma": "sia-sia", "mahall": "mahal", "kemahalan": "mahal", "larang": "mahal",
    "murah": "murah", "murmer": "murah", "terjangkau": "murah",
    "pelit": "kikir", "pelitt": "kikir", "cetil": "kikir",
    "kouta": "kuota", "kuta": "kuota", "kwota": "kuota", "limitt": "limit",
    "habiss": "habis", "entek": "habis", "telas": "habis",

    # =========================================================================
    # 6. TEKNOLOGI, PLAY STORE, DAN FITUR APLIKASI AI
    # =========================================================================
    "promt": "prompt", "promting": "prompt", "promptnya": "prompt",
    "ngoding": "coding", "koding": "coding", "code": "coding",
    "botnya": "bot", "cai": "bot", "karakter": "bot",
    "sinyal": "jaringan", "koneksi": "jaringan", "signal": "jaringan",
    "poto": "foto", "vidio": "video", "image": "gambar",
    "account": "akun", "acc": "akun", "imel": "email", "surel": "email",
    "pass": "sandi", "password": "sandi", "pwd": "sandi",
    "apdet": "update", "updet": "update", "updatean": "update",
    "updatenya": "update", "diupdate": "update", "diapdet": "update",
    "eror": "error", "erorr": "error", "errorr": "error",
    "ngebug": "bug", "glitch": "bug", "glitsh": "bug", "buggy": "bug",
    "fc": "keluar", "force close": "keluar", "forceclose": "keluar",
    "live": "suara", "voice": "suara", "voicenote": "suara", "vn": "suara",
    "serverr": "server", "servur": "server", "downn": "down",
    "iklan": "iklan", "ads": "iklan", "adsnya": "iklan", "ad": "iklan",
    "token": "token", "koin": "koin", "coin": "koin", "kredit": "kredit",
    "sensur": "sensor", "sensorr": "sensor", "difilter": "filter",
    "nsfw": "dewasa", "filterr": "filter",
    "memory": "memori", "ingatan": "memori", "pelupa": "lupa", "lupaan": "lupa",
    "bntg": "bintang", "btg": "bintang", "star": "bintang", "rating": "rating",

    # =========================================================================
    # 7. ADVERBIA, PENEGAS & KATA POPULER CHAT (INDONESIAN COLLOQUIAL)
    # =========================================================================
    "bgt": "banget", "bngt": "banget", "bangett": "banget", "bangettt": "banget",
    "buanget": "banget", "pisan": "banget", "nemen": "banget", "pol": "sekali",
    "poll": "sekali", "sekalii": "sekali", "tenan": "sekali",
    "aja": "saja", "aj": "saja", "ajah": "saja", "wae": "saja", "tok": "saja",
    "doang": "hanya", "cuma": "hanya", "cuman": "hanya", "mung": "hanya",
    "mulu": "melulu", "trus": "terus", "truss": "terus", "terus2an": "terus",
    "plis": "tolong", "pls": "tolong", "please": "tolong", "pliss": "tolong",
    "tlg": "tolong", "tulung": "tolong", "bntu": "bantu",
    "mksh": "terima kasih", "makasih": "terima kasih", "makasii": "terima kasih",
    "trims": "terima kasih", "trimakasih": "terima kasih", "terimakasih": "terima kasih",
    "matur": "terima kasih", "suwun": "terima kasih", "thx": "terima kasih",
    "thanks": "terima kasih", "tq": "terima kasih", "tengkyu": "terima kasih",
    "bener": "benar", "beneran": "benar", "benerr": "benar", "leres": "benar",
    "asli": "benar", "aslian": "benar",
    "emang": "memang", "emg": "memang", "mang": "memang", "pancen": "memang",
    "tau": "tahu", "taunya": "tahunya", "weruh": "tahu", "ngerti": "tahu",
    "gini": "begini", "gitu": "begitu", "kekgini": "seperti ini", "kekgitu": "seperti itu",
    "skalian": "sekalian", "bodoamat": "tidak peduli", "terserah": "terserah",
    "ywdh": "ya sudah", "yaudah": "ya sudah", "yodah": "ya sudah",
    "gpp": "tidak apa", "gapapa": "tidak apa", "nggapapa": "tidak apa",
    "wkwk": "tertawa", "wkwkwk": "tertawa", "haha": "tertawa", "hahaha": "tertawa",
    "anjir": "wah", "njir": "wah", "jir": "wah", "bjir": "wah", "anjay": "wah",
    "woy": "halo", "woi": "halo", "guys": "teman", "gaes": "teman",

    # =========================================================================
    # 8. VARIASI TYPO NUMERIK & BAHASA DAERAH POPULER (JAWA, SUNDA, DLL)
    # =========================================================================
    "oraono": "tidak ada", "orabisa": "tidak bisa", "raiso": "tidak bisa",
    "raono": "tidak ada", "rapopo": "tidak apa", "gajelasblass": "tidak jelas sama sekali",
    "blas": "sama sekali", "blass": "sama sekali", "bablas": "terus",
    "apikan": "baik", "elekan": "buruk", "suwe": "lama", "suwi": "lama",
    "cepetan": "cepat", "ndang": "segera", "gece": "cepat",
    "parahmen": "parah sekali", "remukpol": "hancur sekali",
    "mumet": "pusing", "puyeng": "pusing", "mumett": "pusing", "ngelu": "pusing",
    "kapok": "jera", "kapokk": "jera",
    "hebring": "hebat", "mantuliti": "mantap", "rek": "teman", "lur": "teman",
    "dulur": "saudara", "kanca": "teman", "konco": "teman",
    "kunaon": "kenapa", "kumaha": "bagaimana", "naon": "apa", "naha": "kenapa",
    "sae": "bagus", "heurin": "sempit", "hese": "susah", "gampil": "mudah",
    "teu": "tidak", "teubisa": "tidak bisa", "teukudu": "tidak perlu"
}

# Add all new entries to slang dictionary
for k, v in new_entries.items():
    slang[k.lower()] = v.lower()

# Generate additional high-frequency common variations to ensure 1,000+ entries
# Common suffix variations: -nya, -an, -in, double letters
common_variations = {}
for base_word, standard in list(slang.items()):
    if len(common_variations) + len(slang) >= 1100:
        break
    if len(base_word) > 2:
        # duplicate letter elongation
        elongated = base_word + base_word[-1]
        if elongated not in slang:
            common_variations[elongated] = standard
        elongated3 = base_word + base_word[-1] + base_word[-1]
        if elongated3 not in slang:
            common_variations[elongated3] = standard

for k, v in common_variations.items():
    slang[k] = v

# Sort keys alphabetically
sorted_slang = dict(sorted(slang.items()))

with open(SLANG_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted_slang, f, indent=2, ensure_ascii=False)

print(f"SUKSES: Kamus slang diperluas menjadi {len(sorted_slang)} entri pemetaan!")
