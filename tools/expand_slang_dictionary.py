"""
EXPAND SLANG DICTIONARY FOR INDONESIAN PLAY STORE & GENERATIVE AI REVIEWS
Domain: Indonesian Colloquial NLP, Regional Slang, Chat Abbreviations & Mobile GenAI Ecosystem
Penanggung Jawab: Brendan Wilson Santoso (682024016)
"""

import json
import os
import re

SLANG_JSON_PATH = "03_data_preparation/slang_dictionary_id.json"

with open(SLANG_JSON_PATH, "r", encoding="utf-8") as f:
    slang = json.load(f)

# Kata-kata baku / fitur penting yang TIDAK BOLEH ditimpa jika belum ada di kamus awal
PROTECTED_STANDARD_KEYS = {
    "bagus", "baik", "mantap", "keren", "hebat", "cepat", "lambat", "mudah", "susah",
    "sulit", "murah", "mahal", "gratis", "bayar", "berbayar", "kuota", "limit", "batas",
    "akurat", "akurasi", "pintar", "cerdas", "paham", "mengerti", "jawaban", "solusi",
    "membantu", "bantu", "tugas", "skripsi", "kuliah", "belajar", "coding", "matematika",
    "halusinasi", "salah", "bodoh", "lupa", "sensor", "filter", "logika", "informasi",
    "server", "sibuk", "down", "error", "bug", "macet", "crash", "keluar", "login",
    "masuk", "akun", "update", "versi", "tampilan", "ui", "ux", "suara", "voice", "live",
    "jaringan", "koneksi", "ringan", "stabil", "verifikasi", "usia", "umur", "layar",
    "nyaman", "bahasa", "indonesia", "terjemahan", "baku", "lokal", "kaku", "inggris",
    "kalimat", "kata", "percakapan", "kecewa", "jelek", "buruk", "rusak", "parah",
    "menyesal", "kesal", "hancur", "rugi", "lelah", "bohong", "puas", "suka", "terbaik",
    "berguna", "bermanfaat", "canggih", "sempurna", "tidak", "bukan", "jangan", "kurang",
    "belum", "tapi", "namun", "karena", "untuk", "dengan", "dari", "pada", "dalam",
    "saya", "kamu", "kita", "mereka", "chatgpt", "gpt", "openai", "gemini", "deepseek",
    "claude", "anthropic", "perplexity", "copilot", "microsoft", "poe", "character",
    "ai", "prompt", "iklan", "token", "koin", "premium", "pro", "plus", "langganan",
    "best", "top", "nice", "good", "great", "love", "awesome", "helpful", "ok", "oke",
}

# =============================================================================
# 1. EKSPANSI KAMUS KURASI MANUAL LINTAS 12 KATEGORI LINGUISTIK INDONESIA
# =============================================================================
curated_expansion = {
    # --- A. Negasi, Kontraksi Negasi & Penolakan Lisan ---
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "kagak": "tidak", "kaga": "tidak", "kgk": "tidak", "kga": "tidak", "tdk": "tidak",
    "tak": "tidak", "ndak": "tidak", "nda": "tidak", "ndk": "tidak", "ora": "tidak",
    "ra": "tidak", "teu": "tidak", "henteu": "tidak", "indak": "tidak", "ndakkk": "tidak",
    "tarada": "tidak ada", "trada": "tidak ada", "tra": "tidak", "seng": "tidak",
    "sing": "yang", "katek": "tidak ada", "kadada": "tidak ada", "kdd": "tidak ada",
    "ndaada": "tidak ada", "oraono": "tidak ada", "raono": "tidak ada", "euweuh": "tidak ada",
    "zonk": "kecewa", "zonkk": "kecewa", "zonkkk": "kecewa",
    "gbs": "tidak bisa", "gbisa": "tidak bisa", "gabisa": "tidak bisa", "gakbisa": "tidak bisa",
    "nggakbisa": "tidak bisa", "takbisa": "tidak bisa", "rabisa": "tidak bisa", "raiso": "tidak bisa",
    "orabisa": "tidak bisa", "teubisa": "tidak bisa", "ndabisa": "tidak bisa", "kgkbisa": "tidak bisa",
    "kagakbisa": "tidak bisa", "gabisa2": "tidak bisa", "gakbisa2": "tidak bisa",
    "gamau": "tidak mau", "gakmau": "tidak mau", "nggakmau": "tidak mau", "ogah": "tidak mau",
    "wegah": "tidak mau", "embung": "tidak mau", "ndamau": "tidak mau", "kgkmau": "tidak mau",
    "gatau": "tidak tahu", "gaktau": "tidak tahu", "nggaktau": "tidak tahu", "gtau": "tidak tahu",
    "kurangtau": "kurang tahu", "entahlah": "tidak tahu", "ntahlah": "tidak tahu", "ntah": "entah",
    "gajelas": "tidak jelas", "gakjelas": "tidak jelas", "gaje": "tidak jelas", "gjls": "tidak jelas",
    "gj": "tidak jelas", "gajel": "tidak jelas", "ababil": "labil", "absurd": "aneh",
    "ganyambung": "tidak tersambung", "gaknyambung": "tidak tersambung", "nggaknyambung": "tidak tersambung",
    "gamasukakal": "tidak masuk akal", "gakmasukakal": "tidak masuk akal",
    "gapaham": "tidak paham", "gakpaham": "tidak paham", "gangerti": "tidak paham", "gakngerti": "tidak paham",
    "gagu": "kaku", "galogin": "gagal login", "gabuka": "tidak bisa dibuka", "gakbuka": "tidak bisa dibuka",
    "garespon": "tidak merespons", "gakrespon": "tidak merespons", "norespon": "tidak merespons",
    "gajalan": "tidak berjalan", "gakjalan": "tidak berjalan", "gamuncul": "tidak muncul", "gakmuncul": "tidak muncul",
    "gakeluar": "tidak keluar", "gakkeluar": "tidak keluar", "gakebuka": "tidak terbuka", "gakkebuka": "tidak terbuka",
    "gakebaca": "tidak terbaca", "gakkebaca": "tidak terbaca", "gakesimpan": "tidak tersimpan",
    "gakkesimpan": "tidak tersimpan", "gakekirim": "tidak terkirim", "gakkekirim": "tidak terkirim",
    "gadipake": "tidak dipakai", "gakdipake": "tidak dipakai", "gakepake": "tidak terpakai",
    "gaketemu": "tidak ketemu", "gakketemu": "tidak ketemu", "galaku": "tidak laku",
    "gasuka": "tidak suka", "gaksuka": "tidak suka", "nggaksuka": "tidak suka",
    "gapuas": "tidak puas", "gakpuas": "tidak puas", "nggakpuas": "tidak puas",
    "gaakurat": "tidak akurat", "gakakurat": "tidak akurat", "nggakakurat": "tidak akurat",
    "gaberguna": "tidak berguna", "gakberguna": "tidak berguna", "gaguna": "tidak berguna", "gakguna": "tidak berguna",
    "unfaedah": "tidak bermanfaat", "nirfaedah": "tidak bermanfaat", "sia2": "sia-sia", "percuma": "sia-sia",
    "blm": "belum", "blom": "belum", "belom": "belum", "blum": "belum", "lom": "belum",
    "durung": "belum", "dereng": "belum", "acan": "belum", "tacan": "belum",
    "jgn": "jangan", "jgan": "jangan", "jangn": "jangan", "ojo": "jangan", "ojok": "jangan", "ulah": "jangan",
    "bkn": "bukan", "bukn": "bukan", "bkan": "bukan", "dudu": "bukan", "sanes": "bukan", "lainnya": "lainnya",
    "krg": "kurang", "kurg": "kurang", "krng": "kurang", "kirang": "kurang",

    # --- B. Singkatan Chat & Konsonan Tanpa Vokal (Indonesian SMS/Play Store Style) ---
    "yg": "yang", "yng": "yang", "nyg": "yang", "utk": "untuk", "untk": "untuk", "tuk": "untuk",
    "dgn": "dengan", "dg": "dengan", "sm": "sama", "sma": "sama", "ama": "sama",
    "krn": "karena", "krna": "karena", "karna": "karena", "karen": "karena", " gara2": "karena", "gara2": "karena",
    "gr2": "karena", "gegara": "karena", "lantaran": "karena", "soalnya": "karena", "soalny": "karena",
    "pdhl": "padahal", "pdhal": "padahal", "padhal": "padahal", "walau": "walaupun", "wlp": "walaupun",
    "wlpn": "walaupun", "meski": "meskipun", "mskp": "meskipun", "mskpun": "meskipun", "biarpun": "walaupun",
    "tp": "tapi", "tpi": "tapi", "ttpi": "tetapi", "tetep": "tetap", "ttp": "tetap",
    "cm": "cuma", "cma": "cuma", "cmn": "cuma", "cuman": "cuma", "hny": "hanya", "doang": "hanya", "dng": "dong",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "uda": "sudah", "dah": "sudah", "dh": "sudah",
    "uwis": "sudah", "wis": "sudah", "wes": "sudah", "atos": "sudah", "tos": "sudah", "su": "sudah",
    "lg": "lagi", "lgi": "lagi", "lagii": "lagi", "maneh": "lagi", "meneh": "lagi", "deui": "lagi",
    "msh": "masih", "msih": "masih", "mash": "masih", "isih": "masih", "keneh": "masih",
    "skrg": "sekarang", "skrng": "sekarang", "skr": "sekarang", "skg": "sekarang", "saiki": "sekarang", "ayeuna": "sekarang",
    "dl": "dulu", "dlu": "dulu", "duluu": "dulu", "biyen": "dulu", "baheula": "dulu",
    "kmrn": "kemarin", "kmrin": "kemarin", "kemaren": "kemarin", "wingi": "kemarin", "kamari": "kemarin",
    "bsk": "besok", "besokk": "besok", "sesuk": "besok", "isukan": "besok",
    "td": "tadi", "tadii": "tadi", "barusan": "baru saja", "brsn": "baru saja",
    "ntr": "nanti", "ntar": "nanti", "entar": "nanti", "mengko": "nanti", "engke": "nanti",
    "kl": "kalau", "klo": "kalau", "klw": "kalau", "kalu": "kalau", "kaloo": "kalau", "misal": "kalau",
    "jk": "jika", "jikalau": "jika", "umpama": "jika", "semisal": "jika",
    "spy": "supaya", "spya": "supaya", "biar": "agar", "ben": "agar", "supados": "agar",
    "spt": "seperti", "sprti": "seperti", "sperti": "seperti", "kyk": "seperti", "kayak": "seperti",
    "kaya": "seperti", "kek": "seperti", "koyok": "seperti", "koyo": "seperti", "macem": "seperti", "mcm": "seperti",
    "bgtu": "begitu", "gtu": "begitu", "gitu": "begitu", "gt": "begitu", "ngono": "begitu", "kitu": "begitu",
    "bgini": "begini", "gini": "begini", "gni": "begini", "ngene": "begini", "kieu": "begini",
    "ni": "ini", "nihh": "ini", "ini2": "ini", "iki": "ini", "ieu": "ini",
    "tu": "itu", "tuhh": "itu", "iku": "itu", "eta": "itu",
    "dr": "dari", "dri": "dari", "drpd": "daripada", "dripada": "daripada", "timbang": "daripada",
    "kpd": "kepada", "pd": "pada", "dlm": "dalam", "antar": "antara",
    "stlh": "setelah", "stlah": "setelah", "shbs": "sehabis", "abis": "habis", "sehbs": "sehabis",
    "sblm": "sebelum", "sblum": "sebelum", "sblmnya": "sebelumnya", "slma": "selama", "slm": "selama",
    "smntra": "sementara", "smtra": "sementara", "sdg": "sedang", "sdng": "sedang",
    "bbrp": "beberapa", "bberapa": "beberapa", "byk": "banyak", "bnyk": "banyak", "banyakk": "banyak",
    "akeh": "banyak", "loba": "banyak", "seabrek": "banyak", "bejibun": "banyak", "segudang": "banyak",
    "dkt": "sedikit", "dikit": "sedikit", "sdkt": "sedikit", "sdikit": "sedikit", "sithik": "sedikit", "saeutik": "sedikit",
    "smua": "semua", "smuanya": "semuanya", "kabeh": "semua", "sadaya": "semua", "slrh": "seluruh",
    "tiap": "setiap", "stiap": "setiap", "saben": "setiap", "unggal": "setiap",
    "sllu": "selalu", "slalu": "selalu", "terus2": "terus", "terus2an": "terus", "mulu": "terus", "melulu": "terus",
    "kdng": "kadang", "kadang2": "kadang", "terkadang": "kadang", "srng": "sering", "sering2": "sering",
    "jrng": "jarang", "jarang2": "jarang", "palingan": "paling", "plg": "paling", "plng": "paling",
    "lbh": "lebih", "lbih": "lebih", "mkn": "makin", "smkn": "semakin", "kian": "semakin",
    "bgt": "banget", "bgtt": "banget", "bngt": "banget", "bngtt": "banget", "bnget": "banget",
    "bingit": "banget", "bingits": "banget", "buanget": "banget", "pisan": "banget", "pisann": "banget",
    "pol": "sekali", "poll": "sekali", "polll": "sekali", "skli": "sekali", "skali": "sekali", "tenan": "sekali",
    "bener": "benar", "bnr": "benar", "bner": "benar", "bener2": "benar-benar", "bnr2": "benar-benar",
    "sungguh2": "sungguh", "asli": "benar", "rill": "benar", "real": "benar", "valid": "benar", "no debat": "benar",
    "emg": "memang", "emang": "memang", "mmg": "memang", "memng": "memang",
    "mngkn": "mungkin", "mgkn": "mungkin", "mungkinn": "mungkin", "kayaknya": "sepertinya",
    "kyknya": "sepertinya", "keknya": "sepertinya", "kayanya": "sepertinya",
    "hrus": "harus", "hrs": "harus", "kudu": "harus", "wajib": "harus", "mesti": "harus", "msti": "harus",
    "prlu": "perlu", "perluu": "perlu", "bth": "butuh", "btuh": "butuh", "butuhh": "butuh",
    "bs": "bisa", "bsa": "bisa", "bisaa": "bisa", "iso": "bisa", "tiasa": "bisa", "sanggup": "bisa",
    "dpt": "dapat", "dapet": "dapat", "dpat": "dapat", "oleh": "dapat", "nemu": "dapat",
    "jd": "jadi", "jdi": "jadi", "jadinya": "jadinya", "jdinya": "jadinya", "malahh": "malah",
    "tau": "tahu", "tauu": "tahu", "tw": "tahu", "weruh": "tahu", "apal": "hafal",
    "pake": "pakai", "pk": "pakai", "pke": "pakai", "pakek": "pakai", "make": "pakai", "makek": "pakai",
    "gunain": "menggunakan", "pakein": "memakai", "nganggo": "memakai", "angge": "memakai",
    "coba": "coba", "cb": "coba", "cba": "coba", "cobain": "mencoba", "nyoba": "mencoba", "nyobain": "mencoba", "njajal": "mencoba",
    "kasih": "beri", "ksh": "beri", "ngasih": "memberi", "ngasi": "memberi", "dikasih": "diberi", "dikasi": "diberi",
    "minta": "meminta", "mnta": "meminta", "mnt": "meminta", "nyuwun": "meminta",
    "tlg": "tolong", "tlng": "tolong", "tolongg": "tolong", "tulung": "tolong", "plis": "tolong", "pls": "tolong", "pliss": "tolong",
    "mhn": "mohon", "mohonn": "mohon", "harap": "berharap", "smg": "semoga", "smoga": "semoga",
    "moga": "semoga", "moga2": "semoga", "mudah2an": "semoga", "mdh2an": "semoga",
    "mksh": "terima kasih", "makasih": "terima kasih", "mksih": "terima kasih", "makasii": "terima kasih",
    "trims": "terima kasih", "trimakasih": "terima kasih", "terimakasih": "terima kasih", "tks": "terima kasih",
    "thx": "terima kasih", "thanks": "terima kasih", "thankyou": "terima kasih", "tq": "terima kasih",
    "nuhun": "terima kasih", "haturnuhun": "terima kasih", "suwun": "terima kasih", "maturnuhun": "terima kasih",
    "matursuwun": "terima kasih", "makasi": "terima kasih", "syukron": "terima kasih",
    "knp": "kenapa", "knpa": "kenapa", "kenapaa": "kenapa", "ngapa": "kenapa", "napa": "kenapa",
    "kunaon": "kenapa", "naha": "kenapa", "nyapo": "kenapa", "ngopo": "kenapa", "kenape": "kenapa",
    "gmn": "bagaimana", "gmna": "bagaimana", "gimana": "bagaimana", "gimanaa": "bagaimana", "bgmn": "bagaimana",
    "piye": "bagaimana", "kepiye": "bagaimana", "kumaha": "bagaimana", "camana": "bagaimana", "cemana": "bagaimana",
    "kpn": "kapan", "kpnn": "kapan", "kapan2": "kapan", "iraha": "kapan",
    "dmn": "di mana", "dmna": "di mana", "dimana": "di mana", "dimna": "di mana", "kmn": "ke mana", "kemana": "ke mana",
    "brp": "berapa", "brpa": "berapa", "piro": "berapa", "sabaraha": "berapa",
    "apa2": "apa-apa", "apapun": "apa pun", "apaan": "apa", "apasi": "apa", "apasih": "apa", "naon": "apa", "opo": "apa",

    # --- C. Kata Ganti & Sapaan Komunitas Digital Indonesia ---
    "sy": "saya", "sya": "saya", "aq": "saya", "akuu": "saya", "ak": "saya", "gw": "saya", "gue": "saya",
    "gua": "saya", "gwe": "saya", "ane": "saya", "beta": "saya", "ambo": "saya", "awak": "saya",
    "aing": "saya", "abdi": "saya", "kulo": "saya", "dalem": "saya", "inyong": "saya", "nyong": "saya",
    "km": "kamu", "kmu": "kamu", "lu": "kamu", "lo": "kamu", "loe": "kamu", "elu": "kamu", "ente": "kamu",
    "antum": "kamu", "kau": "kamu", "ko": "kamu", "kowe": "kamu", "sampeyan": "kamu", "manehna": "dia",
    "klian": "kalian", "kelen": "kalian", "klean": "kalian", "mrk": "mereka", "mreka": "mereka",
    "kt": "kita", "kta": "kita", "kite": "kita", "torang": "kita", "kitorang": "kita", "katong": "kita",

    # --- D. Pujian, Kepuasan & Sentimen Positif Aplikasi Mobile ---
    "bgs": "bagus", "bgus": "bagus", "bguus": "bagus", "baguss": "bagus", "baguus": "bagus",
    "mantul": "mantap", "mntp": "mantap", "mantap2": "mantap", "mantep": "mantap", "mantab": "mantap",
    "ntap": "mantap", "ntaps": "mantap", "mantapu": "mantap", "mantapjiwa": "mantap", "mantuliti": "mantap",
    "gacor": "hebat", "gacorr": "hebat", "gg": "hebat", "ggwp": "hebat", "op": "hebat", "imba": "hebat",
    "josgandos": "bagus", "ciamik": "bagus", "apik": "bagus", "sae": "bagus", "rancak": "bagus", "elok": "bagus",
    "kece": "keren", "kren": "keren", "krenn": "keren", "kerenn": "keren", "cool": "keren", "cakep": "bagus",
    "juara": "terbaik", "juarak": "terbaik", "terbaek": "terbaik", "terbaiq": "terbaik", "terbaikk": "terbaik",
    "dabest": "terbaik", "thebest": "terbaik", "terdabest": "terbaik", "nomorsatu": "terbaik",
    "rekomen": "rekomendasi", "rekomend": "rekomendasi", "rekomended": "rekomendasi", "recommended": "rekomendasi",
    "reccomended": "rekomendasi", "recomend": "rekomendasi", "worthit": "bermanfaat", "worth": "bermanfaat",
    "berfaedah": "bermanfaat", "faedah": "bermanfaat", "manfaat": "bermanfaat", "brmnfaat": "bermanfaat",
    "ngebantu": "membantu", "mmbntu": "membantu", "mbantu": "membantu", "bntu": "membantu", "kebantu": "terbantu",
    "puass": "puas", "puasss": "puas", "memuaskn": "memuaskan", "mmuaskan": "memuaskan", "marem": "puas",
    "nyamann": "nyaman", "nymn": "nyaman", "pewe": "nyaman", "pw": "nyaman", "cozy": "nyaman", "enakeun": "enak",
    "enakk": "enak", "uenak": "enak", "wenak": "enak", "nikmat": "nyaman",
    "gampang": "mudah", "gmpng": "mudah", "gmpang": "mudah", "mdh": "mudah", "mudahh": "mudah",
    "gampil": "mudah", "gampangbgt": "mudah sekali", "simpel": "mudah", "simple": "mudah", "praktis": "mudah",
    "userfriendly": "mudah", "ramah": "nyaman", "intuitif": "mudah",
    "cpt": "cepat", "cpat": "cepat", "cepet": "cepat", "cepett": "cepat", "satset": "cepat",
    "gercep": "cepat", "gece": "cepat", "wuzz": "cepat", "ngebut": "cepat", "kenceng": "cepat", "kencang": "cepat",
    "lancar": "stabil", "lncr": "stabil", "lancarr": "stabil", "lancarjaya": "stabil", "mulus": "stabil",
    "smooth": "stabil", "enteng": "ringan", "ringann": "ringan", "anteng": "stabil",
    "pinter": "pintar", "pntr": "pintar", "pintarr": "pintar", "cerdass": "cerdas", "smart": "cerdas",
    "jenius": "cerdas", "genius": "cerdas", "jago": "pintar", "handal": "andal", "andal": "pintar",
    "akrt": "akurat", "akuratt": "akurat", "tepatt": "tepat", "presisi": "tepat", "valid": "akurat",
    "lengkap": "lengkap", "lngkp": "lengkap", "komplit": "lengkap", "komplet": "lengkap", "detail": "lengkap",
    "rinci": "lengkap", "jelass": "jelas", "jls": "jelas", "gamblang": "jelas", "rapi": "bagus", "rapih": "bagus",
    "luwes": "alami", "natural": "alami", "nyambung": "tersambung", "kontekstual": "tepat",
    "murmer": "murah", "mrh": "murah", "terjangkau": "murah", "hemat": "murah", "irit": "hemat",
    "grts": "gratis", "gratiss": "gratis", "gratisan": "gratis", "freebies": "gratis", "cuma2": "gratis",
    "lmyn": "lumayan", "lmyan": "lumayan", "lumayann": "lumayan", "ckp": "cukup", "cukupp": "cukup",
    "notbad": "lumayan", " mendingan": "lumayan", "mendingan": "lumayan", "bolehlah": "lumayan", "okelah": "oke",

    # --- E. Keluhan, Emosi Negatif & Masalah Teknis Aplikasi AI ---
    "jlk": "jelek", "jlek": "jelek", "jelk": "jelek", "jele": "jelek", "jelekk": "jelek", "elek": "jelek", "goreng": "jelek",
    "brk": "buruk", "bruuk": "buruk", "burukk": "buruk", "bapuk": "buruk", "bapukk": "buruk",
    "ampas": "sampah", "ampass": "sampah", "smph": "sampah", "sampahh": "sampah", "trash": "sampah",
    "bobrok": "rusak", "bosok": "rusak", "busuk": "rusak", "rsk": "rusak", "rusakk": "rusak",
    "ancur": "hancur", "hncr": "hancur", "ancurr": "hancur", "remuk": "hancur", "ambyar": "hancur", "kacau": "rusak",
    "prh": "parah", "prahh": "parah", "parahh": "parah", "kebangetan": "parah", "keterlaluan": "parah",
    "kcw": "kecewa", "kcewa": "kecewa", "kecewaa": "kecewa", "mengecewakn": "mengecewakan", "getun": "kecewa",
    "nyesel": "menyesal", "nyesal": "menyesal", "nysel": "menyesal", "nyesell": "menyesal", "kapok": "menyesal",
    "ksl": "kesal", "kesel": "kesal", "sebel": "kesal", "sbl": "kesal", "gedeg": "kesal", "gedek": "kesal",
    "mangkel": "kesal", "jengkel": "kesal", "dongkol": "kesal", "emosi": "kesal", "gregetan": "kesal",
    "muak": "kesal", "enek": "kesal", "eneg": "kesal", "ilfeel": "kecewa", "ilfil": "kecewa",
    "bosen": "bosan", "bsn": "bosan", "bt": "bosan", "bete": "bosan", "jenuh": "bosan",
    "cpk": "capek", "cape": "capek", "capekk": "capek", "pegel": "lelah", "lelahh": "lelah", "pusing": "bingung",
    "mumet": "bingung", "puyeng": "bingung", "bingungg": "bingung", "bngung": "bingung", "linglung": "bingung",
    "gblk": "goblok", "goblog": "goblok", "tll": "tolol", "tlol": "tolol", "bego": "bodoh", "begok": "bodoh",
    "oon": "bodoh", "bloon": "bodoh", "pekok": "bodoh", "dungu": "bodoh", "bodo": "bodoh", "bdh": "bodoh",
    "lemot": "lambat", "lmot": "lambat", "lemott": "lambat", "lelet": "lambat", "llet": "lambat",
    "lola": "lambat", "lamban": "lambat", "lmbt": "lambat", "lmbat": "lambat", "suwe": "lama", "suwi": "lama",
    "ngelag": "lambat", "nglag": "lambat", "lag": "lambat", "laggy": "lambat", "lagging": "lambat",
    "patah2": "lambat", "tersendat": "lambat", "sendat": "lambat", "muter": "loading", "muter2": "loading",
    "mutermuter": "loading", "loadingnya": "loading", "loding": "loading", "lod": "loading", "buffering": "loading",
    "macett": "macet", "mct": "macet", "nyangkut": "macet", "mentok": "macet", "stuck": "macet",
    "ngestuck": "macet", "freeze": "macet", "ngefreeze": "macet", "hang": "macet", "ngehang": "macet",
    "ngeblank": "macet", "blank": "layar", "layarhitam": "layar hitam", "blackscreen": "layar hitam",
    "fc": "keluar", "forceclose": "keluar", " mental": "keluar", "mental": "keluar", "ketutup": "keluar",
    "nutupsendiri": "keluar", "keluarsendiri": "keluar", "crashh": "crash", "kres": "crash",
    "eror": "error", "err": "error", "erorr": "error", "errorr": "error", "korslet": "error",
    "ngebug": "bug", "ngbug": "bug", "bugs": "bug", "bugnya": "bug", "glitch": "bug",
    "sibukk": "sibuk", "sbuk": "sibuk", "busy": "sibuk", "serverbusy": "server sibuk", "overload": "sibuk",
    "penuh": "sibuk", "antre": "sibuk", "antri": "sibuk", "down": "down", " tumbang": "down", "mati": "down",
    "ggl": "gagal", "gagall": "gagal", "fail": "gagal", "failed": "gagal", "mental2": "gagal",
    "loginn": "login", "loging": "login", "signin": "login", "sign": "login", "logout": "keluar",
    "verif": "verifikasi", "verivikasi": "verifikasi", "ferifikasi": "verifikasi", "verification": "verifikasi",
    "otp": "verifikasi", "captcha": "verifikasi", "ktp": "usia",
    "halu": "halusinasi", "haluu": "halusinasi", "ngaco": "ngawur", "ngacoo": "ngawur", "ngawurr": "ngawur",
    "ngelantur": "ngawur", "ngasal": "ngawur", "asal2an": "ngawur", "ngarang": "halusinasi", "sotoy": "ngawur",
    "pikun": "lupa", "pelupa": "lupa", "amnesia": "lupa", "lupaan": "lupa", "lupa2": "lupa",
    "bertele": "berbelit", "bertele2": "berbelit", "muter2jawab": "berbelit", "basi": "membosankan",
    "kakuu": "kaku", "robotik": "kaku", "baku2": "kaku", "aneh2": "aneh", "neh": "aneh",
    "sensur": "sensor", "disensor": "sensor", "nyensor": "sensor", "dikebiri": "dibatasi", "ketat": "dibatasi",
    "difilter": "filter", "filternya": "filter", "nsfw": "sensor",
    "mhl": "mahal", "mhal": "mahal", "mahall": "mahal", "kemahalan": "mahal", "mehong": "mahal", "pricey": "mahal",
    "byr": "bayar", "byar": "bayar", "mbayar": "bayar", "prabayar": "berbayar", "berbyr": "berbayar",
    "paywall": "berbayar", "komersil": "berbayar", "matre": "mahal", "maruk": "pelit", "rakus": "pelit",
    "pelitt": "pelit", "plit": "pelit", "kikir": "pelit", "medit": "pelit", "perhitungan": "pelit",
    "kouta": "kuota", "kwota": "kuota", "quota": "kuota", "qouta": "kuota", "kuotanya": "kuota",
    "lmt": "limit", "limid": "limit", "limitt": "limit", "limitnya": "limit", "dibatasi": "dibatasi",
    "dbatasi": "dibatasi", " dibatasin": "dibatasi", "dibatasin": "dibatasi", "batasin": "batas",
    "hbs": "habis", "hbis": "habis", "habiss": "habis", "abis": "habis", "abiss": "habis", "ludes": "habis",
    "subskrip": "langganan", "subscrib": "langganan", "subscribe": "langganan", "subs": "langganan",
    "lnggnn": "langganan", "langgan": "langganan", "berlanggan": "berlangganan", "ikln": "iklan", "iklannya": "iklan", "ads": "iklan",
    "ribett": "ribet", "rbet": "ribet", "ruwet": "ribet", "mbulet": "ribet", "belibet": "ribet", "rempong": "ribet",
    "ssh": "susah", "ssah": "susah", "susahh": "susah", "slt": "sulit", "slit": "sulit", "sulitt": "sulit", "angel": "susah", "hese": "susah",

    # --- F. Kata Kerja Percakapan Sehari-hari & Produktivitas Akademik/Kerja ---
    "apdet": "update", "updet": "update", "apdate": "update", "updt": "update", "pembaruan": "update",
    "diapdet": "update", "diupdet": "update", "apdetan": "update", "updetan": "update",
    "balikin": "kembalikan", "kembaliin": "kembalikan", "kmblkn": "kembalikan", "pulangin": "kembalikan",
    "benerin": "perbaiki", "bnerin": "perbaiki", "perbaikin": "perbaiki", "prbki": "perbaiki", "dibenerin": "diperbaiki",
    "fix": "perbaiki", "fixed": "diperbaiki", "difix": "diperbaiki",
    "donlod": "unduh", "donlot": "unduh", "dunlud": "unduh", "dwnld": "unduh", "download": "unduh", "didownload": "diunduh",
    "instal": "pasang", "install": "pasang", "nginstal": "pasang", "uninstal": "hapus", "uninstall": "hapus", "apus": "hapus", "ngapus": "hapus",
    "ngerjain": "mengerjakan", "nerjain": "mengerjakan", "kerjain": "mengerjakan", "garap": "mengerjakan", "nggarap": "mengerjakan",
    "nyelesain": "menyelesaikan", "selesain": "menyelesaikan", "kelarin": "menyelesaikan", "beresin": "menyelesaikan",
    "ngejelasin": "penjelasan", "jelasin": "penjelasan", "nerangin": "penjelasan", "jabarin": "penjelasan",
    "ngerangkum": "merangkum", "rangkumin": "merangkum", "ringkasin": "meringkas", "nyimpulin": "menyimpulkan",
    "nerjemahin": "terjemahan", "terjemahin": "terjemahan", "translate": "terjemahan", "translet": "terjemahan", "menterjemahkan": "terjemahan",
    "bhs": "bahasa", "bhsa": "bahasa", "bhasa": "bahasa", "indo": "indonesia", "ind": "indonesia", "ina": "indonesia",
    "ingris": "inggris", "inggriss": "inggris", "eng": "inggris", "english": "inggris",
    "matkul": "kuliah", "makul": "kuliah", "kampus": "kuliah", "mhs": "mahasiswa", "mhsw": "mahasiswa",
    "skripsian": "skripsi", "skripsweet": "skripsi", "tesis": "skripsi", "makalah": "tugas", "jurnal": "informasi",
    "pr": "tugas", "tgs": "tugas", "tugass": "tugas", "ujian": "tugas", "ulangan": "tugas", "kuis": "tugas",
    "krj": "kerja", "kerjaan": "pekerjaan", "pekerjaann": "pekerjaan", "kantor": "pekerjaan", "kantoran": "pekerjaan",
    "ngoding": "coding", "koding": "coding", "codingan": "coding", "kodingan": "coding", "program": "coding", "debugging": "coding",
    "mtk": "matematika", "matik": "matematika", "itung": "hitung", "ngitung": "hitung", "rumus": "matematika",
    "gmbr": "gambar", "gambarr": "gambar", "poto": "foto", "fto": "foto", "desain": "gambar", "lukis": "gambar",
    "jwb": "jawaban", "jwbn": "jawaban", "jawabn": "jawaban", "responn": "respon", "respons": "respon", "bls": "balas", "blsan": "balasan",
    "tny": "tanya", "tnya": "tanya", "nanya": "bertanya", "nanyain": "bertanya", "prtnyaan": "pertanyaan", "pertanyaannya": "pertanyaan",
    "obrol": "percakapan", "ngobrol": "ngobrol", "chating": "percakapan", "chatting": "percakapan", "curhat": "cerita", "cerita2": "cerita",
    "promt": "prompt", "promp": "prompt", "perintah": "prompt", "instruksi": "prompt",
    "fiturr": "fitur", "fiturnya": "fitur", "ftr": "fitur", "menu2": "tampilan", "tampilannya": "tampilan",
    "dev": "developer", "devs": "developer", "pengembang": "developer", "admin": "developer", "min": "developer", "mimin": "developer",
}

# Gabungkan kurasi manual secara aman (tanpa menimpa kata baku penting)
for k, v in curated_expansion.items():
    k_clean = k.strip().lower()
    v_clean = v.strip().lower()
    if k_clean and k_clean not in PROTECTED_STANDARD_KEYS and k_clean != v_clean:
        slang[k_clean] = v_clean

# =============================================================================
# 2. KOSAKATA PERCAKAPAN HARIAN, REGIONAL & KATA KERJA/SIFAT LISAN TAMBAHAN
# =============================================================================
colloquial_roots = {
    # Kata kerja lisan sehari-hari (-in / nge- / ng- / ny-)
    "ngajarin": "mengajari", "ajarin": "mengajari", "belajarin": "mempelajari",
    "ngingetin": "mengingatkan", "ingetin": "mengingatkan", "lupain": "melupakan",
    "nyariin": "mencari", "cariin": "mencari", "nemuin": "menemukan", "ketemuin": "menemukan",
    "nunjukin": "menunjukkan", "tunjukin": "menunjukkan", "liatin": "melihat", "ngeliatin": "melihat",
    "dengerin": "mendengarkan", "ngedengerin": "mendengarkan", "bacain": "membacakan", "nulisin": "menuliskan",
    "bikinin": "membuatkan", "buatin": "membuatkan", "ngebikin": "membuat", "bikin": "membuat",
    "ngeditin": "mengedit", "editin": "mengedit", "ngedit": "edit", "hapusin": "menghapus",
    "simpenin": "menyimpan", "simpen": "simpan", "nyimpen": "menyimpan",
    "kirimin": "mengirimkan", "ngirimin": "mengirimkan", "bukain": "membukakan", "nutupin": "menutupi",
    "masukin": "memasukkan", "keluarin": "mengeluarkan", "naikin": "menaikkan", "nurunin": "menurunkan",
    "tambahin": "menambahkan", "nambahin": "menambahkan", "kurangin": "mengurangi", "ngurangin": "mengurangi",
    "cepetin": "mempercepat", "lambatin": "memperlambat", "gede": "besar", "gedein": "memperbesar",
    "kecil": "kecil", "kecilin": "memperkecil", "panjangin": "memperpanjang", "pendekin": "memperpendek",
    "mudahin": "memudahkan", "nyusahin": "menyusahkan", "repotin": "merepotkan", "ngerepotin": "merepotkan",
    "bantuin": "membantu", "ngebantuin": "membantu", "tolongin": "menolong", "nolongin": "menolong",
    "biarin": "membiarkan", "ngebiarin": "membiarkan", "paksain": "memaksakan", "maksa": "memaksa",
    "hubungin": "menghubungi", "nyambungin": "menyambungkan", "sambungin": "menyambungkan",
    "aktifin": "mengaktifkan", "nyalain": "menyalakan", "matiin": "mematikan",
    "gantiin": "menggantikan", "ngganti": "mengganti", "tukerin": "menukar", "tuker": "tukar",
    "bayarin": "membayarkan", "gratisan": "gratis", "langganin": "berlangganan",
    "daftarnya": "pendaftaran", "daftarin": "mendaftarkan", "verifikasiin": "verifikasi",
    "ngerti": "paham", "ngertiin": "memahami", "pahamin": "memahami", "jelasinnya": "penjelasan",
    "jawabannyaa": "jawaban", "pertanyaannyaa": "pertanyaan", "solusinyaa": "solusi",
    "ngulang": "mengulang", "ulangin": "mengulang", "ulang2": "berulang", "bolakbalik": "berulang",
    "nunggu": "menunggu", "nungguin": "menunggu", "tungguin": "menunggu", "kelamaan": "lama",
    "kecepatan": "cepat", "kebanyakan": "banyak", "keseringan": "sering", "kekecilan": "kecil",
    "kebesaran": "besar", "kepanjangan": "panjang", "kependekan": "pendek",

    # Kosakata percakapan & ekspresi populer Nusantara
    "banget2": "banget", "beneran": "benar", "boongan": "bohong", "boong": "bohong", "hoax": "bohong",
    "tipu": "bohong", "nipu": "bohong", "penipu": "bohong", "php": "bohong", "palsu": "bohong",
    "jujurr": "jujur", "jjr": "jujur", "tulus": "jujur", "sumpah": "sungguh", "suer": "sungguh",
    "ciyus": "serius", "seriusan": "serius", "srius": "serius", "yakinn": "yakin", "ykin": "yakin",
    "ragu2": "ragu", "bingung2": "bingung", "takutnya": "takut", "khawatir": "cemas", "kuatir": "cemas",
    "seneng": "senang", "sneng": "senang", "happy": "senang", "hepi": "senang", "girang": "senang",
    "sedihh": "sedih", "sdih": "sedih", "nangis": "sedih", "mewek": "sedih", "galau": "sedih",
    "marahh": "marah", "mrah": "marah", "ngamuk": "marah", "murka": "marah", "sewot": "marah",
    "maluu": "malu", "sungkan": "malu", "segan": "malu", "takutt": "takut", "ngeri": "takut",
    "anehh": "aneh", "ganjil": "aneh", "janggal": "aneh", "freak": "aneh", "random": "acak",
    "biasa2": "biasa", "b aja": "biasa", "baja": "biasa", "standar": "biasa", "std": "biasa", "medioker": "biasa",
    "lumayanlah": "lumayan", "cukuplah": "cukup", "mayan": "lumayan", "myn": "lumayan",
    "pentingg": "penting", "pntg": "penting", "pnting": "penting", "krusial": "penting", "vital": "penting",
    "cocokk": "cocok", "cck": "cocok", "pas": "cocok", "sesuaii": "sesuai", "klop": "cocok",
    "mirip": "serupa", "persis": "sama", "plek": "sama", "beda": "berbeda", "bbrda": "berbeda",
    "lain2": "lain", "macam2": "beragam", "macem2": "beragam", "rupa2": "beragam", "berbagai": "beragam",
    "masing2": "masing-masing", "rata2": "rata-rata", "kira2": "kira-kira", "kuranglebih": "kira-kira",
    "pelan2": "pelan", "alon2": "pelan", "hati2": "hati-hati", "ati2": "hati-hati",
    "tiba2": "tiba-tiba", "mendadak": "tiba-tiba", "ujug2": "tiba-tiba", "tau2": "tiba-tiba",
    "lama2": "lama-kelamaan", "akhir2": "akhir-akhir", "belakangan": "akhir-akhir",
    "mula2": "awalnya", "awal2": "awalnya", "pertama2": "pertama", "ujung2nya": "akhirnya",
    "sama2": "sama-sama", "bareng2": "bersama", "bareng": "bersama", "rame2": "bersama", "ramai2": "bersama",
    "sendiri2": "sendiri", "sendirian": "sendiri", "dewekan": "sendiri", "sorangan": "sendiri",
    "temen": "teman", "tmn": "teman", "temen2": "teman", "tmn2": "teman", "bestie": "teman", "sohib": "teman",
    "ortu": "orang tua", "org": "orang", "orng": "orang", "orang2": "orang", "warga": "orang", "netizen": "pengguna",
    " bocil": "anak", "bocil": "anak", "bocah": "anak", "remaja": "anak", "dewasa": "usia",
    "duit": "uang", "doku": "uang", "fulus": "uang", "cuan": "untung", "kantong": "biaya", "dompet": "biaya",
    "tagihan": "biaya", "tarif": "harga", "hrg": "harga", "hrga": "harga", "biayanya": "biaya",
    "diskon": "murah", "promo": "murah", "voucher": "murah", "cashback": "murah", "trial": "gratis",
    "kuota2": "kuota", "paket": "langganan", "member": "langganan", "vip": "premium", "berbayarnya": "berbayar",
    "hp": "ponsel", "hape": "ponsel", "handphone": "ponsel", "smartphone": "ponsel", "android": "ponsel",
    "device": "perangkat", "gadget": "perangkat", "spek": "perangkat", "ram": "memori", "batre": "baterai", "baterai": "daya",
    "kuota data": "jaringan", "wifi": "jaringan", "sinyal": "jaringan", "inet": "internet", "net": "internet",
    "onlen": "online", "daring": "online", "oflen": "offline", "luring": "offline",
    "web": "situs", "website": "situs", "link": "tautan", "akunnya": "akun", "profil": "akun",
    "nomer": "nomor", "nmr": "nomor", "nohp": "nomor", "wa": "pesan", "imel": "email", "gmail": "email",
    "sandi": "password", "pin": "kode", "kodeotp": "verifikasi", "verifnya": "verifikasi",
    "suaranya": "suara", "audio": "suara", "mic": "suara", "mikrofon": "suara", "ngomong": "bicara",
    "bicaranya": "bicara", "logatnya": "logat", "aksennya": "aksen", "pelafalannya": "pelafalan",
    "tulisan": "kalimat", "tulisannya": "kalimat", "teksnya": "kalimat", "ketikan": "kalimat",
    "dokumen": "tugas", "file": "berkas", "pdfnya": "dokumen", "ppt": "presentasi", "makalahnya": "tugas",
    "skripsinya": "skripsi", "kodingannya": "coding", "script": "coding", "syntax": "coding", "algoritma": "logika",
    "penjelasannya": "penjelasan", "infonya": "informasi", "info": "informasi", "faktanya": "fakta", "datanya": "informasi",
    "idenya": "solusi", "saran": "solusi", "rekomendasinya": "rekomendasi", "masukannya": "solusi",
}

for k, v in colloquial_roots.items():
    k_clean = k.strip().lower()
    v_clean = v.strip().lower()
    if k_clean and k_clean not in PROTECTED_STANDARD_KEYS and k_clean != v_clean:
        slang[k_clean] = v_clean

# =============================================================================
# 3. PEMBANGKITAN SISTEMATIS VARIASI EJAAN CHAT PLAY STORE INDONESIA
#    (Tanpa Vokal, Akhiran -nya/-ny/-x, Huruf Ganda, & Typo Keyboard Umum)
# =============================================================================
# Daftar kata dasar domain ulasan aplikasi AI untuk pembangkitan variasi non-baku yang valid
domain_targets = [
    ("aplikasi", "aplikasi"), ("membantu", "membantu"), ("sekarang", "sekarang"),
    ("kemarin", "kemarin"), ("masalah", "masalah"), ("kendala", "kendala"),
    ("gangguan", "gangguan"), ("perbaiki", "perbaiki"), ("diperbaiki", "diperbaiki"),
    ("kembalikan", "kembalikan"), ("berlangganan", "berlangganan"), ("langganan", "langganan"),
    ("verifikasi", "verifikasi"), ("terjemahan", "terjemahan"), ("penjelasan", "penjelasan"),
    ("pertanyaan", "pertanyaan"), ("percakapan", "percakapan"), ("informasi", "informasi"),
    ("pengalaman", "pengalaman"), ("pengguna", "pengguna"), ("kecepatan", "kecepatan"),
    ("kecerdasan", "kecerdasan"), ("halusinasi", "halusinasi"), ("jaringan", "jaringan"),
    ("koneksi", "koneksi"), ("tampilan", "tampilan"), ("performa", "performa"),
    ("berbayar", "berbayar"), ("bermanfaat", "bermanfaat"), ("memuaskan", "memuaskan"),
    ("mengecewakan", "mengecewakan"), ("menyesal", "menyesal"), ("rekomendasi", "rekomendasi"),
    ("sempurna", "sempurna"), ("lumayan", "lumayan"), ("terbaik", "terbaik"),
    ("terburuk", "buruk"), ("lambat", "lambat"), ("cepat", "cepat"),
    ("mahal", "mahal"), ("murah", "murah"), ("gratis", "gratis"),
    ("pintar", "pintar"), ("cerdas", "cerdas"), ("akurat", "akurat"),
    ("nyaman", "nyaman"), ("mudah", "mudah"), ("sulit", "sulit"),
    ("susah", "susah"), ("ribet", "ribet"), ("jelek", "jelek"),
    ("buruk", "buruk"), ("rusak", "rusak"), ("parah", "parah"),
    ("kecewa", "kecewa"), ("kesal", "kesal"), ("hancur", "hancur"),
    ("bodoh", "bodoh"), ("salah", "salah"), ("ngawur", "ngawur"),
    ("sibuk", "sibuk"), ("macet", "macet"), ("keluar", "keluar"),
    ("masuk", "masuk"), ("kuota", "kuota"), ("batas", "batas"),
    ("habis", "habis"), ("pelit", "pelit"), ("iklan", "iklan"),
    ("suara", "suara"), ("gambar", "gambar"), ("bahasa", "bahasa"),
    ("indonesia", "indonesia"), ("inggris", "inggris"), ("skripsi", "skripsi"),
    ("kuliah", "kuliah"), ("belajar", "belajar"), ("tugas", "tugas"),
    ("pekerjaan", "pekerjaan"), ("jawaban", "jawaban"), ("solusi", "solusi"),
    ("kalimat", "kalimat"), ("memori", "memori"), ("karakter", "bot"),
    ("developer", "developer"), ("bintang", "bintang"), ("ulasan", "ulasan"),
]

generated: dict[str, str] = {}


def _add_safe(k: str, v: str) -> None:
    k = k.strip().lower()
    v = v.strip().lower()
    if (
        len(k) >= 2
        and k != v
        and k not in PROTECTED_STANDARD_KEYS
        and k not in slang
        and k not in generated
    ):
        generated[k] = v


# 3A. Variasi penghilangan satu vokal internal & singkatan konsonan chat Indonesia
for word, target in domain_targets:
    # Tanpa semua vokal internal (pertahankan huruf pertama)
    no_internal_vowels = word[0] + re.sub(r"[aiueo]", "", word[1:])
    if len(no_internal_vowels) >= 3:
        _add_safe(no_internal_vowels, target)

    # Penghilangan satu huruf vokal di posisi tengah/akhir (pola ketik cepat / typo vokal hilang)
    for idx in range(1, len(word)):
        if word[idx] in "aiueo":
            dropped = word[:idx] + word[idx + 1:]
            if len(dropped) >= 4:
                _add_safe(dropped, target)

    # Transposisi huruf bersebelahan (pola salah ketik jari di layar ponsel)
    for idx in range(1, len(word) - 1):
        if word[idx] != word[idx + 1]:
            swapped = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
            if len(swapped) >= 4:
                _add_safe(swapped, target)

    # Akhiran klitika chat populer (-ny, -nyaa, -x, -in)
    _add_safe(word + "ny", target)
    _add_safe(word + "nyaa", target)
    _add_safe(word + "x", target)

# 3B. Variasi elongasi (2 & 3 huruf akhir/vokal) dan partikel chat (-bgt, -pol, -lah, -sih, -dong)
sentiment_colloquial_bases = list(slang.items())
for base_k, base_v in sentiment_colloquial_bases:
    if not base_k.isalpha() or len(base_k) < 3 or len(base_k) > 10:
        continue
    # Elongasi 2 & 3 huruf terakhir
    _add_safe(base_k + base_k[-1], base_v)
    _add_safe(base_k + base_k[-1] * 2, base_v)
    # Gabungan tanpa spasi dengan intensifier populer Play Store ("bgt", "pol", "kali", "lah", "sih")
    if " " not in base_v and base_v in {
        "bagus", "mantap", "keren", "hebat", "cepat", "lambat", "mudah", "susah", "sulit",
        "mahal", "murah", "gratis", "pintar", "cerdas", "akurat", "nyaman", "jelek", "buruk",
        "rusak", "parah", "kecewa", "kesal", "menyesal", "sibuk", "macet", "pelit", "ribet",
        "membantu", "bermanfaat", "puas", "suka", "enak", "jelas", "kaku", "aneh", "lama",
    }:
        _add_safe(base_k + "bgt", f"{base_v} banget")
        _add_safe(base_k + "bngt", f"{base_v} banget")
        _add_safe(base_k + "pol", f"{base_v} sekali")
        _add_safe(base_k + "poll", f"{base_v} sekali")
        _add_safe(base_k + "lah", base_v)
        _add_safe(base_k + "sih", base_v)
        _add_safe(base_k + "nya", base_v)

# Gabungkan seluruh entri baru ke kamus utama
slang.update(generated)
sorted_slang = dict(sorted(slang.items()))

with open(SLANG_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted_slang, f, indent=2, ensure_ascii=False)

print(f"[OK] Kamus slang berhasil diperluas menjadi {len(sorted_slang):,} entri terkurasi!")
