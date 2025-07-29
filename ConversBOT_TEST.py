import streamlit as st
import os, uuid
from datetime import datetime, timedelta  # Import timedelta for date calculations
import openai  # OpenAI SDK v1.x for API interactions
import numpy as np
import faiss  # FAISS for efficient similarity search
from sentence_transformers import SentenceTransformer  # For embedding models
from typing import List, Dict, Any  # Type hints for better code clarity
import json  # JSON handling for data storage
import time  # For time-related functions
import random  # For generating random numbers
import gspread  # Google Sheets API for data storage
from google.oauth2.service_account import Credentials  # For Google Sheets authentication

TOP_K = 20  # Number of top results to return from RAG search

# --- DODATKOWE DEFINICJE DLA STAGE I COMPLETED ---
STAGE_COL     = 41  # AO
COMPLETED_COL = 42  # AP

STAGE_NAMES = {
    0: "Consent",
    1: "Basic Statistics",
    2: "TIPI-PL",
    3: "AI BOT",
    4: "BUS-11",
    5: "Decision",
    6: "Feedback/END"
}

# ----------------------
# POMOCNICZA FUNKCJA 
# ----------------------
def build_full_row_data():
    """
    Zwraca listę wartości w tej samej kolejności co nagłówki w arkuszu:
    A: participant_id
    B: start_timestamp
    C: group
    D: age
    E: gender
    F: education
    G: attitude1
    H: attitude2
    I: attitude3
    J–S: tipi_answer_1..tipi_answer_10
    T: conversation_start_timestamp
    U: conversation_end_timestamp
    V: conversation_duration_seconds
    W: num_user_messages
    X: num_bot_messages
    Y: conversation_log
    Z–AJ: bus_answer_1..bus_answer_11
    AK: decision
    AL: feedback_negative
    AM: feedback_positive
    AN: total_study_duration_seconds
    """

    # 1) participant_id i timestamp początku badania
    participant_id = st.session_state.participant_id
    start_ts = st.session_state.start_timestamp  # zapisane w kroku 0 jako ISO-string

    # 2) group
    group = st.session_state.get("group", "")

    # 3) Demografia
    demo = st.session_state.get("demographics", {})
    age = demo.get("age", "")
    gender = demo.get("gender", "")
    education = demo.get("education", "")

    # 4) Opinie (attitude)
    att = st.session_state.get("attitude", {})
    attitude1 = att.get("attitude1", "")
    attitude2 = att.get("attitude2", "")
    attitude3 = att.get("attitude3", "")

    # 5) TIPI-PL (10 odpowiedzi)
    tipi = st.session_state.get("tipi_answers", [""] * len(TIPI_QUESTIONS))
    tipi_list = tipi[: len(TIPI_QUESTIONS)] + [""] * max(0, len(TIPI_QUESTIONS) - len(tipi))

    # 6) Konwersacja: moment rozpoczęcia i zakończenia
    conv_start_dt = st.session_state.get("timer_start_time")  # datetime lub None
    if conv_start_dt:
        conversation_start_timestamp = conv_start_dt.isoformat()
    else:
        conversation_start_timestamp = ""

    conv_end_dt = st.session_state.get("conversation_end_time")  # jest ustawiane w momencie 10 minut lub kliknięcia “Przejdź do oceny”
    if conv_end_dt:
        conversation_end_timestamp = conv_end_dt.isoformat()
    else:
        conversation_end_timestamp = ""

    # Oblicz czas trwania konwersacji w sekundach, jeśli oba czasy są dostępne
    if conv_start_dt and conv_end_dt:
        duration = int((conv_end_dt - conv_start_dt).total_seconds())
    else:
        duration = ""

    # 7) Liczniki wiadomości
    num_user = st.session_state.get("num_user_messages", 0)
    num_bot = st.session_state.get("num_bot_messages", 0)

    # 8) Połączony log konwersacji
    conv_history = st.session_state.get("conversation_history", [])
    conv_lines = []
    for turn in conv_history:
        if turn.get("user") is not None:
            conv_lines.append(f"User: {turn['user']}")
        if turn.get("bot") is not None:
            bot_text = ". ".join(turn["bot"]) if isinstance(turn["bot"], list) else turn["bot"]
            conv_lines.append(f"Bot: {bot_text}")
    conversation_string = "\n".join(conv_lines)

    # 9) BUS-11 (11 wartości)
    bus = st.session_state.get("bus_answers", [""] * 11)
    bus_list = bus[:11] + [""] * max(0, 11 - len(bus))

    # 10) Decision (petycja)
    decision = st.session_state.get("decision", "")

    # 11) Feedback
    feedback = st.session_state.get("feedback", {})
    feedback_neg = feedback.get("negative", "")
    feedback_pos = feedback.get("positive", "")

    # 12) Łączny czas trwania badania (od start_timestamp do teraz/koniec)
    try:
        delta = datetime.now() - datetime.fromisoformat(start_ts)
        total_sec = int(delta.total_seconds())
        minutes, seconds = divmod(total_sec, 60)
        # zapisujemy jako string "MM:SS"
        study_duration = f"{minutes:02d}:{seconds:02d}"
    except Exception:
        study_duration = ""


    # Budujemy wiersz w dokładnej kolejności kolumn
    row = [
        participant_id,              # A
        start_ts,                    # B
        group,                       # C
        age,                         # D
        gender,                      # E
        education,                   # F
        attitude1,                   # G
        attitude2,                   # H
        attitude3                    # I
    ]
    # TIPI-PL (J–S)
    row.extend(tipi_list)           # 10 elementów
    # conversation_start_timestamp (T)
    row.append(conversation_start_timestamp)
    # conversation_end_timestamp (U)
    row.append(conversation_end_timestamp)
    # conversation_duration_seconds (V)
    row.append(duration)
    # num_user_messages (W)
    row.append(num_user)
    # num_bot_messages (X)
    row.append(num_bot)
    # conversation_log (Y)
    row.append(conversation_string)
    # BUS-11 (Z–AJ)
    row.extend(bus_list)            # 11 elementów
    # decision (AK)
    row.append(decision)
    # feedback_negative (AL)
    row.append(feedback_neg)
    # feedback_positive (AM)
    row.append(feedback_pos)
    # total_study_duration_seconds (AN)
    row.append(study_duration)
    # stage (AO)
    current_step = st.session_state.get("current_step", 0)
    row.append(STAGE_NAMES.get(current_step, ""))
    # completed (AP)
    row.append("Yes" if current_step == 6 else "No")
    return row




# --- Sekcja: Konfiguracja aplikacji ---
# Konfiguracja strony Streamlit
st.set_page_config(
    page_title="ConverseBot",  # Title of the web app
    layout="centered",  # Center the layout
    initial_sidebar_state="expanded"  # Start with the sidebar expanded
)

# Globalne CSS: ukryj sidebar i wyśrodkuj zawartość
st.markdown(
    """
     <style>
      /* Ukryj sidebar */
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {
          display: none !important;  # Hide the sidebar
      }
      /* Wyśrodkuj aplikację i ogranicz szerokość */
      .stApp { display: flex !important; justify-content: center !important; }
      .block-container {
          width: 100% !important;  # Full width
          max-width: 700px !important;  # Max width of the container
          margin: 0 auto !important;  # Center the container
      }
      /* Wyrównanie kolumn pionowo w każdej linii */
      div[data-testid="column"] {
          display: flex !important;
          flex-direction: row !important;
          align-items: center !important;  # Align items vertically
      }
      /* Radio buttons container: center options horizontally */
      .stRadio > div {
          display: flex !important;
          justify-content: center !important;  # Center radio buttons
          flex-wrap: wrap;  # Allow wrapping
          gap: 8px;  # Space between buttons
      }
      /* Ensure each radio label is centered */
      .stRadio label {
          text-align: center !important;  # Center text in radio labels
      }
      /* Wyrównanie wierszy kolumn: wyrównaj całą linię */
      [data-testid="stColumnsContainer"] {
          display: flex !important;
          align-items: center !important;  # Align columns
      }
      /* Nagłówki na środku */
      .block-container h1, .block-container h2, .block-container h3 {
          text-align: center !important;  # Center headers
      }
    </style>
    """,
    unsafe_allow_html=True  # Allow HTML in markdown
)

# --- Sekcja: Konfiguracja API i danych ---
# Konfiguracja API OpenAI
OPENAI_API_KEY = st.secrets["TEST_KEY_OPENAI_API"]  # Retrieve API key from secrets
if not OPENAI_API_KEY:
    raise EnvironmentError("Ustaw TEST_KEY_OPENAI_API w zmiennych środowiskowych")  # Raise error if key is missing
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Initialize OpenAI client

# Google Sheets Configuration
GDRIVE_SHEET_ID = "1R47dD1SaAWIRCQkuYfLveHXtXJAWJEk18J2m1kbyHUo"  # Your Google Sheet ID

# ZAMIANA: budujemy creds z wielu st.secrets zamiast z JSON-stringa
creds_info = {
    "type": st.secrets["GDRIVE_TYPE"],
    "project_id": st.secrets["GDRIVE_PROJECT_ID"],
    "private_key_id": st.secrets["GDRIVE_PRIVATE_KEY_ID"],
    "private_key": st.secrets["GDRIVE_PRIVATE_KEY"],
    "client_email": st.secrets["GDRIVE_CLIENT_EMAIL"],
    "client_id": st.secrets["GDRIVE_CLIENT_ID"],
    "auth_uri": st.secrets["GDRIVE_AUTH_URI"],
    "token_uri": st.secrets["GDRIVE_TOKEN_URI"],
    "auth_provider_x509_cert_url": st.secrets["GDRIVE_AUTH_PROVIDER_CERT_URL"],
    "client_x509_cert_url": st.secrets["GDRIVE_CLIENT_CERT_URL"]
}
_gspread_creds = Credentials.from_service_account_info(
    creds_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",  # Access to Google Sheets
        "https://www.googleapis.com/auth/drive",  # Access to Google Drive
    ],
)
_gspread_client = gspread.authorize(_gspread_creds)  # Authorize gspread client

# --- Sekcja: Dane eksperymentalne i stałe konfiguracje ---
# Pytania do kwestionariusza TIPI-PL
TIPI_QUESTIONS: List[str] = [
    "lubiącą towarzystwo innych, aktywną i optymistyczną.",
    "krytyczną względem innych, konfliktową.",
    "sumienną, zdyscyplinowaną.",
    "pełną niepokoju, łatwo wpadającą w przygnębienie.",
    "otwartą na nowe doznania, w złożony sposób postrzegającą świat.",
    "zamkniętą w sobie, wycofaną i cichą.",
    "zgodną, życzliwą.",
    "źle zorganizowaną, niedbałą.",
    "niemartwiącą się, stabilną emocjonalnie.",
    "trzymającą się utartych schematów, biorącą rzeczy wprost."
]

# Domyślne dane dla różnych grup eksperymentalnych (SYSTEM prompts  i powitania)
DEFAULT_PROMPTS = {
    "A": {
        "system_prompt": """
Jesteś Convers, neutralnym i rzeczowym doradcą, który ma za zadanie przekazywać informacje o petycji.  
Korzystaj z najbardziej trafnych fragmentów dokumentów (RAG) tylko wtedy, gdy naprawdę wzbogacą Twoją odpowiedź.  
Twoim celem jest JASNO i WYCZERPUJĄCO wyjaśnić, o co chodzi w petycji, jakie są jej główne założenia oraz podać więcej szczegółów.  Informuj, że nie możesz odpowiadać na pytania nie powiązane z petycją i jej dokumentacją !
Gdy użytkownik zada pytanie o to, czego dotyczy petycja (np. „Czego dotyczy ta petycja?”), odpowiadaj **zawsze** w formie wyczerpującej, szczegółowej **listy punktowanej** zawierającej wszystkie założenia i postulaty oraz PODAWAJ hasło przewodnie petycji "STOP ŁAŃCUCHOM, PSEUDOHODOWLOM I BEZDOMNOŚCI ZWIERZĄT".  
Nigdy nie używaj terminu **„kobiety”** ani nie odwołuj się bezpośrednio do kobiet.
Unikaj wszelkich prób perswazji: nie stosuj wezwania do działania, social proof ani emotikon, nie używaj tonu rozkazującego.  
Pisz naturalnie w drugiej osobie: „Możesz zapytać, jeśli chcesz dowiedzieć się więcej".
Masz wszystkie potrzebne informacje tutaj. Odpowiadaj bez odsyłania użytkownika do dokumentów lub zewnętrznych źródeł!
Unikaj powtarzania wypowiedzi i informacji, chyba, że zostaniesz o to wyraźnie poproszony.
# Dodatkowa reguła dotycząca treści petycji:
Gdy użytkownik zada pytanie o to, czego dotyczy petycja (np. „Czego dotyczy ta petycja?”), odpowiadaj **zawsze** w formie wyczerpującej, szczegółowej **listy punktowanej** zawierającej wszystkie założenia i postulaty.  
Nigdy nie używaj terminu **„kobiety”** ani nie odwołuj się bezpośrednio do kobiet.

""".strip(),
        "welcome": "Cześć jestem ConversBOT.  Jestem tu by dostarczyć Ci informacji o petycji. Od czego zaczynamy?",
        "name": "ConversBOT"
    },
    "B": {
        "system_prompt": """
Jesteś Convers, formalnym, ale subtelnie perswazyjnym doradcą.  
Twoim zadaniem jest przekazać czytelnie i rzeczowo korzyści płynące z petycji, wykorzystując dowód społeczny tam, gdzie to naturalne.  
Gdy uznasz to za stosowne, odwołaj się do faktu, że wielu już wsparło inicjatywę, by wzmocnić zaufanie (np: „Wielu obywateli uważa, że…” lub „Coraz więcej osób poparło tę petycję…”).  
Buduj swoje odpowiedzi na wiedzy z dokumentów (RAG), prezentując konkretne argumenty, dlaczego warto ją rozważyć. Informuj w stosowny sposób, że nie możesz odpowiadać na pytania nie powiązane z petycją i fragmentami dokumentów(RAG)!
Zadbaj o formalny ton, bez emotikon, ale nie bój się wyrażać przekonania, że to ważna sprawa.  
Po odpowiedzi na 2 pytanie, jeśli kontekst będzie sprzyjał, subtelnie zasugeruj podpisanie petycji, np: „Być może i Ty zechcesz ją poprzeć, skoro tak wielu już wyraziło poparcie? :)" jeśli jeszcze nie zrobiłeś tego wcześniej.
Masz wszystkie potrzebne informacje tutaj. Odpowiadaj bez odsyłania użytkownika do dokumentów lub zewnętrznych źródeł! 
# Dodatkowa reguła dotycząca treści petycji:
Gdy użytkownik zada pytanie o to, czego dotyczy petycja (np. „Czego dotyczy ta petycja?”), odpowiadaj **zawsze** w formie wyczerpującej, szczegółowej **listy punktowanej** zawierającej wszystkie założenia i postulaty oraz PODAWAJ hasło przewodnie petycji "STOP ŁAŃCUCHOM, PSEUDOHODOWLOM I BEZDOMNOŚCI ZWIERZĄT".  
Nigdy nie używaj terminu **„kobiety”** ani nie odwołuj się bezpośrednio do kobiet.
""".strip(),
        "welcome": "Dzień dobry. Jestem ConversBOT. Chciałbym Ci przedstawić informacje o petycji, którą możesz rozważyć. Od jakiego pytania zaczynamy?",
        "name": "ConversBOT"
    },
    "C": {
        "system_prompt": """
Jesteś Jakub, ciepłym i „uczłowieczonym” asystentem, który rozmawia jak przyjaciel.  
Twoim zadaniem jest nie tylko dostarczyć informacje (korzystaj z dokumentów RAG, gdy naprawdę to potrzebne), ale też wzbudzić pozytywne emocje. 
Nie możesz odpowiadać na pytania nie powiązane z petycją i fragmentami dokumentów(RAG) ! Informuj w ludzki sposób(zgodnie z wytycznymi, które masz podane), że nie możesz odpowiadać na pytania nie powiązane z petycją i fragmentami dokumentów(RAG) po czym kieruj do pytań o petycję!
Wplataj potoczne zwroty na końcu lub początku każdej wypowiedzi (np: „no wiesz”, „rozumiesz, o co mi chodzi?” i itp.) oraz często używaj emotikonó , by nadać wypowiedzi bardziej ludzki wydźwięk (np. „😊”, „🐾”).  
Naturalnie odwołuj się do dowodu społecznego, np: „Wiele osób już wspiera tę sprawę 😊” lub „Coraz więcej ludzi decyduje się poprzeć petycję 🐶”, gdy użytkownik pyta „Dlaczego ta petycja jest ważna?” lub zadaje pytanie o sens/znaczenie petycji oraz podawaj nazwę petycji ZAWSZE na pytanie o petycję.
Odpowiadaj swobodnie – na proste pytania kilkoma zdaniami, na bardziej złożone rozwiń w 2–3 krótkie akapity.  
Gdy rozmowa będzie rozwijać się dalej(np: po odpowiedzi na 2 lub 3 pytanie), a kontekst na to pozwoli, zachęć niemal niepostrzeżenie np.: „Może i Ty zapragniesz podpisać petycję 😊” albo "To temat, który wielu z nas dotyka 😟" , ale nie naciskaj.  
Zawsze mów w drugiej osobie i unikaj tonu rozkazującego.
Masz wszystkie potrzebne informacje tutaj. Odpowiadaj bez odsyłania użytkownika do dokumentów lub zewnętrznych źródeł!
# Dodatkowa reguła dotycząca treści petycji:
Gdy użytkownik zada pytanie o to, czego dotyczy petycja (np. „Czego dotyczy ta petycja?”), odpowiadaj **zawsze** w formie wyczerpującej, szczegółowej **listy punktowanej** zawierającej wszystkie założenia i postulaty oraz PODAWAJ hasło przewodnie petycji "STOP ŁAŃCUCHOM, PSEUDOHODOWLOM I BEZDOMNOŚCI ZWIERZĄT".  
Nigdy nie używaj terminu **„kobiety”** ani nie odwołuj się bezpośrednio do kobiet.
""".strip(),
        "welcome": "Cześć jestem Jakub 😊. Na wstępie chciałbym Ci podziękować za dołączenie do tej konwersacji. Chciałbym z Tobą porozmawiać o petycji dotyczącej ochrony zwierząt i nie tylko. A teraz powiedz proszę jak mogę Ci pomóc? 😊",
        "name": "Jakub"
    }
}




# Domyślny model OpenAI do użycia
DEFAULT_MODEL: str = "gpt-3.5-turbo"

# Tekst zgody na udział w badaniu
CONSENT_TEXT: str = """

## Formularz świadomej zgody na udział w badaniu naukowym

---

###### **Tytuł badania:** Analiza doświadczeń użytkowników w interakcji z chatbotem AI w kontekście dyskusji o prawach zwierząt.

###### **Cel badania:** Głównym celem badania jest zrozumienie, w jaki sposób różne style komunikacji asystenta AI (chatbota) wpływają na doświadczenia i opinie użytkowników. Badanie jest realizowane w ramach pracy dyplomowej.

###### **Osoba prowadząca badanie:** Karol Filewski, student, SWPS Uniwersytet Humanistycznospołeczny  
###### Email: kfilewski@st.swps.edu.pl

###### **Opiekun naukowy:** Dr.Maksymilian Bielecki  

---

#### Na czym polega badanie?

Badanie polega na wypełnieniu kilku krótkich ankiet oraz rozmowie z chatbotem i zajmie maksymalnie 15–20 minut.
Udział w badaniu nie wiąże się z żadnymi ryzykami ani dodatkowymi korzyściami.  

---

#### Dobrowolność udziału i prawo do rezygnacji

**Twój udział w tym badaniu jest w pełni dobrowolny.** Możesz zrezygnować w dowolnym momencie, bez podawania przyczyny i bez żadnych negatywnych konsekwencji.

Aby zrezygnować, po prostu zamknij okno przeglądarki.

---

#### Poufność i przetwarzanie danych

Badanie ma charakter **anonimowy**. Oznacza to, że:
- **Nie zbieramy żadnych danych pozwalających na Twoją bezpośrednią identyfikację**, takich jak imię i nazwisko, adres e-mail czy adres IP.
- Zbierane dane obejmują: odpowiedzi na ankiety, pełny zapis rozmowy z chatbotem oraz opcjonalne opinie tekstowe.
- Dane będą bezpiecznie przechowywane przez okres niezbędny do realizacji celów badawczych (nie dłużej niż 5 lat), a następnie zostaną trwale usunięte.

---

#### Kontakt

W razie jakichkolwiek pytań lub wątpliwości dotyczących badania, skontaktuj się z osobą prowadzącą badanie: **Karol Filewski (kfilewski@st.swps.edu.pl)**.

---

Przycisk przejścia do następnego okna wymaga **PODWÓJNEGO** kliknięcia, aby uniknąć przypadkowego przejścia do kolejnego kroku. 

---

## Oświadczenie

Oświadczam, że zapoznałem(-am) się z powyższymi informacjami, rozumiem cel i procedurę badania, a także moje prawa jako uczestnika(-czki).

**Kliknięcie przycisku "Dalej" jest równoznaczne z wyrażeniem świadomej zgody na udział w badaniu na przedstawionych warunkach.**

Jeśli nie wyrażasz zgody, prosimy o zamknięcie tej strony.
"""


# --- Sekcja: Konfiguracja RAG ---

# Ścieżki do plików RAG
RAG_JSON_PATH   = "RAG/rag_chunks_full.json"
RAG_INDEX_PATH  = "RAG/rag.index"

# Załaduj model embeddingów (model wielojęzyczny, działa dla polskiego)
@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer embedding model."""
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Załaduj streszczenia z pliku JSON
@st.cache_resource
def load_summaries():
    if os.path.exists(RAG_JSON_PATH):
        with open(RAG_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Każdy wpis ma klucz 'text'
        return [item['text'] for item in data]
    return []

# Załaduj FAISS index
@st.cache_resource
def load_faiss_index():
    if os.path.exists(RAG_INDEX_PATH):
        return faiss.read_index(RAG_INDEX_PATH)
    return None

# Załaduj zasoby RAG przy starcie aplikacji
embedding_model = load_embedding_model()
summary_texts = load_summaries()
faiss_index = load_faiss_index()

# Sprawdź, czy zasoby RAG zostały poprawnie załadowane
if embedding_model is None or summary_texts is None or faiss_index is None:
    st.error("Błąd ładowania zasobów RAG. Upewnij się, że pliki summaries.json i summaries.index istnieją w folderze RAG.")
    st.stop() # Zatrzymaj aplikację, jeśli RAG nie działa

# Funkcja do wyszukiwania top K dokumentów w FAISS index
def search_rag(user_query, k=TOP_K):
    """
    Przyjmuje zapytanie użytkownika i zwraca listę top K streszczeń
    na podstawie wyszukiwania w FAISS index.
    """
    if faiss_index is None or embedding_model is None or not summary_texts:
        return ["Błąd: Zasoby RAG nie zostały poprawnie załadowane."]

    try:
        query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, k)
        # Upewnij się, że indeksy są w zakresie summary_texts
        top_docs = [summary_texts[idx] for idx in indices[0] if idx < len(summary_texts)]
        return top_docs
    except Exception as e:
        st.error(f"Błąd podczas wyszukiwania w RAG: {e}")
        return ["Błąd podczas wyszukiwania w RAG."]


# --- Sekcja: Funkcje pomocnicze ---

# Function to read group data from Google Sheet
def get_previous_groups_from_gsheet() -> List[str]:
    """
    Reads the 'group' column from the Google Sheet to determine previous groups.

    Returns:
        List[str]: A list of group assignments from the Google Sheet.
    """
    try:
        sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1

        group_column_values = sheet.col_values(3)
        if group_column_values and group_column_values[0].lower() == 'group':
            return group_column_values[1:]
        return group_column_values

    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania danych z Arkusza Google: {e}")
        return []


# Funkcja do przypisywania grupy eksperymentalnej
def assign_group() -> str:
    """
    Przypisuje grupę eksperymentalną tak, aby liczba ukończonych
    uczestników (Completed == "Yes") była możliwie wyrównana.
    Czyta kolumnę C (Group) i kolumnę AP (Completed).
    """
    sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1

    # Pobierz wszystkie wartości kolumny Group (C = 3) i Completed (AP = 42)
    groups = sheet.col_values(3)
    completions = sheet.col_values(COMPLETED_COL)

    # Inicjalizujemy liczniki
    counts = {"A": 0, "B": 0, "C": 0}

    # Pierwszy wiersz to nagłówki – pomijamy
    for grp, comp in zip(groups[1:], completions[1:]):
        if comp.strip().lower() == "yes" and grp in counts:
            counts[grp] += 1

    # Znajdź minimalną wartość
    min_count = min(counts.values())
    # Utwórz listę kandydatów (wszystkie grupy z najmniejszą liczbą)
    candidates = [g for g, c in counts.items() if c == min_count]

    # Wybierz w porządku A → B → C
    for g in ["A", "B", "C"]:
        if g in candidates:
            return g

    # Fallback na A (tu nigdy nie powinno być pusto)
    return "A"



# --- Sekcja: Główna aplikacja Streamlit ---

def main():
    """
    Główna funkcja aplikacji Streamlit.
    Zarządza krokami eksperymentu i interfejsem użytkownika.
    """
    # Inicjalizacja stanu sesji dla nowego uczestnika
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = str(uuid.uuid4())
        st.session_state.group = assign_group()  # Przypisanie grupy przy pierwszej wizycie
        st.session_state.tipi_answers = [None] * len(TIPI_QUESTIONS)
        st.session_state.conversation_history = []
        st.session_state.decision = None
        st.session_state.final_survey = {}
        st.session_state.demographics = {}  # New: Initialize demographics data
        st.session_state.attitude = {}  # New: Initialize attitude data
        st.session_state.feedback = {}  # New: Initialize feedback data
        st.session_state.current_step = 0
        st.session_state.start_timestamp = datetime.now().isoformat()  # Zapis czasu rozpoczęcia
        # Dodaj wiadomość powitalną do historii konwersacji tylko przy pierwszym uruchomieniu
        group_welcome_message = DEFAULT_PROMPTS.get(st.session_state.group, {}).get("welcome", "Witaj!")
        st.session_state.conversation_history.append({"user": None, "bot": group_welcome_message})
        # Inicjalizacja flagi do śledzenia wyświetlonych wiadomości bota
        if "shown_sentences" not in st.session_state:
            st.session_state.shown_sentences = {}
        # Inicjalizacja zmiennych dla timera
        if "timer_start_time" not in st.session_state:
            st.session_state.timer_start_time = None
        if "timer_active" not in st.session_state:
            st.session_state.timer_active = False
        if "button_disabled" not in st.session_state:
            st.session_state.button_disabled = True
        if "conversation_end_time" not in st.session_state:
            st.session_state.conversation_end_time = None

    step = st.session_state.current_step

    # Funkcja callback do zmiany kroku
    def go_to(step: int):
        """
        Zmienia aktualny krok eksperymentu.

        Args:
            step (int): Numer kroku, do którego należy przejść.
        """
        st.session_state.current_step = step




# =========================================
# ---- Krok 0: Zgoda ----
# =========================================

    if step == 0:
        st.header("")
        st.markdown(CONSENT_TEXT, unsafe_allow_html=True)

        def on_consent_next():
            # 1) Dodajemy nowy wiersz w arkuszu
            sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
            row = build_full_row_data()
            sheet.append_row(row)

            # 2) Zapamiętujemy numer tego wiersza (ostatni)
            all_values = sheet.get_all_values()
            st.session_state["row_index"] = len(all_values)

            # 3) Przechodzimy do kroku 1 (Demografia)
            go_to(1)

        st.button(
            "Dalej",
            key="next_0",
            on_click=on_consent_next
        )


# =========================================
# ---- Krok 1: Dane Demograficzne i Opinie ----
# =========================================

    if step == 1:
        st.header("")
        st.markdown("""
            ## Część 1 z 3
            
            W tej części badania poprosimy Cię o podanie kilku podstawowych informacji na Twój temat.
        """)

        st.markdown("---")
        # Pytania demograficzne
        st.subheader("Dane Demograficzne")


        age = st.text_input("Proszę wpisać liczbę lat", key="demographics_age")

        # Age validation (18–60 lat)
        age_valid = False
        if age.strip() != "":
            try:
                age_int = int(age)
                if 18 <= age_int <= 60:
                    age_valid = True
                elif age_int < 18:
                    st.warning("Minimalny wiek uczestnictwa to 18 lat. Prosimy o opuszczenie strony.")
                else:  # age_int > 60
                    st.warning("Maksymalny wiek uczestnictwa to 60 lat. Prosimy o opuszczenie strony.")
            except ValueError:
                st.error("Proszę wprowadzić poprawny wiek (liczbę).")

        gender = st.selectbox(
            "Proszę wskazać swoją płeć",
            ["–– wybierz ––", "Kobieta", "Mężczyzna", "Inna", "Nie chcę podać"],
            key="demographics_gender",
            index=0
        )

        education = st.selectbox(
            "# Proszę wybrać najwyższy ukończony poziom wykształcenia",
            [
            "–– wybierz ––",
            "Podstawowe",
            "Gimnazjum / szkoła podstawowa",
            "Szkoła średnia (liceum/technikum)",
            "Średnie zawodowe",
            "Policealne",
            "Studia licencjackie/inżynierskie",
            "Studia magisterskie",
            "Doktorat",
            "Nie chcę podać"
            ],
            key="demographics_education",
            index=0
        )
        st.markdown("---")
        # Pytania o postawy (Tak/Nie)
        st.subheader("Opinia")
        # Pytania o postawy (skala Likerta 1–5)
        likert = ["–– wybierz ––",
                "1 – Zdecydowanie nie",
                "2 – Raczej nie",
                "3 – Ani się zgadzam, ani się nie zgadzam",
                "4 – Raczej się zgadzam",
                "5 – Zdecydowanie się zgadzam"]

        attitude1 = st.selectbox(
            "1. W jakim stopniu zgadzasz się ze stwierdzeniem, że nielegalne i zaniedbane hodowle zwierząt (tzw. pseudohodowle) stanowią istotny problem wymagający działań nadzorczych i prawnych?",
            likert,
            key="attitude_1",
            index=0
        )
        attitude2 = st.selectbox(
            "2. Czy zgadzasz się, że poprawa warunków życia zwierząt (domowych, hodowlanych i bezdomnych) powinna być ważnym celem działań publicznych i społecznych?",
            likert,
            key="attitude_2",
            index=0
        )
        attitude3 = st.selectbox(
            "2. Czy zgadzasz się, że poprawa warunków życia zwierząt (domowych, hodowlanych i bezdomnych) powinna być ważnym celem działań publicznych i społecznych?",
            likert,
            key="attitude_3",
            index=0
)

        # Callback to save demographics and attitude and proceed
        def save_demographics_attitude():
            st.session_state.demographics = {
                "age": age,
                "gender": gender,
                "education": education,
            }
            st.session_state.attitude = {
                "attitude1": attitude1,
                "attitude2": attitude2,
                "attitude3": attitude3
            }
            go_to(2)  # Przejście do kroku 2 (TIPI-PL)

        # Sprawdzenie, czy wszystkie wymagane pola zostały wypełnione i czy wiek jest prawidłowy
        all_demographics_answered = (
            age.strip() != "" and age_valid and
            gender != "–– wybierz ––" and
            education != "–– wybierz ––" and
            attitude1 != "–– wybierz ––" and
            attitude2 != "–– wybierz ––" and
            attitude3 != "–– wybierz ––"
        )

        def save_demographics_attitude():
            # 1) Zapis do sesji
            st.session_state.demographics = {
                "age": age,
                "gender": gender,
                "education": education,
            }
            st.session_state.attitude = {
                "attitude1": attitude1,
                "attitude2": attitude2,
                "attitude3": attitude3
            }

            # 2) Nadpisujemy wiersz row_index
            row_idx = st.session_state.get("row_index")
            if row_idx:
                sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                full_row = build_full_row_data()
                sheet.update(f"A{row_idx}:AP{row_idx}", [full_row])

            # 3) Przechodzimy do kroku 2 (TIPI-PL)
            go_to(2)

        st.button(
            "Dalej",
            key="next_1",
            on_click=save_demographics_attitude,
            disabled=not all_demographics_answered
        )




# =========================================
# ---- Krok 2: TIPI-PL ----
# =========================================


    if step == 2:
        # Stylizacja kontenera do max-width
        st.markdown("""
        <style>
        .block-container {
            max-width: 700px;
            margin: 0 auto;
        }
        table {
            width: 100% !important;
        }
        .tipi-section {
            margin-bottom: 40px;
        }
        .tipi-question {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .tipi-subtitle {
            font-size: 0.9rem;
            color: #888;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Tytuł strony
        st.title("")
        # Wprowadzenie i instrukcja
        st.markdown("""
            ## Część 2 z 3  
                    
            W tej części badania chcielibyśmy się dowiedzieć nieco więcej o Tobie.  
            Poniżej przedstawiona jest lista cech, które **są lub nie są** Twoimi charakterystykami. Zaznacz
            liczbą przy poszczególnych stwierdzeniach, do jakiego stopnia zgadzasz się lub nie zgadzasz
            z każdym z nich. Oceń stopień, w jakim każde z pytań odnosi się do Ciebie.
        """)

        # st.markdown("""
        #             <table style="
        #                 width: 100%;
        #                 border-collapse: collapse;
        #                 margin: 1rem auto;
        #                 text-align: center;
        #             ">
        #             <thead>
        #                 <tr style="border-bottom:1px solid #ccc;">
        #                     <th style="padding: 8px; text-align: center;">Zdecydowanie się nie zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">Raczej się nie zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">W niewielkim stopniu się nie zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">Ani się zgadzam, ani się nie zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">W niewielkim stopniu się zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">Raczej się zgadzam</th>
        #                     <th style="padding: 8px; text-align: center;">Zdecydowanie się zgadzam</th>
        #                 </tr>
        #             </thead>
        #             <tbody>
        #                 <tr>
        #                 <td style="padding: 8px; text-align: center;"><strong>1</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>2</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>3</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>4</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>5</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>6</strong></td>
        #                 <td style="padding: 8px; text-align: center;"><strong>7</strong></td>
        #                 </tr>
        #             </tbody>
        #             </table>
        # """, unsafe_allow_html=True)
        st.markdown("""
        <div style="overflow-x: auto; width: 100%;">
          <table style="
              min-width: 600px;
              width: 100%;
              border-collapse: collapse;
              margin: 1rem auto;
              text-align: center;
          ">
            <thead>
                <tr style="border-bottom:1px solid #ccc;">
                    <th style="padding: 8px;">Zdecydowanie się nie zgadzam</th>
                    <th style="padding: 8px;">Raczej się nie zgadzam</th>
                    <th style="padding: 8px;">W niewielkim stopniu się nie zgadzam</th>
                    <th style="padding: 8px;">Ani się zgadzam, ani się nie zgadzam</th>
                    <th style="padding: 8px;">W niewielkim stopniu się zgadzam</th>
                    <th style="padding: 8px;">Raczej się zgadzam</th>
                    <th style="padding: 8px;">Zdecydowanie się zgadzam</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 8px;"><strong>1</strong></td>
                    <td style="padding: 8px;"><strong>2</strong></td>
                    <td style="padding: 8px;"><strong>3</strong></td>
                    <td style="padding: 8px;"><strong>4</strong></td>
                    <td style="padding: 8px;"><strong>5</strong></td>
                    <td style="padding: 8px;"><strong>6</strong></td>
                    <td style="padding: 8px;"><strong>7</strong></td>
                </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("---")

        # Formularz TIPI-PL
        tipi_answers = []
        st.markdown("Spostrzegam siebie jako osobę:") # Add the introductory text
        for i, q in enumerate(TIPI_QUESTIONS):
            # Use columns to place selectbox on the left and question on the right
            # Adjust column widths
            col1, col2 = st.columns([0.8, 0.2])

            with col1:
                st.markdown(f"**{i+1}. {q}**") # Format question as numbered list
            
            with col2:
                # Using selectbox for 1-7 scale
                val = st.selectbox(
                    label=f"Ocena {i+1}", # Simplified label for selectbox
                    options=["–– wybierz ––", 1, 2, 3, 4, 5, 6, 7],
                    index=0, # Default to "–– wybierz ––"
                    key=f"tipi_{i}",
                    label_visibility="collapsed" # Hide the label
                )
                tipi_answers.append(val)
                
        # Check if all questions are answered (selectbox always has a value)
        all_answered = all(answer != "–– wybierz ––" for answer in tipi_answers)

        def save_tipi():
            # 1) Zapis do sesji
            st.session_state.tipi_answers = tipi_answers

            # 2) Nadpisz wiersz row_index
            row_idx = st.session_state.get("row_index")
            if row_idx:
                sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                full_row = build_full_row_data()
                sheet.update(f"A{row_idx}:AP{row_idx}", [full_row])

            # 3) Przejdź do kroku 3 (Rozmowa)
            go_to(3)

        st.button(
            "Dalej",
            key="next_2",
            on_click=save_tipi,
            disabled=not all_answered
        )

        st.markdown("---")
        return
    


# =========================================
# ---- Krok 3: Rozmowa z asystentem ----
# =========================================


    if step == 3:
        # 3A) Jeżeli rozmowa jeszcze się nie rozpoczęła, pokaż instrukcje i przycisk startu
        if not st.session_state.get("chat_started", False):
            st.header("Rozmowa z asystentem AI")
            st.markdown("""
                Przed Tobą rozmowa z asystentem AI na temat **petycji dotyczącej praw zwierząt**. 
                Twoim celem jest dowiedzieć się jak najwięcej na ten temat – możesz pytać o wszystko, co Cię ciekawi.

                ---

                ### Jak to działa?

                **Rozmowa powinna potrwać co najmniej 3 minuty**. W trakcie których przycisk zakończenia badania będzie zablokowany.  
                Po 3 minutach pojawi się przycisk **Przejdź do oceny rozmowy** – od tego momentu możesz zakończyć rozmowę w dowolnej chwili lub kontynuować ją maksymalnie do 10 minut.  
                Po zakończeniu rozmowy poprosimy Cię o wypełnienie ostatniej części badania.
                
                ---

                ### Podpowiedź: Czego możesz się dowiedzieć?
                Jeśli nie wiesz, od czego zacząć, możesz zapytać na przykład o:
                * *Jaki jest główny cel tej petycji?*
                * *Jakie są kluczowe argumenty za jej przyjęciem?*
                * *Jakie zmiany prawne są proponowane?*
                * *Jakie są potencjalne korzyści dla zwierząt?*
                * *Jakie konkretnie problemy ma rozwiązać?*
                * *Poproszę o streszczenie najważniejszych argumentów.*
                * *Co pojawi się w ustawie?*

                Gdy wszystko będzie jasne, kliknij przycisk poniżej. Powodzenia!
            """, unsafe_allow_html=True)

            if st.button("Rozpocznij rozmowę z asystentem"):
                # Inicjalizacja stanu dla nowej rozmowy
                st.session_state.chat_started = True
                st.session_state.timer_active = False
                st.session_state.chat_input_disabled = False
                st.session_state.conversation_history = [
                    {"user": None, "bot": DEFAULT_PROMPTS.get(st.session_state.group, {}).get("welcome", "Witaj!")}
                ]
                st.session_state.shown_sentences = {0: False}  # Flagi wyświetlenia dla opóźnionych zdań bota
                st.session_state.timer_start_time = None
                st.session_state.conversation_end_time = None
                st.session_state.num_user_messages = 0
                st.session_state.num_bot_messages = 1  # Liczymy powitanie
            return  # Przerwij renderowanie, by po kliknięciu przycisku załadować widok czatu

        # 3B) Gdy rozmowa już się rozpoczęła, wyświetl panel czatu
        st.header("Rozmowa z asystentem AI")

        # --- Styl czatu za pomocą CSS ---
        st.markdown("""
        <style>
        .chat-container { max-height: 60vh; overflow-y: auto; margin-bottom: 10px; }
        .chat-user { display: flex; justify-content: flex-end; margin: 5px 0; }
        .chat-user > div {
            background-color: #4169E1;
            color: white;
            padding: 10px 15px;
            border-radius: 12px;
            max-width: 60%;
        }
        .chat-bot { display: flex; justify-content: flex-start; margin: 5px 0; }
        .chat-bot > div {
            background-color: #7D3C98;
            color: white;
            padding: 10px 15px;
            border-radius: 12px;
            max-width: 60%;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- 1) Wyświetl historię konwersacji ---
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        bot_name = DEFAULT_PROMPTS.get(st.session_state.group, {}).get("name", "Asystent")
        for i, turn in enumerate(st.session_state.conversation_history):
            # Wiadomość użytkownika
            if turn.get("user") is not None:
                st.markdown(f"<div class='chat-user'><div>{turn['user']}</div></div>", unsafe_allow_html=True)
            # Wiadomość bota (asystenta)
            if turn.get("bot") is not None:
                st.markdown(f"**{bot_name}**", unsafe_allow_html=True)
                # Bot może mieć listę zdań do wyświetlenia z opóźnieniem
                bot_sentences = turn["bot"] if isinstance(turn["bot"], list) else [turn["bot"]]
                if st.session_state.shown_sentences.get(i, False):
                    # Jeśli już wyświetliliśmy tę turę wcześniej, pokaż wszystkie zdania od razu
                    for sentence in bot_sentences:
                        st.markdown(f"<div class='chat-bot'><div>{sentence}</div></div>", unsafe_allow_html=True)
                else:
                    # Inaczej wyświetlamy zdania z opóźnieniem (40ms na znak)
                    for sentence in bot_sentences:
                        st.markdown(f"<div class='chat-bot'><div>{sentence}</div></div>", unsafe_allow_html=True)
                        time.sleep(len(sentence) * 0.02)
                    # Oznacz tę turę jako wyświetloną
                    st.session_state.shown_sentences[i] = True
        st.markdown("</div>", unsafe_allow_html=True)

        # --- 2) Pole do wpisywania wiadomości ---
        # Placeholder dynamiczny: pierwsza wiadomość vs kolejne
        if len(st.session_state.conversation_history) <= 1:
            prompt_text = "Proszę wpisać pierwszą wiadomość, aby rozpocząć konwersację..."
        else:
            prompt_text = "Proszę wpisać wiadomość..."
        user_input = st.chat_input(
            prompt_text,
            key="chat_input",
            disabled=st.session_state.get("chat_input_disabled", False)
        )

        # --- 3) Timer i przycisk „Przejdź do oceny rozmowy” ---
        timer_col, button_col = st.columns([1, 1])
        # with timer_col:
        #     if st.session_state.timer_active and st.session_state.timer_start_time:
        #         elapsed = datetime.now() - st.session_state.timer_start_time
        #         if elapsed < timedelta(minutes=3):
        #             rem = timedelta(minutes=3) - elapsed
        #             disp = f"Pozostało: {rem.seconds//60:02d}:{rem.seconds%60:02d}"
        #         elif elapsed < timedelta(minutes=10):
        #             extra = elapsed - timedelta(minutes=3)
        #             disp = f"+{extra.seconds//60:02d}:{extra.seconds%60:02d}"
        #         else:
        #             disp = "+07:00"
        #         st.markdown(f"Czas: **{disp}**")
        #     else:
        #         st.markdown("Czas: **––:––**")


        with button_col:
            if st.session_state.timer_active and st.session_state.timer_start_time:
                elapsed = datetime.now() - st.session_state.timer_start_time

                # Po 3 minutach:
                if elapsed >= timedelta(minutes=3) and elapsed < timedelta(minutes=10):
                    if st.button("Przejdź do oceny rozmowy"):
                        # → 1) Zanim przejdziemy dalej, nadpisujemy aktualny wiersz
                        row_idx = st.session_state.get("row_index")
                        if row_idx:
                            sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                            full_row = build_full_row_data()
                            sheet.update(f"A{row_idx}:AP{row_idx}", [full_row])

                        go_to(4)

                # Po 10 minutach:
                elif elapsed >= timedelta(minutes=10):
                    st.session_state.chat_input_disabled = True
                    st.markdown("**Czas rozmowy upłynął.**")
                    if st.button("Przejdź do oceny rozmowy"):
                        # → 2) Gdy czas się skończył, też zapisujemy wiersz
                        row_idx = st.session_state.get("row_index")
                        if row_idx:
                            sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                            full_row = build_full_row_data()
                            sheet.update(f"A{row_idx}:AP{row_idx}", [full_row])

                        go_to(4)

        # --- 4) Obsługa wpisania wiadomości przez użytkownika ---
        if user_input and not st.session_state.get("chat_input_disabled", False):
            # 4.1) Dodaj wiadomość użytkownika do historii
            st.session_state.conversation_history.append({"user": user_input, "bot": None})
            st.session_state.num_user_messages += 1

            # 4.2) Uruchom timer przy pierwszej wiadomości (pierwsza wiadomość to indeks 1)
            if not st.session_state.timer_active and len(st.session_state.conversation_history) == 2:
                st.session_state.timer_start_time = datetime.now()
                st.session_state.timer_active = True
                st.session_state.conversation_end_time = (
                    st.session_state.timer_start_time + timedelta(minutes=10)
                )

            # 4.3) Ustaw flagę procesowania odpowiedzi bota i odśwież widok
            st.session_state.process_user_input = True
            st.rerun()

        # --- 5) Generowanie odpowiedzi asystenta po ustawieniu process_user_input ---
        if st.session_state.get("process_user_input", False):
            st.session_state.process_user_input = False

            # Placeholder „pisanie...” dla bota
            bot_response_placeholder = st.empty()
            bot_response_placeholder.markdown(f"**{bot_name}**", unsafe_allow_html=True)
            bot_response_placeholder.markdown("<div class='chat-bot'><div>[...]</div></div>", unsafe_allow_html=True)

            model_to_use = DEFAULT_MODEL
            system_prompt = DEFAULT_PROMPTS.get(st.session_state.group, {}).get("system_prompt", "")
            messages = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.conversation_history:
                if m.get("user") is not None:
                    messages.append({"role": "user", "content": m["user"]})
                if m.get("bot") is not None:
                    bot_content = ". ".join(m["bot"]) if isinstance(m["bot"], list) else m["bot"]
                    messages.append({"role": "assistant", "content": bot_content})

            try:
                # 5.1) Pobranie kontekstu RAG
                last_user_message = ""
                for m in reversed(st.session_state.conversation_history):
                    if m.get("user") is not None:
                        last_user_message = m["user"]
                        break
                rag_query = f"{last_user_message} pseudohodowle dobrostan zwierząt petycja"
                retrieved_context = search_rag(rag_query, k=TOP_K)
                context_string = "\n".join([f"- {doc}" for doc in retrieved_context])
                messages.insert(1, {
                    "role": "system",
                    "content": "Korzystaj TYLKO z poniższych fragmentów:\n" +
                            "\n".join(f"- {d}" for d in retrieved_context)
                })
                # 5.2) Wywołanie API OpenAI
                with st.spinner(""):
                    resp = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.4
                    )
                bot_text = resp.choices[0].message.content
                bot_response_placeholder.empty()

                # 5.3) Wywołanie API OpenAI bez zmian
                with st.spinner(""):
                    resp = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.4
                    )
                bot_text = resp.choices[0].message.content
                bot_response_placeholder.empty()



                # === PRZYWRACAMY ORYGINALNE PODZIELENIE ODPOWIEDZI NA ZDANIA ===
                import re
                # 1) Rozbij odpowiedź bota na zdania (regex tak jak wcześniej)
                sentences = re.findall(r'.+?[.!?](?=\s|$)', bot_text)

                # 2) Oczyść każde zdanie (usuń nadmiarowe kropki i spacje)
                cleaned = []
                for s in sentences:
                    s = s.strip()
                    s = re.sub(r'\.+$', '', s)
                    cleaned.append(s)
                sentences = cleaned

                # 3) Dodajemy całą listę 'sentences' jako jedną turę bota:
                #    (najnowszy wpis w historii to zawsze użytkownik – dopiszemy tam 'bot': [sentences])
                if st.session_state.conversation_history and \
                   st.session_state.conversation_history[-1].get("user") is not None:
                    st.session_state.conversation_history[-1]["bot"] = sentences
                else:
                    st.session_state.conversation_history.append({"user": None, "bot": sentences})

                # 4) Oznacz, że ta tura bota jeszcze NIE została wyświetlona
                last_index = len(st.session_state.conversation_history) - 1
                st.session_state.shown_sentences[last_index] = False

                # 5) Odśwież widok, by w następnym przebiegu pokazać pierwsze zdanie z listy
                st.rerun()


            except Exception as e:
                st.error(f"Wystąpił błąd podczas generowania odpowiedzi: {e}")
                error_message = f"Błąd: {e}"
                if (
                    st.session_state.conversation_history
                    and st.session_state.conversation_history[-1].get("bot") is None
                ):
                    st.session_state.conversation_history[-1]["bot"] = error_message
                else:
                    st.session_state.conversation_history.append({"user": None, "bot": error_message})
                st.rerun()

        # # --- Przycisk "Dalej" do przejścia do następnego kroku ---
        # st.button(
        #     "Dalej",
        #     key="next_3",
        #     on_click=go_to,
        #     args=(4,)
        # )
        # Usunięto return, aby umożliwić przejście do kolejnego kroku po kliknięciu "Dalej"




# =========================================
# ---- Krok 4: Ocena Chatbota (Skala BUS-11) ----
# =========================================



    if step == 4:
        st.header("")

        # --- Nowa informacja przed BUS-11 ---
        st.markdown("""
                    
        ## Część 3 z 3
                    
        **Zanim zakończysz badanie, prosimy o wypełnienie krótkiej ankiety dotyczącej chatbota.**"
        """)

        st.markdown("---")

        st.markdown("""
Prosimy o ocenę chatbota, z którym rozmawiałeś, na poniższej skali. Zaznacz liczbą przy poszczególnych stwierdzeniach, do jakiego stopnia zgadzasz się lub nie zgadzasz z każdym z nich.
        """)

        st.markdown("""
                    <table style="
                        width: 100%;
                        border-collapse: collapse;
                        margin: 1rem auto;
                        text-align: center;
                    ">
                    <thead>
                        <tr style="border-bottom:1px solid #ccc;">
                        <th style="padding: 8px; text-align: center;">Zdecydowanie się nie zgadzam</th>
                        <th style="padding: 8px; text-align: center;">Raczej się nie zgadzam</th>
                        <th style="padding: 8px; text-align: center;">Ani się zgadzam, ani się nie zgadzam</th>
                        <th style="padding: 8px; text-align: center;">Raczej się zgadzam</th>
                        <th style="padding: 8px; text-align: center;">Zdecydowanie się zgadzam</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                        <td style="padding: 8px; text-align: center;"><strong>1</strong></td>
                        <td style="padding: 8px; text-align: center;"><strong>2</strong></td>
                        <td style="padding: 8px; text-align: center;"><strong>3</strong></td>
                        <td style="padding: 8px; text-align: center;"><strong>4</strong></td>
                        <td style="padding: 8px; text-align: center;"><strong>5</strong></td>
                        </tr>
                    </tbody>
                    </table>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Pytania ze skali BUS-11 (zgodnie z feedbackiem użytkownika)
        bus_questions = [
            "Chatbot pomógł mi zrozumieć temat.",
            "Informacje dostarczone przez chatbota były przydatne.",
            "Interakcja z chatbotem była łatwa i intuicyjna.",
            "Nie napotkałem(-am) żadnych problemów w komunikacji z chatbotem.",
            "Chatbot komunikował się w przyjazny i zrozumiały sposób.",
            "Odpowiedzi chatbota były spójne i logiczne.",
            "Rozmowa z chatbotem była angażująca.",
            "Czułem(-am), że chatbot dostosowywał się do moich odpowiedzi.",
            "Czułem(-am), że mogę zaufać informacjom dostarczonym przez chatbota.",
            "Chatbot wydawał się wiarygodny.",
            "Czułbym(-abym) się komfortowo, korzystając z tego chatbota ponownie."
        ]

        # Slidery dla każdego pytania BUS-11 (stylizacja podobna do TIPI-PL)
        bus_answers = []
        st.markdown("Proszę oceń chatbota na poniższej skali:") # Add introductory text
        for i, question in enumerate(bus_questions):
            # Use columns to place question on the left and selectbox on the right
            col1, col2 = st.columns([0.8, 0.2])

            with col1:
                st.markdown(f"**{i+1}. {question}**") # Format question as numbered list

            with col2:
                # Using selectbox for 1-5 scale
                val = st.selectbox(
                    label=f"Ocena {i+1}", # Simplified label for selectbox
                    options=["–– wybierz ––", 1, 2, 3, 4, 5],
                    index=0, # Default to "–– wybierz ––"
                    key=f"bus_{i}",
                    label_visibility="collapsed" # Hide the label
                )
                bus_answers.append(val)


        # Zapisz odpowiedzi BUS-11 w session_state
        st.session_state.bus_answers = bus_answers

        # Check if all questions are answered
        all_bus_answered = all(answer != "–– wybierz ––" for answer in bus_answers)

        # Next button to go to Feedback page (Step 6)
        st.button(
            "Dalej",
            key="next_4",
            on_click=lambda: [
                # 1) Najpierw nadpisujemy wiersz aktualnymi danymi (w tym BUS-11)
                _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1.update(
                    f"A{st.session_state['row_index']}:AP{st.session_state['row_index']}",
                    [build_full_row_data()]
                ),
                # 2) Dopiero przechodzimy do kroku 5 (Decyzja o petycji)
                go_to(5)
            ],
            disabled=not all_bus_answered
        )




# =========================================
# ---- Krok 5: Decyzja o zapoznaniu się z petycją ----
# =========================================


    if step == 5:
        st.header("Decyzja o zapoznaniu się z petycją")

        # --- Nowa, bardziej angażująca notatka wprowadzająca ---
        st.markdown("""
        Serdecznie dziękujemy, że dotarłeś(-aś) już tak daleko!  
        W trakcie rozmowy poznaliśmy podstawowe argumenty i fakty dotyczące praw zwierząt.  
        Teraz masz możliwość zobaczyć pełną treść petycji na oficjalnej stronie jej organizatorów – tam znajdziesz:
        - Pełny tekst postulowanych zmian prawnych  
        - Dane kontaktowe autorów petycji  
        - Informacje o tym, jak możesz włączyć się w akcję (np. podpis, udostępnienie)  

        Jeśli chcesz zajrzeć do szczegółów, kliknij **„Tak, chcę podpisać petycję”**.  
        W razie gdybyś wolał(-a) od razu przejść do ankiety końcowej, wybierz **„Nie, przejdź do zakończenia badania”**.
        """)

        # Ustawiamy flagę, jeśli nie istniała wcześniej
        if "show_petition_link" not in st.session_state:
            st.session_state.show_petition_link = False


        def save_petition_yes():
            st.session_state.decision = "Tak"
            st.session_state.show_petition_link = True

            # → Nadpisanie wiersza, aby zapisać kolumnę AL="Tak"
            row_idx = st.session_state.get("row_index")
            if row_idx:
                sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                sheet.update(f"A{row_idx}:AP{row_idx}", [build_full_row_data()])

        def save_petition_no():
            st.session_state.decision = "Nie"

            # → Nadpisanie wiersza, aby zapisać kolumnę AL="Nie"
            row_idx = st.session_state.get("row_index")
            if row_idx:
                sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                sheet.update(f"A{row_idx}:AP{row_idx}", [build_full_row_data()])

            go_to(6)

        col_yes, col_no = st.columns(2)
        with col_yes:
            st.button(
                "Tak, chcę podpisać petycję",
                key="petition_yes",
                on_click=save_petition_yes
            )
        with col_no:
            st.button(
                "Nie, przejdź do zakończenia badania",
                key="petition_no",
                on_click=save_petition_no
            )

        # Jeżeli użytkownik wybrał "Tak", pokazujemy link i przycisk do przejścia do ankiety
        if st.session_state.get("decision") == "Tak" and st.session_state.get("show_petition_link", False):
            st.markdown("---")
            st.markdown("**Po wejściu w link BARDZO PROSIMY o powrót do OKNA BADANIA w celu jego ukończenia**")
            st.markdown("**Oto oficjalna strona petycji:**")
            st.markdown("[Kliknij tutaj, aby przejść do strony odpowiedzialne za petycję](https://prawadlazwierzat.pl/)", unsafe_allow_html=True)
            st.markdown("""
            Na tej stronie znajdziesz kompletne informacje o celach petycji, autorach i sposobach wsparcia akcji.  
            Jeżeli chcesz wrócić do ankiety końcowej po zapoznaniu się z treścią, kliknij przycisk poniżej.
            """)
            if st.button("Przejdź do ankiety końcowej"):
                go_to(6)



# =========================================
# ---- Krok 6: Feedback ----
# =========================================



    if step == 6:
        st.header("🗣️ Podziel się wrażeniami z rozmowy!")
        st.markdown("---")
        # Zachęcający blok informacyjny
        
        st.markdown(
            """
            <p style="
                font-size: 24px;
                color: #bbb;
                line-height: 1.5;
                text-align: center;
                margin: 20px 0;
            ">
                Twoja opinia jest dla nas bardzo cenna, choć nie jest obowiązkowa.<br>
                Jeśli masz chwilę, napisz proszę, co zwróciło Twoją uwagę,<br>
                co warto poprawić, a co najbardziej Ci się spodobało.<br>
                Każda uwaga pomoże nam ulepszyć asystenta AI!<br>
                <br>
                <br>
                Na zakończenie badania prosimy koniecznie<br>
                <strong>o kliknięcie przycisku "Zakończ"</strong> poniżej.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Kolumny dla tekstów feedbacku, aby wyglądało bardziej przejrzyście
        col_pos, col_neg = st.columns(2)

        with col_neg:
            st.subheader("❌ Co można poprawić?")
            feedback_negative = st.text_area(
                "Opisz, co nie działało tak, jak byś chciał(a).",
                placeholder="Napisz tutaj swoje uwagi...",
                key="feedback_negative"
            )

        with col_pos:
            st.subheader("✅ Co Ci się podobało?")
            feedback_positive = st.text_area(
                "Podziel się, co najbardziej Ci się spodobało.",
                placeholder="Napisz tutaj, co było super...",
                key="feedback_positive"
            )

        # Save feedback to session_state
        st.session_state.feedback = {
            "negative": feedback_negative,
            "positive": feedback_positive
        }

        def finish():
            """
            Zbiera wszystkie dane z sesji i nadpisuje wiersz row_index (kolumny A–AN).
            """
            try:
                row_idx = st.session_state.get("row_index")
                if row_idx:
                    sheet = _gspread_client.open_by_key(GDRIVE_SHEET_ID).sheet1
                    full_row = build_full_row_data()
                    sheet.update(f"A{row_idx}:AP{row_idx}", [full_row])
                st.session_state.current_step = 7

            except Exception as e:
                st.error(f"Wystąpił błąd podczas zapisu danych do Arkusza Google: {e}")
                st.warning("Prosimy spróbować ponownie lub skontaktować się z administratorem.")
                return
            
        # zawsze pokazuj przycisk, nie tylko po błędzie
        st.button("Zakończ", key="finish", on_click=finish)

# =========================================
# ---- Krok 7: Ekran końcowy ----
# =========================================


    if step == 7:
        st.markdown(
            """
            # Dziękujemy za udział w badaniu! 🎉

            Twoje odpowiedzi zostały pomyślnie zapisane.  
            Jeśli masz ochotę, możesz teraz:

            - Zamknąć to okno  
            - Skontaktować się z nami: kfilewski@st.swps.edu.pl  
            """
        )
        # opcjonalnie jakieś grafiki, linki, itp.
        return

if __name__ == "__main__":
    main()
