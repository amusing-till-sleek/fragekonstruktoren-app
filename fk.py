import streamlit as st
from openai import OpenAI  # Updated import per migration
import pdfplumber
import docx
import json

# ---- Page Configuration ----
st.set_page_config(page_title="Frågekonstruktören", layout="centered")
st.title("Frågekonstruktören")

# ---- Reset Button ----
if st.button("Starta om"):
    # Clear session state keys except the API key
    for key in ["faktabas", "larandemal_och_indikatorer", "mcqs"]:
        st.session_state.pop(key, None)
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---- Utility Functions ----
def get_api_key():
    """Retrieve and store the OpenAI API key in session state."""
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    st.session_state["api_key"] = st.text_input(
        "OpenAI API-nyckel", type="password", value=st.session_state["api_key"]
    )
    return st.session_state["api_key"]

def extract_text(file) -> str:
    """Extract text from an uploaded .txt, .pdf, or .docx file."""
    text = ""
    try:
        if file.name.endswith(".txt"):
            text = file.getvalue().decode("utf-8")
        elif file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Fel vid extrahering av text: {e}")
    return text

def create_openai_client():
    """Instantiate and return an OpenAI client with the API key."""
    return OpenAI(api_key=st.session_state["api_key"])

# ---- Core Functions ----
def generate_learning_objectives(faktabas: str) -> list:
    """
    Analyze the fact base and generate a specified number of learning objectives 
    (lärandemål) along with a complete reference excerpt and up to 5 indicators each.
    
    Each learning objective must be formulated using one of Bloom's taxonomy verbs
    ("Lista", "Återge", or "Redogör för"). The 'referens' field should contain a
    full, coherent excerpt from the fact base – at least one complete paragraph (min. ~100 words)
    – that directly supports the objective and its indicators.

    Returns a JSON array.
    """
    client = create_openai_client()
    text_length = len(faktabas)
    num_goals = min(max(text_length // 1000, 3), 8)

    learning_objectives_prompt = f"""
Analysera den nedanstående faktabasen och identifiera {num_goals} relevanta lärandemål baserade på innehållet.
Varje lärandemål ska formuleras med ett verb enligt Bloom's taxonomi: "Lista", "Återge", eller "Redogör för".

För varje lärandemål, ange följande fält:
- "larandemal": En kort titel/formulerat lärandemål som innehåller ett av verben.
- "indikatorer": En lista med upp till 5 stödord eller korta meningar.
- "referens": Ett fullständigt, sammanhängande textavsnitt från faktabasen (minst 100 ord och upp till 200 ord) 
  som tydligt visar hur lärandemålet och dess indikatorer stöds av texten. Inkludera en eller två kompletta stycken.

Returnera endast ett JSON‐array i följande struktur (inga extra texter utanför JSON):
[
  {{
    "larandemal": "Titel med Bloom-verb",
    "indikatorer": ["Indikator 1", "Indikator 2", ...],
    "referens": "Fullständig text som stödjer lärandemålet och indikatorerna"
  }},
  ...
]

**Faktabas:**
{faktabas}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du är expert på att skapa pedagogiska lärandemål och indikatorer enligt Bloom's taxonomi. "
                        "Följ anvisningarna noggrant och inkludera ett fullständigt textavsnitt (minst 100 ord) som referens."
                    )
                },
                {"role": "user", "content": learning_objectives_prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()

        if not raw_output:
            st.error("API:s svar var tomt.")
            return []

        try:
            learning_objectives = json.loads(raw_output)
        except Exception as json_err:
            st.error("Fel vid tolkning av JSON-svar från API: " + str(json_err))
            st.write("Rå output från API:", raw_output)
            return []

        return learning_objectives
    except Exception as e:
        st.error(f"Fel vid generering av lärandemål och indikatorer: {e}")
        return []

def generate_mcq(larandemal: str, indikatorer: list, faktabas: str) -> list:
    """
    Generate 4 multiple-choice questions (MCQs) for the given learning objective.
    For each question, include:
      - A carefully formulated question that does not directly reveal the correct answer.
      - Attractive, challenging, and plausible distractors that may be slightly longer.
      - A short explanation of why the correct answer is right.
      - A complete reference excerpt (one or two full paragraphs, at least 100 words and up to 200 words)
        from the fact base that clearly shows where the correct answer is supported.
    
    Return only a JSON array in the following structure (no extra text outside JSON):
    [
      {{
        "fraga": "Frågetext",
        "ratt_svar": "Det rätta svaret",
        "distraktorer": ["Alternativ 1", "Alternativ 2", "Alternativ 3"],
        "forklaring": "Förklaringstext",
        "referens": "Fullständig text som visar var i faktabasen svaret framgår"
      }},
      ...
    ]
    """
    client = create_openai_client()

    mcq_prompt = f"""
Skapa 4 flervalsfrågor för det angivna lärandemålet med de angivna indikatorerna.
För varje fråga, ange följande fält:
- "fraga": En **välformulerad fråga** som inte kan besvaras enbart genom att känna igen nyckelord; testar förståelse genom att kräva analys jämförelse eller tillämpning av kunskap; gärna använder **ett scenario eller ett exempel** om det är relevant.
- "ratt_svar": Det rätta svaret.
- "distraktorer": En lista med 3 alternativa svar som är **Utmanande men trovärdiga**, dvs. varje distraktor ska vara baserad på vanliga missförstånd, nära korrekta tolkningar eller alternativ som verkar rimliga vid en snabb analys; **jämförbara i längd och komplexitet** med det rätta svaret, dvs. de ska inte vara avsevärt kortare eller enklare; **strategiskt utvalda för att testa förståelse**, t.ex. genom att använda ett felaktigt resonemang, en vanlig men felaktig förenkling eller en felaktig tillämpning av fakta.
- "forklaring": En kort, men **pedagogisk och analytisk förklaring** som förklarar **varför det rätta svaret är korrekt**; förklarar **varför varje distraktor är fel**, gärna genom att peka ut specifika fel eller missförstånd. Förklaringen får inte använda termen 'distraktor', utan ska istället använda omskrivningar som 'de andra svaren' eller liknande.
- "referens": En **klar och tydlig referens** (minst 100 ord) från faktabasen som Innehåller **både bakgrundsinformation och direkt stöd för svaret**; Gör det enkelt att förstå varför det rätta svaret är rätt.

Returnera endast ett JSON-array i följande form (inga extra texter utanför JSON):
[
  {{
    "fraga": "Frågetext",
    "ratt_svar": "Det rätta svaret",
    "distraktorer": ["Alternativ 1", "Alternativ 2", "Alternativ 3"],
    "forklaring": "Förklaringstext",
    "referens": "Fullständig text som visar var i faktabasen svaret framgår"
  }},
  ...
]

**Lärandemål:**
{larandemal}

**Indikatorer:**
{"\n".join(indikatorer)}

**Faktabas:**
{faktabas}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Du är expert på att skapa pedagogiska flervalsfrågor enligt Bloom's taxonomi. "
                    "Skapa frågor med attraktiva, utmanande distraktorer som är realistiska men inte uppenbara. "
                    "Inkludera ett fullständigt textavsnitt (minst 100 ord) som referens."
                )},
                {"role": "user", "content": mcq_prompt}
            ],
            temperature=0.1,
            max_tokens=2500
        )
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            lines = raw_output.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()

        mcqs = json.loads(raw_output)
        return mcqs
    except Exception as e:
        st.error(f"Fel vid generering av MCQs: {e}")
        return []

# ---- Main Application Flow ----
def main():
    api_key = get_api_key()
    if not api_key:
        st.warning("Vänligen ange din OpenAI API-nyckel.")
        return

    st.subheader("Ladda upp en faktabas")
    uploaded_file = st.file_uploader("Ladda upp en .txt, .pdf eller .docx-fil", type=["txt", "pdf", "docx"])
    
    if uploaded_file:
        faktabas = extract_text(uploaded_file)
        if faktabas:
            st.session_state["faktabas"] = faktabas
            st.success(f"Filen {uploaded_file.name} har laddats upp!")
        else:
            st.error("Kunde inte extrahera text från filen.")

    if "faktabas" in st.session_state and st.button("Generera lärandemål och indikatorer"):
        with st.spinner("Analyserar faktabasen och genererar lärandemål och indikatorer ..."):
            learning_objectives = generate_learning_objectives(st.session_state["faktabas"])
            st.session_state["larandemal_och_indikatorer"] = learning_objectives

    # Display Learning Objectives
    if "larandemal_och_indikatorer" in st.session_state and st.session_state["larandemal_och_indikatorer"]:
        st.subheader("Genererade lärandemål och indikatorer")
        for obj in st.session_state["larandemal_och_indikatorer"]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Referens i faktabas:**")
                st.write(obj.get("referens", ""))
            with col2:
                st.markdown("**Lärandemål:** " + obj.get("larandemal", ""))
                st.markdown("**Indikatorer:**")
                for ind in obj.get("indikatorer", []):
                    st.write(f"- {ind}")
            st.write("---")

        if st.button("Generera flervalsfrågor"):
            with st.spinner("Genererar flervalsfrågor ..."):
                mcqs_all = {}
                for obj in st.session_state["larandemal_och_indikatorer"]:
                    larandemal = obj.get("larandemal", "")
                    indikatorer = obj.get("indikatorer", [])
                    mcqs = generate_mcq(larandemal, indikatorer, st.session_state["faktabas"])
                    mcqs_all[larandemal] = mcqs
                st.session_state["mcqs"] = mcqs_all

    if "mcqs" in st.session_state:
        st.subheader("Genererade flervalsfrågor")
        for larandemal, mcq_list in st.session_state["mcqs"].items():
            st.markdown(f"### {larandemal}")
            for mcq in mcq_list:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Referens i faktabasen:**")
                    st.write(mcq.get("referens", ""))
                with col2:
                    st.markdown("**Fråga:**")
                    st.write(mcq.get("fraga", ""))
                    st.markdown("**Rätt svar:**")
                    st.write(mcq.get("ratt_svar", ""))
                    st.markdown("**Distraktorer:**")
                    for d in mcq.get("distraktorer", []):
                        st.write(f"- {d}")
                    st.markdown("**Förklaring:**")
                    st.write(mcq.get("forklaring", ""))
                st.write("---")

if __name__ == "__main__":
    main()
