print('caricamento librerie...')
# importazione librerie per la manipolazione dei dati
from datetime import datetime
import os
import pandas as pd
import re
from collections import Counter
import unicodedata

# importazione librerie per il clustering
import spacy
nlp = spacy.load("it_core_news_lg")
'''
installazione corpus in italiano per NLTK (small e large)
python -m spacy download it_core_news_sm
python -m spacy download it_core_news_lg
'''
import nltk
from nltk.stem.snowball import SnowballStemmer
import fasttext.util
from nltk.corpus import stopwords
nltk.download("stopwords")

# clustering vettoriale
import hdbscan
import numpy as np
from gensim.models import Word2Vec
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
print('librerie caricate...')

#############################

print('crezione o verifica delle cartelle...')
# Ottieni il percorso della directory dello script
current_directory = os.path.dirname(os.path.abspath(__file__))
print(f"Percorso corrente: {current_directory}")

# Nomi delle cartelle da verificare
folders = ["input", "output"]

# Verifica e crea le cartelle se necessario
for folder in folders:
    folder_path = os.path.join(current_directory, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Cartella '{folder}' creata.")
    else:
        print(f"Cartella '{folder}' esiste già.")

#############################

print('importazione file Excel...')
# Elenca i file nella cartella input
input_folder = os.path.join(current_directory, "input")
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

if not files:
    print("La cartella 'input' è vuota. Aggiungi un file Excel e riprova.")
    exit()

print("\nFile disponibili nella cartella 'input':")
for i, file in enumerate(files, 1):
    print(f"{i}. {file}")

# Seleziona il file da aprire
while True:
    try:
        file_index = int(input("\nSeleziona il numero del file da aprire: ")) - 1
        if 0 <= file_index < len(files):
            selected_file = files[file_index]
            break
        else:
            print("Numero non valido. Riprova.")
    except ValueError:
        print("Inserisci un numero valido.")

selected_file_path = os.path.join(input_folder, selected_file)

# Importa il file Excel
excel_data = pd.ExcelFile(selected_file_path)
print("\nFogli disponibili nel file:")
for i, sheet in enumerate(excel_data.sheet_names, 1):
    print(f"{i}. {sheet}")

# Seleziona il foglio da importare
while True:
    try:
        sheet_index = int(input("\nSeleziona il numero del foglio da importare: ")) - 1
        if 0 <= sheet_index < len(excel_data.sheet_names):
            selected_sheet = excel_data.sheet_names[sheet_index]
            break
        else:
            print("Numero non valido. Riprova.")
    except ValueError:
        print("Inserisci un numero valido.")

# Leggi il foglio selezionato
sheet_data = excel_data.parse(selected_sheet)
print("\nColonne disponibili nel foglio:")
for i, column in enumerate(sheet_data.columns, 1):
    print(f"{i}. {column}")

# Seleziona la colonna da importare
while True:
    try:
        column_index = int(input("\nSeleziona il numero della colonna da importare: ")) - 1
        if 0 <= column_index < len(sheet_data.columns):
            selected_column = sheet_data.columns[column_index]
            break
        else:
            print("Numero non valido. Riprova.")
    except ValueError:
        print("Inserisci un numero valido.")

# Crea un DataFrame con la colonna selezionata
print("\nImportazione dei dati nel dataframe in corso...")
dataframe = sheet_data[[selected_column]].copy()
print('creazione della colonna keywords')
dataframe.columns = ["keywords"]
print('dropna per eliminare i valori nulli')
dataframe = dataframe.dropna()
print("\nDati importati:")
print(dataframe.head(10))
print('')
print('Dataframe pronto per il clustering.')

#############################

# Funzione per pulire le parole chiave
print('pulizia della lista')
def clean_keywords(df):
    # Contatore delle righe originali
    original_row_count = len(df)
    stop_words = set(stopwords.words("italian"))  # Stopword italiane
    # Crea una nuova colonna "Cleaned" con i valori ripuliti
    df["Cleaned"] = df["keywords"].str.lower()  # Converti in minuscolo
    df["Cleaned"] = df["Cleaned"].str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
    df["Cleaned"] = df["Cleaned"].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn'))
    df["Cleaned"] = df["Cleaned"].str.replace(r"\s+", " ", regex=True).str.strip()  # Rimuovi spazi extra

    # Funzione per rimuovere stopword
    def remove_stopwords(text):
        words = text.split()  # Divide in parole
        filtered_words = [word for word in words if word not in stop_words]  # Rimuovi stopword
        return " ".join(filtered_words)

    df["Cleaned"] = df["Cleaned"].apply(remove_stopwords)

    df["Cleaned"] = df["Cleaned"].str.strip()  # Rimuovi spazi iniziali
    df = df.drop_duplicates(subset=["Cleaned"]).reset_index(drop=True)  # Rimuovi duplicati
    df = df[df["Cleaned"].str.strip() != ""] # Rimuovi righe vuote
    
    # Contatore delle righe rimanenti
    final_row_count = len(df)
    # Stampa dei risultati
    print(f"Numero di righe originali: {original_row_count}")
    print(f"Numero di righe rimanenti: {final_row_count}")

    return df

# Pulisci le parole chiave
dataframe = clean_keywords(dataframe)
print("\nParole chiave pulite:")
print(dataframe.head(10))


#############################


print('stemmi')

#### stemmi
def apply_stemming_optimized(df):
    stemmer = SnowballStemmer("italian")
    cleaned_terms = df["Cleaned"].tolist()  # Converti in lista per operazioni più veloci
    stemmed_terms = []

    for term in cleaned_terms:
        # Stemma ogni parola del termine e ricrea la stringa
        stemmed_terms.append(" ".join(stemmer.stem(word) for word in term.split()))
    
    df["Stemmi"] = stemmed_terms  # Aggiungi la nuova colonna
    return df

# Applica lo stemming
dataframe = apply_stemming_optimized(dataframe)

print("\nParole chiave stemmate:")
print(dataframe[["Cleaned", "Stemmi"]].head(10))

############################# stemmi ordinati

print('stemmi ordinati')

def add_top_stems_columns_optimized(df):
    # Calcola le frequenze di tutti gli stemmi
    all_stems = " ".join(df["Stemmi"]).split()
    print('Calcolo delle frequenze degli stemmi...')
    stem_frequencies = Counter(all_stems)  # Conta la frequenza di ciascuno stemma
    print('Ordinamento degli stemmi...')
    # Ordina gli stemmi per frequenza (decrescente)
    sorted_stems = [stem for stem, _ in stem_frequencies.most_common()]
    print('Preparazione delle colonne con i top stemmi...')
    # Prepara le colonne con i top stemmi
    top_3_stems = []
    top_2_stems = []
    top_1_stem = []

    print('Iterazione sulle righe...')
    # Itera direttamente sulle righe della colonna "Stemmi"
    for stem_string in df["Stemmi"]:
        stems = stem_string.split()  # Dividi gli stemmi della riga
        # Filtra i top stemmi in base all'ordine globale
        top_3_stems.append(" ".join([stem for stem in sorted_stems if stem in stems][:3]))
        top_2_stems.append(" ".join([stem for stem in sorted_stems if stem in stems][:2]))
        top_1_stem.append(" ".join([stem for stem in sorted_stems if stem in stems][:1]))
    print('Aggiunta delle colonne al DataFrame...')
    # Aggiungi le nuove colonne al DataFrame
    df["3 top stems"] = top_3_stems
    df["2 top stems"] = top_2_stems
    df["top stems"] = top_1_stem

    return df

# Applica la funzione ottimizzata al DataFrame
print('Applicazione della funzione...')
dataframe = add_top_stems_columns_optimized(dataframe)

print("\nParole chiave con top stems:")
print(dataframe[["Cleaned", "Stemmi", "3 top stems", "2 top stems", "top stems"]].head(10))



############################# lemmi

print('lemmi')
def apply_lemmatization(df):
    # Ottieni la lista delle parole chiave ripulite
    cleaned_terms = df["Cleaned"].tolist()  
    print('Elaborazione batch con spaCy...')
    
    # Usa spaCy per elaborare il testo in batch
    docs = list(nlp.pipe(cleaned_terms, batch_size=1000))  # Processo batch (puoi regolare il batch_size)
    
    # Estrai i lemmi da ogni documento
    lemmatized_terms = [" ".join([token.lemma_ for token in doc]) for doc in docs]
    
    print('Aggiunta della colonna al DataFrame...')
    df["Lemmi"] = lemmatized_terms  # Aggiungi la nuova colonna
    return df

# Applica la lemmatizzazione
dataframe = apply_lemmatization(dataframe)

print("\nParole chiave con lemmi:")
print(dataframe[["Cleaned", "Lemmi"]].head(10))


############################# top lemma
print('Creazione della colonna top lemma...')

def add_top_lemmas_columns(df):
    # Calcola le frequenze di tutti i lemmi
    print("Calcolo delle frequenze dei lemmi...")
    all_lemmas = " ".join(df["Lemmi"]).split()
    lemma_frequencies = Counter(all_lemmas)  # Conta la frequenza di ciascun lemma
    print('Ordinamento dei lemmi...')
    # Ordina i lemmi per frequenza (decrescente)
    sorted_lemmas = [lemma for lemma, _ in lemma_frequencies.most_common()]
    print('Preparazione delle colonne con i top lemmi...')
    # Prepara le colonne con i top lemmi
    top_3_lemmas = []
    top_2_lemmas = []
    top_1_lemma = []

    print('Iterazione sulle righe...')
    # Itera direttamente sulle righe della colonna "Lemmi"
    for lemma_string in df["Lemmi"]:
        lemmas = lemma_string.split()  # Dividi i lemmi della riga
        # Filtra i top lemmi in base all'ordine globale
        top_3_lemmas.append(" ".join([lemma for lemma in sorted_lemmas if lemma in lemmas][:3]))
        top_2_lemmas.append(" ".join([lemma for lemma in sorted_lemmas if lemma in lemmas][:2]))
        top_1_lemma.append(" ".join([lemma for lemma in sorted_lemmas if lemma in lemmas][:1]))

    print('Aggiunta delle colonne al DataFrame...')
    # Aggiungi le nuove colonne al DataFrame
    df["3 top lemmas"] = top_3_lemmas
    df["2 top lemmas"] = top_2_lemmas
    df["top lemma"] = top_1_lemma

    return df

# Applica la funzione al DataFrame
print('Applicazione della funzione per i top lemmi...')
dataframe = add_top_lemmas_columns(dataframe)

print("\nParole chiave con top lemmi:")
print(dataframe[["Cleaned", "Lemmi", "3 top lemmas", "2 top lemmas", "top lemma"]].head(10))


############################# fasttextimport fasttext

print('fasttext')
# Scarica e carica il modello FastText per l'italiano
print("Scaricamento del modello FastText per l'italiano...")  # Questo può richiedere tempo
fasttext.util.download_model('it', if_exists='ignore')  # Italiano
print("Caricamento del modello FastText per l'italiano...")
ft = fasttext.load_model('cc.it.300.bin')  # Carica il modello scaricato

def calculate_embeddings_fasttext(df):
    embeddings = []
    for term in df["Cleaned"]:
        words = term.split()  # Dividi in parole
        word_vectors = [
            ft.get_word_vector(word) for word in words
        ]  # Ottieni il vettore per ogni parola
        if word_vectors:
            # Calcola la media dei vettori
            avg_vector = sum(word_vectors) / len(word_vectors)
        else:
            # Usa un vettore di zeri se nessuna parola è nel modello
            avg_vector = [0] * ft.get_dimension()
        embeddings.append(avg_vector)
    df["emb ft"] = embeddings
    return df

# Calcola gli embeddings e aggiungi la colonna
print("Calcolo degli embeddings FastText in corso...")
dataframe = calculate_embeddings_fasttext(dataframe)

print("\nEsempi di embeddings calcolati:")
print(dataframe[["Cleaned", "emb ft"]].head(3))



############################# clustering FT
print('clustering fasttext')
# analizza 1000 blocchi per volta
n_blocks = 1000
def cluster_embeddings_with_hdbscan(df, n_blocks=n_blocks):
    # Estrai gli embeddings dalla colonna "emb ft" come matrice numpy
    print('Estrazione degli embeddings...')
    embeddings = np.array(df["emb ft"].tolist())

    # Suddividi il dataset in blocchi
    block_size = len(embeddings) // n_blocks
    all_clusters = np.full(len(embeddings), -1, dtype=int)  # Array per i cluster assegnati

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else len(embeddings)
        block = embeddings[start:end]
        
        # Stampa il progresso
        print(f"Processando blocco {i + 1}/{n_blocks} (da {start} a {end})...")

        # Esegui il clustering sul blocco corrente
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, core_dist_n_jobs=-1)
        block_clusters = clusterer.fit_predict(block)

        # Assegna i cluster al blocco corrispondente
        all_clusters[start:end] = block_clusters

    # Aggiungi i cluster al DataFrame
    print('Aggiunta dei cluster fasttext al DataFrame...')
    df["FT Cluster"] = all_clusters

    return df

# Applica la clusterizzazione al DataFrame
dataframe = cluster_embeddings_with_hdbscan(dataframe, n_blocks=n_blocks)

print("\nCluster ft assegnati:")
print(dataframe[["Cleaned", "FT Cluster"]].head(10))

############################# w2v
print('word2vec')

# Addestra il modello Word2Vec sul testo delle parole chiave
def train_word2vec_and_calculate_embeddings(df):
    # Prepara i dati per Word2Vec (tokenizza le frasi)
    print("Tokenizzazione delle frasi in corso...")
    tokenized_sentences = [text.split() for text in df["Cleaned"]]

    # Addestra il modello Word2Vec
    print("Addestramento del modello Word2Vec in corso...")
    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=5, min_count=1, workers=4)

    # Calcola gli embeddings medi per ogni riga
    embeddings = []
    print("Calcolo degli embeddings medi per ogni riga...")
    for tokens in tokenized_sentences:
        word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if word_vectors:
            avg_vector = np.mean(word_vectors, axis=0)
        else:
            avg_vector = np.zeros(w2v_model.vector_size)  # Vettore di zeri se nessuna parola è nel modello
        embeddings.append(avg_vector)

    df["w2v"] = embeddings
    return df, w2v_model

# Addestra Word2Vec e calcola gli embeddings
dataframe, w2v_model = train_word2vec_and_calculate_embeddings(dataframe)

print("\nEsempi di embeddings Word2Vec calcolati:")
print(dataframe[["Cleaned", "w2v"]].head(3))



############################# clustering w2v
print('clustering word2vec')
# analizza 1000 blocchi per volta
n_blocks = 1000
def cluster_word2vec_with_hdbscan(df, n_blocks=n_blocks):
    # Estrai gli embeddings dalla colonna "w2v" come matrice numpy
    print("Estrazione degli embeddings Word2Vec...")
    embeddings = np.array(df["w2v"].tolist())

    # Calcola il blocco
    block_size = len(embeddings) // n_blocks
    all_clusters = np.full(len(embeddings), -1, dtype=int)  # Inizializza array per i cluster assegnati

    # Clusterizza per blocchi
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else len(embeddings)
        block = embeddings[start:end]
        
        # Stampa il progresso
        print(f"Processando blocco {i + 1}/{n_blocks} (da {start} a {end})...")

        # Esegui il clustering sul blocco corrente
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        block_clusters = clusterer.fit_predict(block)

        # Assegna i cluster al blocco corrispondente
        all_clusters[start:end] = block_clusters

    # Aggiungi i cluster al DataFrame
    print("Aggiunta dei cluster al DataFrame...")
    df["W2V Cluster"] = all_clusters

    return df

# Applica la clusterizzazione
dataframe = cluster_word2vec_with_hdbscan(dataframe, n_blocks=n_blocks)

print("\nCluster assegnati ai vettori Word2Vec:")
print(dataframe[["Cleaned", "W2V Cluster"]].head(10))


############################# bertopic

print('bertopic')

def calculate_bertopic_clusters_with_labels(df):
    # Modello di embedding specifico per l'italiano
    print("Caricamento del modello di embedding...")

    # MODELLI SPECIFICI ITALIANO
    # compatibile con sentence transformer
    # embedding_model = SentenceTransformer("dbmdz/bert-base-italian-uncased") # Modello di dimensioni medie, ben bilanciato tra prestazioni e complessità computazionale.
    # embedding_model = SentenceTransformer('pritamdeka/MDEBERTA-V3-base-italian') # Modello di dimensioni medie.
    # embedding_model = SentenceTransformer('nickprock/sentence-bert-base-italian-uncased') # Modello di dimensioni medie specificamente progettato per generare rappresentazioni di frasi (sentence embeddings).

    # da testare
    # embedding_model = SentenceTransformer("Musixmatch/umberto-commoncrawl-cased-v1") # accurato, modello più grande per l'italiano.
    
    # MODELLI MULTILINGUA
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") # multilingual sentence embeddings.
    # embedding_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1') # fast, efficiency and good performance across multiple languages.
    
    # Genera gli embeddings per le frasi della colonna "Cleaned"
    print("Calcolo degli embeddings con SentenceTransformer...")
    embeddings = embedding_model.encode(df["Cleaned"].tolist(), show_progress_bar=True)

    # Inizializza BERTopic
    print("Creazione del modello BERTopic...")
    topic_model = BERTopic(language="Italian")
  
    # Applica BERTopic per calcolare i cluster
    print("Esecuzione di BERTopic...")
    topics, _ = topic_model.fit_transform(df["Cleaned"].tolist(), embeddings)

    # Genera una mappatura dei topic numerici e delle relative descrizioni
    topic_labels = {topic: " ".join([word for word, _ in topic_model.get_topic(topic)]) 
                    if topic != -1 else "No Topic" 
                    for topic in set(topics)}

    # Aggiungi la descrizione del topic alla colonna "Bertopic"
    df["Bertopic"] = [topic_labels[topic] for topic in topics]
    return df, topic_model

# Calcola i cluster BERTopic e aggiungi la colonna con le descrizioni
dataframe, bertopic_model = calculate_bertopic_clusters_with_labels(dataframe)

print("\nCluster assegnati con BERTopic:")
print(dataframe[["Cleaned", "Bertopic"]].head(10))


############################# export

print('esportazione')
# Genera il nome del file basato sulla data e ora attuali
print('Generazione del nome del file...')
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
print(f"Timestamp: {timestamp}")
output_file_name = f"cluster_{timestamp}.xlsx"
print(f"Nome del file: {output_file_name}")
# Percorso completo per salvare il file
output_file_path = os.path.join(current_directory, "output", output_file_name)
print(f"Percorso del file: {output_file_path}")
# Esporta il DataFrame in Excel
print("Esportazione del DataFrame in corso...")
#dataframe.to_excel(output_file_path, index=False)
dataframe.to_excel(output_file_path, index=False, engine='xlsxwriter')
print(f"\nFile esportato con successo: {output_file_path}")
