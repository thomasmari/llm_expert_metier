import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""On suit ici le tuto : https://python.langchain.com/docs/tutorials/retrievers/""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Paramétrage de l'API LongChain""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # On va utiliser LongChain ici :

    import getpass
    import os

    # Pour gérer l'API de Longchain :
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
    return getpass, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Tests avec des documents générés sur place""")
    return


@app.cell
def _():
    # Création de documents pour le test : 
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    print(documents)
    print(len(documents))
    print(documents[0])
    return (Document,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Test d'import de documents avec PyPDFLoader""")
    return


@app.cell
def _():
    # Mais on peut aussi (et surtout en fait) importer des documents !

    from langchain_community.document_loaders import PyPDFLoader

    file_path = "data/Code_penal.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    print(len(docs))

    # On affiche les 200 premiers caractères de la première page :
    print(f"{docs[0].page_content[:200]}\n")

    # Et les métadonnées :
    print(docs[0].metadata)
    return (docs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Splitting""")
    return


@app.cell
def _(docs):
    # Splitting du document

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(len(all_splits))
    return (all_splits,)


@app.cell
def _(all_splits):
    print(all_splits[56])
    return


@app.cell
def _(all_splits):
    print(all_splits[1423])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Embeddings

    Ici on teste avec l'API Google Gemini donc il faut une clé API Google
    """
    )
    return


@app.cell
def _(getpass, os):
    #import getpass
    #import os

    if not os.environ.get("GOOGLE_API_KEY"):
      os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return (embeddings,)


@app.cell
def _(all_splits, embeddings):
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Vector Stores""")
    return


@app.cell
def _(embeddings):
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    return (vector_store,)


@app.cell
def _(all_splits, vector_store):
    # Having instantiated our vector store, we can now index the documents.
    ids = vector_store.add_documents(documents=all_splits[1000:(1000+5)]) # 5 pour pas cramer mes limites d'utilisation de l'API
    return


@app.cell
def _(vector_store):
    results = vector_store.similarity_search(
        "A qui est applicable le code pénal ?"
    )

    print(results[0])
    return (results,)


@app.cell
async def _(vector_store):
    # On peut faire un requête asynchrone :
    results_bis = await vector_store.asimilarity_search("A qui est applicable le code pénal ?")
    return (results_bis,)


@app.cell
def _(results_bis):
    for res in results_bis:
        print("-------------------")
        print(res)
        print("-------------------")
    return


@app.cell
def _(vector_store):
    # On peut aussi afficher le score d ela requête :

    # Note that providers implement different scores; the score here
    # is a distance metric that varies inversely with similarity.

    results_ter = vector_store.similarity_search_with_score("C'est quoi la loi ?")
    doc, score = results_ter[0]
    print(f"Score: {score}\n")
    print(doc)
    print("--")
    print(doc.metadata)
    return


@app.cell
def _(embeddings, results, vector_store):
    # Return documents based on similarity to an embedded query:
    my_embedding = embeddings.embed_query("Quelle est la peine de prison maximale pour meurtre ?")

    my_results = vector_store.similarity_search_by_vector(my_embedding)
    print(results[0])
    return


@app.cell
def _(Document, vector_store):
    # On peut aussi faire des traitement par batch

    from typing import List

    from langchain_core.runnables import chain


    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)


    resultats = retriever.batch(
        [
            "Quelle est la peine de prison maximale pour meurtre ?",
            "C'est quoi la loi ?",
        ],
    )

    print(resultats[0])
    print(resultats[1])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
