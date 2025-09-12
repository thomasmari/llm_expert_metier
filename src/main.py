import uvicorn

def main():

    # Affichage
    print("🚀🚀🚀🚀 Lancement de l'application : 🚀🚀🚀🚀 ")
    print("💡 Vous pourrez l'ouvrir avec votre navigateur sur http://localhost:8000 💡")

    # Run the server
    uvicorn.run(
        "interface:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
