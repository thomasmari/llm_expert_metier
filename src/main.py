import uvicorn

def main():
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
