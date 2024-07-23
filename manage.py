import typer

app = typer.Typer()


## -- CRON JOBS -- ##

@app.command()
def hello():
    print(f"Hello")

## MANAGE COMMANDS ##

@app.command()
def reset_chroma():
    """
    Resets the ChromaDB
    """
    from DB.chroma import chroma_client
    chroma_client.client.reset()
    typer.echo("ChromaDB reset successfully.")


if __name__ == "__main__":
    app()