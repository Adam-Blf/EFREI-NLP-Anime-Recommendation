"""
Script interactif pour explorer le système de recommandation d'anime.
Interface améliorée avec Rich pour une meilleure expérience utilisateur.
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
from rich import box

from data_load import load_data
from preprocess import preprocess_synopsis
from vectorize import vectorize_synopsis
from recommend import recommend_anime

console = Console()

def display_menu():
    menu_text = """
[bold cyan]1.[/bold cyan] Afficher les 5 premières lignes du jeu de données
[bold cyan]2.[/bold cyan] Obtenir des recommandations d'anime
[bold cyan]3.[/bold cyan] Afficher l'explication du modèle
[bold cyan]4.[/bold cyan] Quitter
    """
    console.print(Panel(menu_text, title="[bold magenta]Menu Principal[/bold magenta]", border_style="cyan", box=box.ROUNDED))

def display_dataframe(df, title="Aperçu des données"):
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    
    # Ajouter les colonnes
    for column in df.columns:
        table.add_column(str(column), style="cyan", no_wrap=False)
    
    # Ajouter les lignes
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])
        
    console.print(table)

def main() -> None:
    with console.status("[bold green]Chargement et préparation des données...[/bold green]", spinner="dots"):
        # Chargement et préparation des données
        df = load_data('Anime.csv')
        df = preprocess_synopsis(df)
        _, cosine_sim, indices = vectorize_synopsis(df)
    
    console.print("[bold green]✓ Données chargées avec succès![/bold green]\n")

    # Explication statique du modèle
    explanation = """
# Fonctionnement du Modèle

Ce système de recommandation est basé sur la **similarité du synopsis**.

1. **Vectorisation** : Chaque résumé est converti en vecteur TF‑IDF.
2. **Similarité** : La similarité cosinus entre ces vecteurs est calculée.
3. **Recommandation** : En fournissant une liste d'anime favoris, le programme combine les scores de similarité pour recommander les titres les plus proches.

> *Note : Aucun grand modèle de langage (LLM) externe n'est utilisé, car la connexion à des services distants n'est pas disponible.*
    """

    while True:
        display_menu()
        choice = Prompt.ask("Votre choix", choices=["1", "2", "3", "4"], default="1")

        if choice == '1':
            display_dataframe(df.head(5))
            
        elif choice == '2':
            console.print(Panel("Entrez les titres favoris séparés par des virgules\n(exactement comme dans la colonne 'Title')", title="Recommandation", border_style="green"))
            fav_input = Prompt.ask("[bold yellow]Titres favoris[/bold yellow]")
            
            favorites = [title.strip() for title in fav_input.split(',') if title.strip()]
            
            if not favorites:
                console.print("[bold red]Aucun favori fourni. Veuillez réessayer.[/bold red]")
                continue
                
            with console.status("[bold blue]Recherche de recommandations...[/bold blue]", spinner="earth"):
                top = recommend_anime(favorites, top_n=10, df=df, cosine_sim=cosine_sim, indices=indices)
            
            if top.empty:
                console.print("[bold red]Aucune recommandation disponible (vérifiez les titres saisis).[/bold red]")
            else:
                # Créer une table pour les résultats
                result_table = Table(title=f"Recommandations pour : {', '.join(favorites)}", box=box.DOUBLE_EDGE)
                result_table.add_column("Rang", style="magenta", justify="center")
                result_table.add_column("Titre", style="green")
                
                for idx, title in enumerate(top['Title'], 1):
                    result_table.add_row(str(idx), title)
                
                console.print(result_table)
                
        elif choice == '3':
            console.print(Panel(Markdown(explanation), title="Explication", border_style="blue"))
            
        elif choice == '4':
            console.print("[bold magenta]Au revoir ![/bold magenta]")
            sys.exit(0)

if __name__ == '__main__':
    main()
