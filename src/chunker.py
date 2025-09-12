# Author : Claude.ai, itération 3 avec débug de Xavier Bednarek
# Date   : 2025-09-09

import re
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class CodePenalChunker:
    """
    Chunker intelligent pour le Code pénal français qui respecte la structure juridique
    et extrait les métadonnées pertinentes.
    """
    
    def __init__(self):
        # Pattern pour identifier les articles (ex: 711-1, 226-28, etc.)
        self.article_pattern = r'^(\d{3}-\d+(?:-\d+)?(?:\s*[A-Z])?)\s+'
        
        # Patterns pour extraire les métadonnées structurelles
        self.livre_pattern = r'Livre\s+([IVX]+)\s*:\s*([^-\n]+)'
        self.titre_pattern = r'Titre\s+([IVX]+\w*)\s*:\s*([^-\n]+)'
        self.chapitre_pattern = r'Chapitre\s+([IVX]+\w*)\s*:\s*([^-\n]+)'
        self.section_pattern = r'Section\s+([IVX]+\w*)\s*:\s*([^-\n]+)'
        
        # Pattern pour extraire les références légales
        self.loi_pattern = r'LOI\s+n°(\d{4}-\d+)\s+du\s+([^-]+)'
        self.ordonnance_pattern = r'Ordonnance\s+n°(\d{4}-\d+)\s+du\s+([^-]+)'

        # Élements de navigation à retirer des chunks # < Rajouté par XB
        self.navigation_elements = ["Legif.", "Legif", "Plan", "Jp.Judi.", "Jp.Judi", "Jp.Admin.", "Jp.Admin", "Juricaf"] # < Rajouté par XB

        
    def load_and_chunk_code_penal(self, file_path: str) -> List[Document]:
        """
        Charge le PDF du Code pénal et le divise en chunks intelligents.
        
        Args:
            file_path: Chemin vers le fichier PDF du Code pénal
            
        Returns:
            Liste de Documents avec métadonnées enrichies
        """
        # Chargement du PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Concaténation de tout le texte
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # Chunking intelligent
        chunks = self._split_by_articles(full_text)
        
        return chunks
    
    def _split_by_articles(self, text: str) -> List[Document]:
        """
        Divise le texte en chunks basés sur les articles du Code pénal.
        """
        chunks = []
        lines = text.split('\n')
        
        current_chunk = []
        current_metadata = {}
        current_structure = {
            'livre': None,
            'titre': None, 
            'chapitre': None,
            'section': None
        }
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Ignorer les lignes vides
            if not line:
                i += 1
                continue
            
            # Mise à jour de la structure hiérarchique
            self._update_structure(line, current_structure)
            
            # Vérifier si c'est le début d'un nouvel article
            article_match = re.match(self.article_pattern, line)
            
            if article_match:
                # Sauvegarder le chunk précédent s'il existe
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    # Nettoyage du chunk # < Rajouté par XB
                    chunk_text = self._clean_chunk_text(chunk_text) # < Rajouté par XB
                    if chunk_text.strip():
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=current_metadata.copy()
                        ))
                
                # Commencer un nouveau chunk
                article_num = article_match.group(1)
                current_chunk = [line]
                current_metadata = self._extract_article_metadata(
                    article_num, line, current_structure
                )
                
                # Collecter tout le contenu de l'article
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    # Arrêter si on trouve un nouvel article ou une nouvelle structure
                    if (re.match(self.article_pattern, next_line) or 
                        self._is_structural_element(next_line)):
                        break
                        
                    if next_line:  # Ignorer les lignes vides
                        current_chunk.append(next_line)
                    i += 1
                continue
            
            # Si ce n'est pas un article, ajouter à la structure actuelle
            if not self._is_structural_element(line):
                current_chunk.append(line)
            
            i += 1
        
        # Ajouter le dernier chunk s'il existe
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            # Nettoyage du chunk # < Rajouté par XB
            chunk_text = self._clean_chunk_text(chunk_text) # < Rajouté par XB
            if chunk_text.strip():
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=current_metadata
                ))
        
        return chunks
    
    def _update_structure(self, line: str, structure: Dict):
        """Met à jour la structure hiérarchique actuelle."""
        livre_match = re.search(self.livre_pattern, line, re.IGNORECASE)
        if livre_match:
            structure['livre'] = {
                'numero': livre_match.group(1),
                'titre': livre_match.group(2).strip()
            }
            return
        
        titre_match = re.search(self.titre_pattern, line, re.IGNORECASE)
        if titre_match:
            structure['titre'] = {
                'numero': titre_match.group(1),
                'titre': titre_match.group(2).strip()
            }
            return
        
        chapitre_match = re.search(self.chapitre_pattern, line, re.IGNORECASE)
        if chapitre_match:
            structure['chapitre'] = {
                'numero': chapitre_match.group(1),
                'titre': chapitre_match.group(2).strip()
            }
            return
        
        section_match = re.search(self.section_pattern, line, re.IGNORECASE)
        if section_match:
            structure['section'] = {
                'numero': section_match.group(1),
                'titre': section_match.group(2).strip()
            }
    
    def _is_structural_element(self, line: str) -> bool:
        """Vérifie si une ligne est un élément structurel (Livre, Titre, etc.)."""
        patterns = [
            self.livre_pattern,
            self.titre_pattern, 
            self.chapitre_pattern,
            self.section_pattern
        ]
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    def _extract_article_metadata(self, article_num: str, article_line: str, 
                                structure: Dict) -> Dict:
        """
        Extrait les métadonnées d'un article.
        """
        metadata = {
            'article_numero': article_num,
            'type': 'article',
            'source': 'Code pénal français'
        }
        
        # Ajouter la structure hiérarchique
        if structure['livre']:
            metadata['livre_numero'] = structure['livre']['numero']
            metadata['livre_titre'] = structure['livre']['titre']
        
        if structure['titre']:
            metadata['titre_numero'] = structure['titre']['numero']
            metadata['titre_titre'] = structure['titre']['titre']
        
        if structure['chapitre']:
            metadata['chapitre_numero'] = structure['chapitre']['numero']
            metadata['chapitre_titre'] = structure['chapitre']['titre']
        
        if structure['section']:
            metadata['section_numero'] = structure['section']['numero']
            metadata['section_titre'] = structure['section']['titre']
        
        # Extraire les références légales
        loi_match = re.search(self.loi_pattern, article_line)
        if loi_match:
            metadata['loi_numero'] = loi_match.group(1)
            metadata['loi_date'] = loi_match.group(2).strip()
        
        ordonnance_match = re.search(self.ordonnance_pattern, article_line)
        if ordonnance_match:
            metadata['ordonnance_numero'] = ordonnance_match.group(1)
            metadata['ordonnance_date'] = ordonnance_match.group(2).strip()
        
        return metadata
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Nettoie le texte complet d'un chunk en supprimant les éléments indésirables.
        
        Args:
            text: Le texte du chunk à nettoyer
            
        Returns:
            Le texte nettoyé
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Ignorer les lignes vides
            if not line:
                continue

            # Supprimer les éléments de navigation s'ils sont seuls sur la ligne
            if line.strip() in self.navigation_elements: # < Rajouté par XB
                continue

            # Garder toutes les autres lignes
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

# Fonction utilitaire pour utiliser le chunker
def chunk_code_penal(file_path: str) -> List[Document]:
    """
    Fonction pratique pour chunker le Code pénal.
    
    Args:
        file_path: Chemin vers le fichier PDF du Code pénal
        
    Returns:
        Liste de chunks avec métadonnées
    """
    chunker = CodePenalChunker()
    return chunker.load_and_chunk_code_penal(file_path)

# Exemple d'utilisation
if __name__ == "__main__":

    # Chemin vers le PDF du code pénal
    file_path = "../data/decoupage/Code_penal_01.pdf"

    # Splits en utilisant le code généré par Claude.ai 
    all_splits = chunk_code_penal(file_path)

    # Exploration des splits
    print(f"Nombre de splits : {len(all_splits):d}")

    # Afficher quelques exemples
    N = [0, 24, 25, 26, 242]
    for n in N:
        chunk = all_splits[n]
        print(f"\n--- Chunk {n+1} ---")
        print(f"Article: {chunk.metadata.get('article_numero', 'N/A')}")
        print(f"Livre: {chunk.metadata.get('livre_titre', 'N/A')}")
        print(f"Chapitre: {chunk.metadata.get('chapitre_titre', 'N/A')}")
        print(f"Contenu : {chunk.page_content}")
        print(f"Métadonnées complètes: {chunk.metadata}")

    # Liste de tous les articles extraits :
    print("================")
    liste_articles = []
    for n in range(0, len(all_splits)):
        chunk = all_splits[n]
        liste_articles.append(chunk.metadata.get('article_numero', 'N/A'))
    print(*liste_articles, sep="\n")

