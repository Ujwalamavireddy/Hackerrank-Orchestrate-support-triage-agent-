import os
from typing import Dict, List


def load_support_documents(folder_path: str = "support_issues") -> List[Dict[str, str]]:
    """Load all support documents from text files in the corpus folder."""
    documents = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".txt"):
            continue

        path = os.path.join(folder_path, filename)
        with open(path, encoding="utf-8") as file:
            text = file.read().strip()

        if not text:
            continue

        source = os.path.splitext(filename)[0]
        documents.append({"id": source, "source": source, "text": text})

    return documents
