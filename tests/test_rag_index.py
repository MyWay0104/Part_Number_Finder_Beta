from part_finder.rag_index import build_rag_index, search_rag_index


def test_build_rag_index_with_internal_embedding_provider(tmp_path):
    csv_path = tmp_path / "parts.csv"
    index_path = tmp_path / "index.json"
    csv_path.write_text(
        "part_number,part_name,description,equipment_module,vendor_part_number,vendor\n"
        "P2200043,Robot Blade 049,Robot Blade for Endura,Endura,TEL-ROB-0049,TEL\n",
        encoding="utf-8",
    )

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Robot Blade" in text else [0.0, 1.0] for text in texts]

    payload = build_rag_index(csv_path, index_path, embed)
    rows = search_rag_index(index_path, "Robot Blade", embed, top_k=1)

    assert index_path.exists()
    assert len(payload["chunks"]) == 1
    assert rows[0]["part_number"] == "P2200043"
