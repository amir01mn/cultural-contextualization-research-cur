from src.construction import graph_builder


def test_build_graph_nodes_and_edges(sample_triples):
    g = graph_builder.build_graph(sample_triples)
    assert g.number_of_nodes() >= 3
    assert g.number_of_edges() == len(sample_triples)


def test_edges_preserve_provenance(sample_triples):
    g = graph_builder.build_graph(sample_triples)
    edge_data = list(g.edges(keys=True, data=True))
    found = False
    for _s, _o, key, data in edge_data:
        if key == "t_1":
            found = True
            assert data["instance_id"] == "test_001"
            assert data["predicate"] == "removes"
            assert data["generated"] is False
            assert data["source_text"]
    assert found


def test_node_layer_distribution_normalizes(sample_triples):
    g = graph_builder.build_graph(sample_triples)
    view = graph_builder._normalized_node_view(g)
    for _n, data in view.nodes(data=True):
        dist = data["layer_distribution"]
        assert abs(sum(dist) - 1.0) < 1e-6
        assert data["dominant_layer"] in {
            "LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE"
        }


def test_export_all_writes_files(sample_triples, tmp_path):
    g = graph_builder.build_graph(sample_triples)
    paths = graph_builder.export_all(g, "test_kg", outputs_dir=tmp_path)
    for fmt in ("graphml", "json", "edges_csv", "nodes_csv"):
        assert (tmp_path / f"test_kg{_suffix(fmt)}").exists() or (tmp_path).joinpath(
            paths[fmt]).exists()
    # JSON + graphml + csvs must exist regardless of pyvis availability.
    assert (tmp_path / "test_kg.json").exists()
    assert (tmp_path / "test_kg.graphml").exists()
    assert (tmp_path / "test_kg_edges.csv").exists()
    assert (tmp_path / "test_kg_nodes.csv").exists()


def _suffix(fmt: str) -> str:
    return {
        "graphml": ".graphml",
        "json": ".json",
        "edges_csv": "_edges.csv",
        "nodes_csv": "_nodes.csv",
        "html": ".html",
    }[fmt]
