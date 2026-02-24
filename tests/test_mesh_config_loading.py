"""Integration tests for mesh-based config loading.

Tests that config files are parsed correctly and validated.
"""

import pytest
import json
from pathlib import Path
from syssim.network.profiler import load_hierarchy_config, HierarchyConfig
from syssim.network.device_mesh import DeviceMesh


class TestMeshConfigLoading:
    """Test loading mesh-based hierarchy configs."""

    def test_load_perlmutter_config(self):
        """Test loading Perlmutter mesh config."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        # Check top-level fields
        assert config.topology_name == "perlmutter"
        assert "Perlmutter" in config.description
        assert len(config.layers) == 2

        # Check mesh
        mesh = config.get_device_mesh()
        assert mesh.shape == (4, 4)
        assert mesh.dimension_names == ["node", "gpu_in_node"]
        assert mesh.total_ranks == 16

        # Check layers (now a dict)
        assert "intra_node_nvlink" in config.layers
        assert "inter_node_slingshot" in config.layers

        assert config.layers["intra_node_nvlink"].topology_type == "nvlink"
        assert config.layers["intra_node_nvlink"].scope["vary_dims"] == ["gpu_in_node"]
        assert config.layers["intra_node_nvlink"].scope["fix_dims"] == {"node": 0}

        assert config.layers["inter_node_slingshot"].topology_type == "slingshot"
        assert config.layers["inter_node_slingshot"].scope["vary_dims"] == ["node"]
        assert config.layers["inter_node_slingshot"].scope["fix_dims"] == {"gpu_in_node": 0}

    def test_load_dgx_config(self):
        """Test loading DGX A100 mesh config."""
        config_path = Path("examples/configs/dgx_a100_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        # Check mesh (8 GPUs per node)
        mesh = config.get_device_mesh()
        assert mesh.shape == (2, 8)
        assert mesh.dimension_names == ["node", "gpu_in_node"]
        assert mesh.total_ranks == 16

        # Check layers
        assert len(config.layers) == 2
        # num_pairs is optional (defaults to 1)

    def test_load_3d_mesh_config(self):
        """Test loading 3D hierarchical mesh config."""
        config_path = Path("examples/configs/3d_mesh_example.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)

        # Check 3D mesh
        mesh = config.get_device_mesh()
        assert mesh.shape == (2, 4, 4)
        assert mesh.dimension_names == ["rack", "node_in_rack", "gpu_in_node"]
        assert mesh.total_ranks == 32

        # Check 3 layers (now a dict)
        assert len(config.layers) == 3
        assert "intra_node_nvlink" in config.layers
        assert "inter_node_same_rack" in config.layers
        assert "inter_rack" in config.layers

    def test_get_rank_pairs_from_config(self):
        """Test that rank pairs are correctly derived from mesh."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)
        mesh = config.get_device_mesh()

        # Intra-node layer (node=0, vary GPU)
        layer0 = config.layers["intra_node_nvlink"]
        pairs0 = layer0.get_rank_pairs(mesh)
        assert len(pairs0) == 1  # Default num_pairs=1
        assert pairs0[0] == (0, 1)  # GPU 0→1 on node 0

        # Inter-node layer (gpu=0, vary node)
        layer1 = config.layers["inter_node_slingshot"]
        pairs1 = layer1.get_rank_pairs(mesh)
        assert len(pairs1) == 1  # Default num_pairs=1
        assert pairs1[0] == (0, 4)  # Node 0→1, GPU 0

    def test_get_all_ranks_from_config(self):
        """Test that all ranks are correctly derived from mesh."""
        config_path = Path("examples/configs/perlmutter_mesh.json")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_hierarchy_config(config_path)
        mesh = config.get_device_mesh()

        # Intra-node layer: all GPUs on node 0
        layer0 = config.layers["intra_node_nvlink"]
        ranks0 = layer0.get_all_ranks(mesh)
        assert ranks0 == [0, 1, 2, 3]

        # Inter-node layer: GPU 0 on all nodes
        layer1 = config.layers["inter_node_slingshot"]
        ranks1 = layer1.get_all_ranks(mesh)
        assert ranks1 == [0, 4, 8, 12]


class TestMeshConfigValidation:
    """Test config validation catches errors."""

    def test_missing_mesh_field(self, tmp_path):
        """Test that missing mesh field is caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({
            "topology_name": "test",
            "description": "Missing mesh field",
            "layers": [],
            "profiling_params": {}
        }))

        with pytest.raises(ValueError, match="Missing 'mesh' field"):
            load_hierarchy_config(config_file)

    def test_missing_scope_field(self, tmp_path):
        """Test that missing scope field is caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({
            "topology_name": "test",
            "description": "Missing scope field",
            "mesh": {"shape": [2, 2], "dimension_names": ["a", "b"]},
            "layers": {
                "layer1": {
                    "topology_type": "nvlink"
                    # Missing scope field
                }
            },
            "profiling_params": {}
        }))

        with pytest.raises(ValueError, match="Missing 'scope' in layer"):
            load_hierarchy_config(config_file)

    def test_invalid_dimension_in_fix_dims(self, tmp_path):
        """Test that invalid dimension names in fix_dims are caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({
            "topology_name": "test",
            "description": "Invalid dimension in fix_dims",
            "mesh": {"shape": [2, 2], "dimension_names": ["node", "gpu"]},
            "layers": {
                "layer1": {
                    "topology_type": "nvlink",
                    "scope": {
                        "vary_dims": ["gpu"],
                        "fix_dims": {"invalid_dim": 0}  # Invalid dimension
                    }
                }
            },
            "profiling_params": {}
        }))

        with pytest.raises(ValueError, match="not in mesh.dimension_names"):
            load_hierarchy_config(config_file)

    def test_invalid_dimension_in_vary_dims(self, tmp_path):
        """Test that invalid dimension names in vary_dims are caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({
            "topology_name": "test",
            "description": "Invalid dimension in vary_dims",
            "mesh": {"shape": [2, 2], "dimension_names": ["node", "gpu"]},
            "layers": {
                "layer1": {
                    "topology_type": "nvlink",
                    "scope": {
                        "vary_dims": ["invalid_dim"],  # Invalid dimension
                        "fix_dims": {"node": 0}
                    }
                }
            },
            "profiling_params": {}
        }))

        with pytest.raises(ValueError, match="not in mesh.dimension_names"):
            load_hierarchy_config(config_file)

    def test_out_of_bounds_fix_dims_value(self, tmp_path):
        """Test that out-of-bounds fix_dims values are caught."""
        config_file = tmp_path / "bad_config.json"
        config_file.write_text(json.dumps({
            "topology_name": "test",
            "description": "Out of bounds fix_dims value",
            "mesh": {"shape": [2, 2], "dimension_names": ["node", "gpu"]},
            "layers": {
                "layer1": {
                    "topology_type": "nvlink",
                    "scope": {
                        "vary_dims": ["gpu"],
                        "fix_dims": {"node": 5}  # Out of bounds (only 0-1 valid)
                    }
                }
            },
            "profiling_params": {}
        }))

        with pytest.raises(ValueError, match="out of bounds"):
            load_hierarchy_config(config_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
